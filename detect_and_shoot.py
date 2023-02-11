import os
import json
from time import sleep, process_time, time
from threading import Thread
import sys
import argparse
from pathlib import Path
from datetime import datetime

import RPi.GPIO as GPIO
import cv2
import numpy as np
from cloud_utils import report_activity, notify_heartbeat, notify_hibernation


GPIO.setmode(GPIO.BOARD)
 
motor = 16    # Motor control pin
lights = 40   # Lights control pin

GPIO.setup(motor,GPIO.OUT)
GPIO.setup(lights,GPIO.OUT)

duration = 3
cooldown = 10
duration_left = 0
cooldown_left = 0


# launch thread for uploading files to cloud
def upload_to_cloud(file, clip_detections):
    t = Thread(target=report_activity, args=(file, clip_detections))
    t.start()


# start heartbeat
def heartbeat_loop():
    while True:
        notify_heartbeat()
        sleep(60)

# gun control loop
def gun_control_loop():
    global duration_left
    global cooldown_left

    while remaining_frames != 0:
        if duration_left > 0:
            turn_on_pump()
            duration_left -= 1
        else:
            turn_off_pump()
            if cooldown_left > 0:
                print("cooling down gun")
                cooldown_left -= 1

        sleep(1)


def turn_on_pump():
    sleep(0.1)
    GPIO.output(motor,GPIO.HIGH)
    sleep(0.1)


def turn_off_pump():
    sleep(0.1)
    GPIO.output(motor,GPIO.LOW)
    sleep(0.1)


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(box0, box1):
    dx = max(box0[0], box1[0]) - min(box0[2], box1[2])
    dy = max(box0[1], box1[1]) - min(box0[3], box1[3])

    return dx * dy


def union(box0, box1):
    return box_area(box0) + box_area(box1) - intersection(box0, box1)


def intersection_over_union(box0, box1):
    return intersection(box0, box1) / union(box0, box1)


def intersection_over_target(target, box):
    return intersection(target, box) / box_area(target)


def go_to_sleep():
    turn_off_lights()
    notify_hibernation()
    bashCommand = "sudo shutdown -h now"
    os.system(bashCommand)


def turn_on_lights():
    sleep(0.1)
    GPIO.output(lights,GPIO.HIGH)
    sleep(0.1)


def turn_off_lights():
    sleep(0.1)
    GPIO.output(lights,GPIO.LOW)
    sleep(0.1)


def format_detections(clip_detections, names):
    formatted_detections = {}

    for idx, frame_detections in enumerate(clip_detections):
        if frame_detections is not None:
            frame_detections_list = frame_detections.tolist()
            # Replace class index with class name
            for idxx, detection in enumerate(frame_detections_list):
                frame_detections_list[idxx][5] = names[int(detection[5])]
            formatted_detections[str(idx)] = frame_detections_list
        else:
            formatted_detections[str(idx)] = []

    return formatted_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_alive", action="store_true")
    parser.add_argument("--peaceful", action="store_true")
    parser.add_argument("--record_frames", type=int, default=-1)

    args = parser.parse_args()

    turn_off_pump()
    turn_on_lights()

    if args.peaceful:
        print("Running in peaceful mode")
    else:
        print("Running in war mode")

    detection_hits = 0
    
    from yolov5.detect import Detector
    detector = Detector(source="picamera", imgsz=640, weights="yolov5s_filtered_old.engine")

    #from yolov7.detect import Detector
    #detector = Detector(source="picamera", imgsz=640, weights="best.pt", half=False, dnn=False)

    if not args.peaceful:
        # target config
        target_width = 200
        target_height = 200
        target_color = (255, 0, 0)
        target_p1, target_p2 = None, None
        target_lw = 3

    # initialize background substractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    minimum = 300

    # output settings
    save_feed = False
    save_clips = True
    vid_path, vid_writer = None, None
    save_path = "output/picamera.mp4"

    # configure program lifespan
    frame_buffer = []
    empty_frames_count = 0
    no_activity_count = 0
    remaining_frames = args.record_frames

    if not args.peaceful:
        # start gun control thread
        gun_thread = Thread(target=gun_control_loop)
        gun_thread.start()

    # start heartbeat thread
    heartbeat_thread = Thread(target=heartbeat_loop)
    heartbeat_thread.start()

    # output path
    output_clips_path = Path("output/clips")
    output_clips_path.mkdir(parents=True, exist_ok=True)
    output_detections_path = Path("output/detections")
    output_detections_path.mkdir(parents=True, exist_ok=True)

    detect_motion = False

    clip_detections = []

    frame_counter = 0
    log_frequency = 30

    while True:
        if frame_counter == 0:
            time_marker = time()
        elif frame_counter % log_frequency == 0:
            time_delta = time() - time_marker
            print(f"Processing frames at {'%.1f' % (frame_counter / time_delta)} fps ({frame_counter} in {('%.1f' % time_delta)} s)")
            time_marker = time()
            frame_counter = 0

        if no_activity_count >= 60 and not args.keep_alive:
            print("Seems like nobody is here. Going to sleep.")
            go_to_sleep()

        detections, im0 = detector.detect_once(conf_thres=0.75, grayscale=True)

        if not args.peaceful:
            # target setup
            if (target_p1, target_p2) == (None, None):
                image_shape = im0.shape
                target_p1 = (int((image_shape[0] - target_width)/ 2), int((image_shape[1] - target_height) / 2))
                target_p2 = (int((image_shape[0] + target_width)/ 2), int((image_shape[1] + target_height) / 2))
        
        if detect_motion:
            # detect motion
            fgmask = fgbg.apply(im0)
            contours,_ = cv2.findContours(fgmask, mode= cv2.RETR_TREE, method= cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours,key=cv2.contourArea,reverse= True)

            motion_detected = False
            
            #########################
            ### AQUI HAY UN ERROR!!!
            ######################### solo se cuenta el primer contorno valido
            filtered_contours = []

            for cnt in contours:
                if cv2.contourArea(cnt) < minimum:
                        continue
                
                (x, y, w, h) = cv2.boundingRect(cnt)
                motion_contour_box = (x, y, x + w, y + h)
                # cv2.rectangle(im0, (x, y), (x + w, y + h), (255, 255, 255), 1)
                filtered_contours.append(motion_contour_box)
        if len(detections[0]) > 0:
            # initialize activity flag
            activity_detected = False

            if (detect_motion and len(filtered_contours) > 0) or not detect_motion:
                frame_detections = []
                for detection in detections[0]:
                    detection = detection.cpu().numpy()
                    print(detection)
                    detection_intersects_motion = False
                    if detect_motion:
                        for motion_contour_box in filtered_contours:
                            if intersection_over_target(motion_contour_box, detection) > 0.3:
                                detection_intersects_motion = True
                                break
                    if detection_intersects_motion or not detect_motion:
                        # Add frame detections only once
                        frame_detections.append(detection)
                        # Reset sleep counter
                        activity_detected = True
                        no_activity_count = 0

                        if not args.peaceful:
                            # if detection class is raton and there is enough overlap between target and detection and there is enough overlap between detection and motion contour
                            if int(detection[5]) == 0 and (intersection_over_target((target_p1 + target_p2), detection) > 0.3 or box_area(detection) / (image_shape[0] * image_shape[1]) > 0.3) and cooldown_left == 0:
                                print("Feuer!")
                                duration_left = 3
                                target_color = (0, 0, 255)
                                cooldown_left = cooldown
                                
                                # Draw activated target
                                # cv2.rectangle(im0, target_p1, target_p2, target_color, thickness=target_lw, lineType=cv2.LINE_AA)

                        if not args.peaceful:
                            target_color = (255, 0, 0)

                if activity_detected:
                    print("Activity detected!")
                    if save_clips:
                        frame_buffer.append(im0)
                        empty_frames_count = 5

                if len(frame_detections) > 0:
                    clip_detections.append(np.stack(frame_detections))
        else:
            no_activity_count += 1

            if len(frame_buffer) > 0 and save_clips:
                if empty_frames_count > 0:
                    frame_buffer.append(im0)
                    clip_detections.append(None)
                    empty_frames_count -= 1
                else:
                    print("saving video clip")
                    # Save clip
                    fps, w, h = 1, im0.shape[1], im0.shape[0] # 1 fps

                    dt = datetime.now()
                    clip_save_path = os.path.join(output_clips_path, f'{str(dt).split(".")[0].replace(" ", "_")}.mp4')
                    clip_writer = cv2.VideoWriter(clip_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    for frame in frame_buffer:
                        clip_writer.write(frame)

                    clip_writer.release()
                    frame_buffer = []

                    print("Reporting activity to cloud...")

                    response = upload_to_cloud(clip_save_path, clip_detections=format_detections(clip_detections, detector.names))
                    clip_detections = []

        if remaining_frames >= 0:
            if remaining_frames > 0:
                # Save results (image with detections)
                if vid_writer is None:
                    fps, w, h = 1, im0.shape[1], im0.shape[0]
                    #save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                res = vid_writer.write(im0)
                remaining_frames -= 1
            else:
                vid_writer.release()

                if not args.peaceful:
                    gun_thread.join()
                
                detector.source.video_capture.release()
                turn_off_pump()
                turn_off_lights()
                break
        
        frame_counter += 1
