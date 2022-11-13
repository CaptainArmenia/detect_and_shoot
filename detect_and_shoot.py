import RPi.GPIO as GPIO
from time import sleep
from threading import Thread
import sys
import argparse
import cv2
from pathlib import Path
import os
from datetime import datetime


GPIO.setmode(GPIO.BOARD)
 
motor = 16    # Motor control pin
lights = 40   # Lights control pin

GPIO.setup(motor,GPIO.OUT)
GPIO.setup(lights,GPIO.OUT)

duration = [0]

def shoot():
    while True:
        if duration[0] > 0:
            GPIO.output(motor,GPIO.HIGH)
            
            duration[0] -= 1
        else:
            turn_off_pump()

        sleep(1)


def turn_off_pump():
    sleep(0.1)
    GPIO.output(motor,GPIO.LOW)
    sleep(0.1)


def intersection_over_target(target, box):
    dx = max(target[0], box[0]) - min(target[2], box[2])
    dy = max(target[1], box[1]) - min(target[3], box[3])

    return dx * dy / ((target[2] - target[0]) * (target[3] - target[1]))


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def go_to_sleep():
    turn_off_lights()

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

turn_off_pump()
turn_on_lights()

detection_hits = 0
 
from yolov5.detect import Detector
detector = Detector(source="picamera", imgsz=640, weights="models/best_yolov5s.pt", half=False, dnn=False)

# target config
target_width = 200
target_height = 200
target_color = (255, 0, 0)
target_p1, target_p2 = None, None
target_lw = 3

# output settings
save_img = False
save_clips = True
vid_path, vid_writer = None, None
save_path = "output/picamera.mp4"

# start shooting thread
t = Thread(target=shoot)
t.start()

frame_buffer = []
empty_frames_count = 0
no_activity_count = 0

while True:
    detections, im0 = detector.detect_once(conf_thres=0.6, grayscale=True)

    if no_activity_count >= 60:
        print("Seems like there is nobody here. Going to sleep.")
        go_to_sleep()

    # target setup
    if (target_p1, target_p2) == (None, None):
        image_shape = im0.shape
        target_p1 = (int((image_shape[0] - target_width)/ 2), int((image_shape[1] - target_height) / 2))
        target_p2 = (int((image_shape[0] + target_width)/ 2), int((image_shape[1] + target_height) / 2))

    for detection in detections:
        if len(detection) > 0:
            
            # Reset sleep counter
            no_activity_count = 0

            if int(detection[0][5]) == 0 and (intersection_over_target((target_p1 + target_p2), detection[0]) > 0.8 or box_area(detection[0]) / (image_shape[0] * image_shape[1]) > 0.3):
                print("Feuer!")
                duration[0] = 3
                target_color = (0, 0, 255)
            
            if save_clips:
                # Draw target
                cv2.rectangle(im0, target_p1, target_p2, target_color, thickness=target_lw, lineType=cv2.LINE_AA)
                frame_buffer.append(im0)
                empty_frames_count = 5

            target_color = (255, 0, 0)
           
        else:
            no_activity_count += 1

            if len(frame_buffer) > 0 and save_clips:
                if empty_frames_count > 0:
                    cv2.rectangle(im0, target_p1, target_p2, target_color, thickness=target_lw, lineType=cv2.LINE_AA)
                    frame_buffer.append(im0)
                    empty_frames_count -= 1
                else:
                    print("saving video clip")
                    # Save clip
                    fps, w, h = 1, im0.shape[1], im0.shape[0] # 1 fps

                    dt = datetime.now()
                    clip_save_path = f'output/clips/{str(dt).split(".")[0].replace(" ", "_")}.mp4'
                    clip_writer = cv2.VideoWriter(clip_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    for frame in frame_buffer:
                        clip_writer.write(frame)

                    clip_writer.release()
                    frame_buffer = []

    if save_img:
        # Draw target
        cv2.rectangle(im0, target_p1, target_p2, target_color, thickness=target_lw, lineType=cv2.LINE_AA)

        # Save results (image with detections)
        if vid_writer is None:
            fps, w, h = 30, im0.shape[1], im0.shape[0]
            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(im0)
