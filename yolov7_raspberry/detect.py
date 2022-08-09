import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from yolov7_raspberry.models.experimental import attempt_load
from yolov7_raspberry.utils.datasets import LoadImages, LoadPiCamera
from yolov7_raspberry.utils.general import check_img_size, non_max_suppression,  \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7_raspberry.utils.plots import plot_one_box
from yolov7_raspberry.utils.torch_utils import time_synchronized, TracedModel


class Detector:

    def __init__(self, save_img=False, view_img=False, save_txt=False, trace=True, source="picamera", weights="models/yolov7-tiny-320-ratones.pt", imgsz=320, augment=False, agnostic_nms=False):
        self.save_img = save_img
        self.view_img = view_img
        self.save_txt = save_txt
        self.trace = trace
        self.source = source
        self.weights = weights
        self.imgsz = imgsz
        self.name = "dummy"
        self.project = "dummyProject"
        self.augment = augment
        self.img_size = imgsz
        self.agnostic_nms = agnostic_nms

        # Directories
        self.save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=True))  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = "cpu"

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, self.img_size)

        # Set Dataloader
        vid_path, vid_writer = None, None
        if source == "picamera":
            view_img = False
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadPiCamera(img_size=imgsz, stride=stride)
            self.bs = len(self.dataset)  # batch_size
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect_once(self, conf_thres=0.25, iou_thres=0.45):
        
        t0 = time.time()
        path, img, im0s, vid_cap = self.dataset.__next__()
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if self.source == "picamera":  # batch_size >= 1
                p, s, im0, frame = path, '%g: ' % i, im0s.copy(), self.dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.view_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}s) NMS')

            # Save results (image with detections)
            if self.save_img:
                if self.dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        print(f"saved first frame with shape ({w}, {h}) at {fps} fps in {save_path}")
                    vid_writer.write(im0.copy())
                    print(f"saved frame with shape ({im0.shape[1]}, {im0.shape[0]})")

        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''

        print(f'Done. ({time.time() - t0:.3f}s)')
        return pred
