import RPi.GPIO as GPIO
from time import sleep
from threading import Thread
import sys

from yolov7_raspberry.detect import Detector

GPIO.setmode(GPIO.BOARD)
 
Motor1 = 16    # Input Pin
Motor2 = 18    # Input Pin
Motor3 = 22    # Enable Pin
 
GPIO.setup(Motor1,GPIO.OUT)
GPIO.setup(Motor2,GPIO.OUT)
GPIO.setup(Motor3,GPIO.OUT)

detection_hits = 0

def shoot():
    GPIO.output(Motor1,GPIO.LOW)
    GPIO.output(Motor2,GPIO.HIGH)
    GPIO.output(Motor3,GPIO.HIGH)
 
    sleep(3)

    GPIO.output(Motor1,GPIO.LOW)
    GPIO.output(Motor2,GPIO.LOW)
    GPIO.output(Motor3,GPIO.LOW)
 
# def detect():

detector = Detector()

while True:
    detections = detector.detect_once()
    for detection in detections:
        if len(detection) > 0 and int(detection[0][5]) == 0:
            shoot()
    print(f"detection: {detections}")
    

GPIO.cleanup()
