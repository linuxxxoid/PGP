#!/usr/bin/env python3
"""Usage: create-video <images>..."""
import sys
import cv2  # $ pip install opencv-python

frames = sys.argv[1:]          # paths to images in order
frame = cv2.imread(frames[0])  # get size from the 1st frame
writer = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),   # codec
    25.0,  # fps
    (frame.shape[1], frame.shape[0]),  # width, height
    isColor=len(frame.shape) > 2)
for frame in map(cv2.imread, frames):
    writer.write(frame)
writer.release()
cv2.destroyAllWindows()