# What we are going to do ->
# reading stream from camera -> Loading Yolo net
# -> Read frame in loop -> Getting blob of the frame
# -> Implementing forward pass -> getting bounding boxes
# -> Non maximum suppression -> Drawing bounded boxes with labels
# -> Showing processed frames

import cv2
import numpy as np
import time


# Step 1: Reading stream
camera = cv2.VideoCapture()

h, w = None, None #Predeclare for height and width of frame

# Step 2: Loading yolov3 Net

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]
# print(labels)
yolo = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                  'yolo-coco-data/yolov3.weights')

all_layers = yolo.getLayerNames()

output_layers = \
    [all_layers[i-1] for i in yolo.getUnconnectedOutLayers()]
# print(output_layers)

probability_minimum = 0.5
threshold = 0.3

colours = np.random.randint(0,255, size = (labels,3), dtype='uint8')

# Step 3: Reading frames from the loop
while True:
    ret, frame = camera.read()

    if w is None or h is None:
        h, w = frame.shape[:2]

# Step 4: Implementing blob on image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416)
                                 , swapRB = True, crop = False)

# Step 5: Forward Pass
    yolo.setInput(blob)
    start = time.time()
    result = yolo.forward(output_layers)
    end = time.time()
    print(end-start)

# Step 6: Getting bounded boxes


