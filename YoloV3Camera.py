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
camera = cv2.VideoCapture(1)

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

colours = np.random.randint(0,255, size = (len(labels),3), dtype='uint8')

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
    output = yolo.forward(output_layers)
    end = time.time()
    print(end-start)

# Step 6: Getting bounded boxes
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output:
        #first we are iterating through all the output layers
        for detected_objects in result:
            # Second we get numpy array(detected_objects)
            # that contain coordinates at first four indexs
            # then scores of all the labels
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if (confidence_current > probability_minimum):
                box_current = detected_objects[0:4] * np.array(
                    [w,h,w,h])
                x_center,y_center,box_width,box_height = box_current
                x_min = int(x_center - (box_width/2))
                y_min = int(y_center - (box_height/2))

                bounding_boxes.append([x_min,y_min,int(box_width),int(box_height)])

                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

# Step 7: Non-Maximum suppression
    results = cv2.dnn.NMSBoxes(bounding_boxes,confidences,probability_minimum,
                               threshold)

# Step 8: Adding bounding boxes on image
    if len(results)>0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist()
            cv2.rectangle(frame,(x_min,y_min),
                          (x_min+box_width,y_min+box_height),colour_box_current,2)

            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

# Step 9: Showing processed frames

    cv2.namedWindow('YOLO v3 real time detections', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 real time detections',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


