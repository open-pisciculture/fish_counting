#!/usr/bin/python3

import cv2
import numpy as np
import glob
import random
import os

cam = cv2.VideoCapture(-1) #0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
CV_LOAD_IMAGE_GRAYSCALE = 0

# Load Yolo
net = cv2.dnn.readNet("yolo_weights/yolov4-tiny_training_bw3_last.weights", "config/yolov4-tiny_testing.cfg")
# net = cv2.dnn.readNet("yolo_weights/yolov3_training_last.weights", "config/yolov3_testing.cfg")

# Name custom object
classes = ["fish"]

# Images path
cwd_0 = os.getcwd()
img_folder_path = os.path.join(cwd_0, 'test_images')
img_path = os.path.join(img_folder_path, 'img1.jpg')
# print(img_folder_path)
# images_path = glob.glob(r"{}/*.jpg".format(img_folder_path))
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_fish(imagen):
    colors = np.random.uniform(0, 255, size=(len(classes)))
    img = imagen
    # img = cv2.resize(imagen, None, fx=0.4, fy=0.4)
    print(img.shape)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    peces = 0

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print('####### DETECTION #######')
                print(detection, '\n')
                peces += 1
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print('Hay {} peces'.format(peces))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

    return img


while True:
    ## read frames
    ret, imagen = cam.read()
    # print(img_path)

    # imagen = cv2.imread(img_path, CV_LOAD_IMAGE_GRAYSCALE) # Test with image in directory test_images/
    # imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    ## Fish Detection
    boxed_image = detect_fish(imagen)
    cv2.imshow('Yolo in action', imagen)
    ## press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cam.release()
cv2.destroyAllWindows()

# detect_fish(imagen)