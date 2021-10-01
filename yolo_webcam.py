#!/usr/bin/python3

import cv2
import numpy as np
# import glob
# import random
import time
import os

cam = cv2.VideoCapture(-1) #0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
CV_LOAD_IMAGE_GRAYSCALE = 0

# Load Yolo

# Number of channels, set to 3 for color images
ch_num = 1

if ch_num == 3:
    # # 3 channels # #
    net = cv2.dnn.readNet("yolo_weights/yolov4-tiny_training_bw1_last.weights", "config/yolov4-tiny_testing_3chan.cfg")

    # # # 3 Channels for YoloV3 # #
    # net = cv2.dnn.readNet("yolo_weights/yolov3_training_last.weights", "config/yolov3_testing.cfg")

    # # # 3 Channels With no pretrained weights # #
    # net = cv2.dnn.readNet("yolo_weights/yolov4-tiny_training_3ch_no_pretraining.weights", "config/yolov4-tiny_testing_3chan.cfg")
else:
    # # B&W # #
    net = cv2.dnn.readNet("yolo_weights/yolov4-tiny_training_bw1_last.weights", "config/yolov4-tiny_testing_bw1.cfg")


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
    dim = (416,416)
    colors = np.random.uniform(0, 255, size=(len(classes)))

    if ch_num == 3:
        # # 3 Channels # #
        img = imagen
        color_img = imagen
    else:
        # # B&W # #
        img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # Use only 1 channel in B&W mode
        color_img = imagen

    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # color_img = cv2.resize(color_img, dim, interpolation = cv2.INTER_AREA)

    if ch_num == 3:
        # # 3 Channels # #
        height, width, channels = img.shape
    else:
        # # B&W # #
        height, width = img.shape
        channels = 1

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                # Object detected
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
    num_peces = 0

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            num_peces += 1
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(color_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(color_img, label, (x, y + 30), font, 2, color, 2)
    
    # print(f'Hay {num_peces} peces')
    cv2.putText(color_img, 'Count:'+str(num_peces), (30, 30), font, 1, (255,255,255), 2)

    return color_img


while True:
    # imagen = cv2.imread(img_path) #, CV_LOAD_IMAGE_GRAYSCALE) # Test with image in directory test_images/

    ## read frames
    ret, imagen = cam.read()
    t0 = time.time()
    ## Fish Detection
    boxed_image = detect_fish(imagen)
    t1 = time.time()
    cv2.putText(boxed_image, 'FPS:'+str(int(1/(t1-t0))), (100, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
    cv2.imshow('Yolo in action', boxed_image)
    ## press q or Esc to quit
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

cam.release()
cv2.destroyAllWindows()

# detect_fish(imagen)
