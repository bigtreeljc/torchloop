import cv2
import tensorflow as tf
import numpy as np
from torchloop.util import fs_utils
import os
import json

#####
# initialize some constants
#####
cap = cv2.VideoCapture(0)
cv2.namedWindow("Window")
model_fname = os.path.join(fs_utils.default_model_dir(), 
    "ssd_mobilenet_v1_coco_2017_11_17", "frozen_inference_graph.pb")
coco_label_file = os.path.join(fs_utils.default_model_dir(), 
    "ssd_mobilenet_v1_coco_2017_11_17", "coco_categories.json")
font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

with tf.gfile.FastGFile(model_fname, mode='rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

####
# load json labels
####
id2label = {}
with open(coco_label_file, 'r') as f:
    labels_list = json.load(f)
for labels in labels_list:
    id2label[labels["id"]] = labels["name"]

#####
# TODO generate a pbtxt file for this
#####
# cvNet = cv2.dnn.readNetFromTensorflow(model_fname)

######
# enter train loop
######
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    while True:
        rect, frame = cap.read()
        #####
        # object detection module
        #####
        # img = frame #cv.imread('example.jpg')
        # rows = img.shape[0]
        # cols = img.shape[1]
        # cvNet.setInput(cv.dnn.blobFromImage(
        #   img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), 
        #   swapRB=True, crop=False))
        # cvOut = cvNet.forward()
    
        # Read and preprocess an image.
        img = frame #cv.imread('example.jpg')
        # if img is not None:
        #     print("can't load img, retrying")
        #     continue
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    
        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name(
                          'num_detections:0'),
                        sess.graph.get_tensor_by_name(
                          'detection_scores:0'),
                        sess.graph.get_tensor_by_name(
                          'detection_boxes:0'),
                        sess.graph.get_tensor_by_name(
                          'detection_classes:0')],
                        feed_dict={'image_tensor:0': \
                            inp.reshape(1, inp.shape[0], 
                              inp.shape[1], 3)})
        # print(type(out)) # list
        # print("out[0] is {}".format(out[0]))
        #####
        # getting those rectangles one the image
        # Visualize detected bounding boxes.
        #####
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            
            if score > 0.3:
                ####
                # print bbox to be draw
                ####
                class_name = id2label[classId]
                print("------------------------")
                print(classId)
                print(class_name)
                print(score)
                print(bbox)
                print("---------------------------")

                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv2.rectangle(img, (int(x), int(y)), 
                    (int(right), int(bottom)), (125, 255, 51), 
                    thickness=2)
                bottomLeftCornerOfText = (int(x), int(y)+20)
                cv2.putText(img, "{} {}".format(classId, class_name), 
                    bottomLeftCornerOfText, font, 1, 
                    (200, 200, 200), 2, cv2.LINE_AA)
                # cv2.putText(img, "classid {}".format(classId), 
                #     (10, 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #####
        # draw the image
        #####
        if rect:
            # print("showing image of web camp")
            # print(type(frame))
            cv2.imshow('Window', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
