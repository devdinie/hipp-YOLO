"""
# REFERENCES:
# Original darknet kmeans and related code:
# - https://github.com/AlexeyAB/darknet/blob/master/scripts/kmeansiou.c
# Other references:
# - https://github.com/lars76/kmeans-anchor-boxes
# - https://github.com/xuannianz/EfficientDet/blob/master/utils/anchors.py
# - https://github.com/decanbay/YOLOv3-Calculate-Anchor-Boxes/blob/master/YOLOv3_get_anchors.py
"""

import os
import copy
import math

import settings
import numpy as np
import matplotlib.pyplot as plt

img_size = settings.img_size[0]
no_clusters = 3

def iou(box, clusters):
        
        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        z = np.minimum(clusters[:, 2], box[2])
        
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0 or np.count_nonzero(z == 0) > 0:
                raise ValueError("Box has no area")
        
        intersection = x * y * z
        box_area = box[0] * box[1] * box[2]
        cluster_area = clusters[:, 0] * clusters[:, 1] * clusters[:, 2]

        iou_ = intersection / (box_area + cluster_area - intersection)
        print(iou_)
        return iou_


def avg_iou(boxes, clusters):
        for i in range(boxes.shape[0]):
                avg_iou_ = np.mean([np.max(iou(boxes[i], clusters))])
        return avg_iou_

def kmeans(boxes, k=no_clusters, dist=np.median):
        
        no_boxes = boxes.shape[0]
        
        distances = np.empty((no_boxes, k))
        last_clusters = np.zeros((no_boxes,))
        
        np.random.seed()
        clusters = boxes[np.random.choice(no_boxes, k, replace=False)]
        
        while True:
                for box_no in range(no_boxes):
                        distances[box_no] = 1 - iou(boxes[box_no], clusters)
                
                nearest_clusters = np.argmin(distances, axis=1)
                
                if (last_clusters == nearest_clusters).all():
                        break
                for cluster in range(k):
                        clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
                
                last_clusters = nearest_clusters
        return clusters
                        
def get_anchors(annotations):
        
        annotations_yolo = list()
        annot_file = open(annotations,"r")
        
        with annot_file as file:
                for line in file:
                        line_arr = line.strip('\n').split(",")
                        dim_arr  = copy.copy(line_arr[1:len(line_arr)]) 

                        width  = round((float(dim_arr[3]) - float(dim_arr[0])),3)
                        height = round((float(dim_arr[4]) - float(dim_arr[1])),3)
                        depth  = round((float(dim_arr[5]) - float(dim_arr[2])),3)

                        annotations_yolo.append([width, height, depth])
        annot_file.close()

        kmeans_output = kmeans(np.array(annotations_yolo))
        
        #print("Accuracy: {:.2f}%".format(avg_iou(np.array(annotations_yolo), kmeans_output) * 100))
        #print("Boxes:\n {}".format(kmeans_output))
        
        #ratios = np.around(kmeans_output[:, 0] / kmeans_output[:, 1], decimals=3).tolist()
        #print("Ratios:\n {}".format(sorted(ratios)))
        
        anchor_file = open(os.path.join("model_files","anchors.txt"),"a")
        with anchor_file as file: 
                for anchor in kmeans_output:
                        anchor_file.write(np.array2string(anchor.astype(int),separator=',').strip('[]'))
                        anchor_file.write(' ')
        anchor_file.close()
        print("Custom anchors saved to ",annotations )
        