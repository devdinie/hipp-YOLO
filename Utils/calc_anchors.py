'''
Created on Feb 20, 2017
@author: jumabek
'''
from os import listdir
from os.path import isfile, join
import argparse
from cv2 import kmeans
#import cv2
import numpy as np
import sys
import os
import shutil
import random
import math
from sklearn.cluster import KMeans
width_in_cfg_file = 144.
height_in_cfg_file = 144.


keras_path  = os.path.join("/data/chamal/projects/swapna/YOLO_22v2/hipp-YOLO/2_Training/src/keras_yolo3")

Data_Folder = os.path.join("/data/chamal/projects/swapna/YOLO_22v2/hipp-YOLO","Data")
Annot_Folder= os.path.join(Data_Folder, "Source_Images", "Training_Images", "annotations")
Annot_csv = os.path.join(Annot_Folder, "annotations-export.csv")
YOLO_filename = os.path.join(keras_path,"model_data", "data_train.txt")

def calc_anchors(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-filelist', default = YOLO_filename,
                        help='path to filelist\n' )
    parser.add_argument('-output_dir', default =os.path.join(keras_path,"model_data"), type = str,
                        help='Output anchor directory\n' )
    parser.add_argument('-num_clusters', default = 9, type = int,
                        help='number of clusters\n' )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    f = open(args.filelist)
    print(f)
    lines = [line.rstrip('\n') for line in f.readlines()]
    annotation_dims = []
    annotation_cents= []
    #size = np.zeros((1,1,3))
    yo_w = 128 ; yo_h = 128 ; yo_d = 128
    num_clusters = 4
    
    for line in lines:
        line = line.replace('.nii','.txt')
        
        f2 = line.split(',')[0]
        
        w  = round((((float(line.split(',')[4]) - float(line.split(',')[1])))/144)*yo_w,3)
        h  = round((((float(line.split(',')[5]) - float(line.split(',')[2])))/144)*yo_h,3)
        d  = round((((float(line.split(',')[6]) - float(line.split(',')[3])))/144)*yo_d,3)
        
        annotation_dims.append(tuple(map(float,(w,h,d))))
        
    #annotation_cents = np.array(annotation_cents)
    annotation_dims  = np.array(annotation_dims)
    print(annotation_dims)
    eps = 0.005
            
    anchor_file = os.path.join(args.output_dir,'anchors%d.txt'%(num_clusters))
    
    kmeans = KMeans(num_clusters, n_init=20).fit(annotation_dims)
    anchors = kmeans.cluster_centers_
    
    
    with open(anchor_file,"a") as file:
        anchor_file.write(anchors)
        
    print('anchors\n----------------\n', anchors)
            
    
   