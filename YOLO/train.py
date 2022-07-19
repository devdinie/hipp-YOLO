import os
import sys
import shutil

import augment
import argparse
import settings
import numpy as np
import pandas as pd
import SimpleITK as sitk

from preprocess import preprocess_data
from createjson import create_json_file
from yolo_model import init_yolo_model, create_yolo_model

if __name__ == "__main__":
        
        #region create yolo inputs and annotations
        
        #region initialize directories
        datainput_dir= os.path.join(settings.data_dir,settings.augdata_dname)
        datanet1_dir = os.path.join(settings.data_dir,settings.net1data_dname)
        datanet2_dir = os.path.join(settings.data_dir,settings.net2data_dname)
        log_dir = os.path.join("logs")
        
        annot_csv   = os.path.join(datainput_dir,"yolo_inputs","annotations.csv")
        train_txt   = os.path.join(datainput_dir,"yolo_inputs","data_train.txt")
        classes_txt = os.path.join(datainput_dir,"yolo_inputs", "data_classes.txt")
        
        config_fname  = os.path.join("model_files","yolov3.cfg")
        anchors_fname = os.path.join("model_files","anchors.txt")
        weights_fname = os.path.join("model_files","yolov3.weights")
        outmodel_fname= os.path.join("model_files", "yolov3.h5")
        
        if not os.path.exists("model_files"): os.mkdir("model_files")
        if not os.path.exists("logs"): os.mkdir("logs")
        #endregion initialize directories

        #region augmentation
        """
        if settings.augment:
                augment.augment_data(data_dir=settings.data_dir, augtypes_in="n",output_dir=datainput_dir)        
        else:
                augment.augment_data(data_dir=settings.data_dir,augtypes_in = None, output_dir=datainput_dir)
        """
        #endregion augmentation
        
        #region preprocessing
        """
        create_json_file(datainput_dir)
        preprocess_data(1, settings.data_dir)
        
        create_json_file(datanet1_dir)
        create_json_file(datanet2_dir)
        
        #endregion preprocessing
        #endregion create yolo inputs and annotations
        """
        #region init model
        init_yolo_model(config_fname, weights_fname,outmodel_fname)
        #endregion init model
        
        with open(classes_txt) as file: class_names = file.readlines()
        class_names = [c for c in class_names]
        no_classes = len(class_names)
        
        
        if os.path.exists(anchors_fname):
                with open(anchors_fname) as file: anchors_ = file.readline()
                anchors_ = anchors_.strip(" ").split(" ")
                anchors_ = [float(x) for anchor in anchors_ for x in anchor.split(",")]
                no_anchors = int(len(anchors_)/3) ; anchors = list()
                for i in range(0,no_anchors):
                        anchors.append(anchors_[(3*i):(3*i)+3])
        
        input_shape = settings.img_size
        epoch1, epoch2 = settings.epochs, settings.epochs
        
        
        #region create model
        model = create_yolo_model(input_shape, anchors, no_classes, load_pretrained=True, 
                                  freeze_body=2, weights_path=os.path.join("model_data","yolov3.weights"))
        #endregion create model
        