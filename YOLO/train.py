import os
import sys
import shutil

import augment
import argparse
import settings
import numpy as np
import pandas as pd
import SimpleITK as sitk

from yolo_model import init_yolo_model
from preprocess import preprocess_data
from createjson import create_json_file

if __name__ == "__main__":
        
        #region create yolo inputs and annotations
        
        #region initialize directories
        datainput_dir= os.path.join(settings.data_dir,settings.augdata_dname)
        datanet1_dir = os.path.join(settings.data_dir,settings.net1data_dname)
        datanet2_dir = os.path.join(settings.data_dir,settings.net2data_dname)
        log_dir = os.path.join("logs")
        
        annot_csv   = os.path.join(datainput_dir,"yolo_inputs","annotations.csv")
        train_txt   = os.path.join(datainput_dir,"yolo_inputs","data_train.txt")
        classes_txt = os.path.join(datainput_dir,"yolo_inputs", "classes.txt")

        config_fname  = os.path.join("model_files","yolov3.cfg")
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
        """
        #endregion preprocessing
        #endregion create yolo inputs and annotations
         
        #region create model
        init_yolo_model(config_fname, weights_fname,outmodel_fname)
        #endregion create model
        