import os
import sys
import shutil
from tkinter import N

import numpy as np
import augment
import argparse
import settings
import pandas as pd
import SimpleITK as sitk

from createjson import create_json_file
from preprocess import preprocess_data

sys.path.append("/data/chamal/projects/swapna/YOLO_22v2/hipp-YOLO/Utils/")
from Convert_Format import convert_annot_csv_to_yolo
#region initialize directories
keras_path  = os.path.join("/data/chamal/projects/swapna/YOLO_22v2/hipp-YOLO/2_Training/src/keras_yolo3")

Data_Folder = os.path.join(os.path.dirname(__file__),"..","Data")
Annot_Folder= os.path.join(Data_Folder, "Source_Images", "Training_Images", "annotations")
Annot_csv = os.path.join(Annot_Folder, "annotations-export.csv")
YOLO_filename = os.path.join(Annot_Folder, "data_train.txt")

model_folder = os.path.join(Data_Folder, "Model_Weights")
classes_filename = os.path.join(model_folder, "data_classes.txt")

if not os.path.exists(Data_Folder):
        os.mkdir(Data_Folder)
if not os.path.exists(os.path.join(Data_Folder,"Source_Images")): 
        os.mkdir(os.path.join(Data_Folder,"Source_Images"))
if not os.path.exists(model_folder): 
        os.mkdir(model_folder)

datainput_dir= os.path.join(settings.data_dir,settings.augdata_dname)
datanet1_dir = os.path.join(settings.data_dir,settings.net1data_dname)
datanet2_dir = os.path.join(settings.data_dir,settings.net2data_dname)
#endregion initialize directories

if __name__ == "__main__":
        
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
        preprocess_data(settings.data_dir)
        
        create_json_file(datanet1_dir)
        create_json_file(datanet2_dir)
        """
        if not os.path.exists(os.path.join(Data_Folder,"Source_Images", "Training_Images")):
                shutil.copytree(os.path.join(datanet1_dir,"brains"), os.path.join(Data_Folder,"Source_Images", "Training_Images"))
        if not os.path.exists(Annot_Folder): os.mkdir(Annot_Folder)
        if not os.path.exists(Annot_csv): shutil.copy(os.path.join(datainput_dir,"yolo_inputs","data_train.csv"), Annot_csv)
        
        if not os.path.exists(classes_filename): shutil.copy(os.path.join(datainput_dir,"yolo_inputs","data_classes.txt"), os.path.dirname(classes_filename))
        #endregion preprocessing
        
        # Prepare the dataset for YOLO
        multi_df = pd.read_csv(Annot_csv)
        multi_df.drop(columns=multi_df.columns[0], axis=1,inplace=True)
        multi_df.columns = ["image","xmin","ymin","zmin","xmax","ymax","zmax","label"]
        multi_df["image"] = os.path.dirname(__file__) + multi_df["image"].astype(str) 
        multi_df.to_csv(Annot_csv)
        
        labels = multi_df["label"].unique()
        labeldict = dict(zip(labels, range(len(labels))))
        multi_df.drop_duplicates(subset=None, keep="first", inplace=True)
        train_path = Annot_Folder
        
        shutil.copy(os.path.join(datainput_dir,"yolo_inputs","data_train.csv"), os.path.join(keras_path,"model_data","data_train.txt"))
        
        #convert_annot_csv_to_yolo(multi_df, labeldict, path=train_path, target_name=YOLO_filename )
        
         