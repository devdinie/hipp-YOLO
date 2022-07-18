import os
import json
import time

import settings
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
#import tensorflow_datasets as tfds
from sklearn.utils import shuffle

from scipy import ndimage
from skimage import exposure
from argparser import args
from augment import resample_img


#region get files list from json file
def get_fileslist(datainput_dir):
        
        json_fname = os.path.join(datainput_dir,"dataset_dict.json")  
        try:
                if not os.path.exists(json_fname):
                        raise IOError()         
                else:
                        with open(json_fname, "r") as json_file: experiment_data = json.load(json_file) 
        except IOError as e:
                print("File {} doesn't exist in {}.".format(json_fname, datainput_dir))
        
        fnames = {}
        if not (args.mode == "test"): 
                no_files = experiment_data["numTraining"]
                for idx in range(no_files):
                        fnames[idx] = [os.path.join(experiment_data["training"][idx]["image"]),
                                          os.path.join(experiment_data["training"][idx]["label"])]
		
        else: 
                no_files = experiment_data["numTesting"]
                for idx in range(no_files):
                        if settings.labels_available:
                                fnames[idx] = [os.path.join(experiment_data["testing"][idx]["image"]),
                                               os.path.join(experiment_data["testing"][idx]["label"])]
                        else: fnames[idx] = [os.path.join(experiment_data["testing"][idx]["image"])]
        return fnames
#endregion get files list from json file

#region functions to process images

#region normalize and adjust input images
def normalize_images(img_fname, msk_fname):
        
        img_file = sitk.ReadImage(img_fname, imageIO=settings.imgio_type)
        img_arr = sitk.GetArrayFromImage(img_file)
        
        imgnorm_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))
        imgnorm_arr = exposure.equalize_adapthist(imgnorm_arr)

        imgnorm_file = sitk.GetImageFromArray(imgnorm_arr)
        imgnorm_file.CopyInformation(img_file)
        
        if (not msk_fname==None) and settings.labels_available:
                msk_file = sitk.ReadImage(msk_fname, imageIO=settings.imgio_type)
        else: msk_file = None
        
        return imgnorm_file, msk_file
#endregion normalize and adjust input images

#region get bounding box for region of interest
def get_roi(msk_file, reserve=2):
        rsv = reserve
        
        msk_arr = sitk.GetArrayFromImage(msk_file)
        msk_arr[msk_arr <= 0] =0 ; msk_arr[msk_arr >0] =1
        
        Z, Y, X = msk_arr.shape[:3]
        idx = np.nonzero(msk_arr > 0)
        
        lower_idx = np.array([max(idx[2].min()-rsv, 0), max(idx[1].min()-rsv, 0), max(idx[0].min()-rsv, 0)],dtype="int")
        upper_idx = np.array([min(idx[2].max()+rsv, X), min(idx[1].max()+rsv, Y), min(idx[0].max()+rsv, Z)],dtype="int")
        
        """
        center_idx = np.array([(upper_idx[0] - lower_idx[0])/2,(upper_idx[1] - lower_idx[1])/2,(upper_idx[2] - lower_idx[2])/2])
        center_idx = np.round(np.divide(center_idx,settings.img_size),4)
        
        normalized_whd = np.array([(upper_idx[0] - lower_idx[0]),(upper_idx[1] - lower_idx[1]),(upper_idx[2] - lower_idx[2])])
        normalized_whd = np.round(np.divide(normalized_whd, settings.img_size),4)
        """
        return [lower_idx, upper_idx]
        
#endregion get bounding box for region of interest

#endregion functions to process images

#region generating train/test datasets 
def create_yolo_inputs(net, annot_list):
        
        yolo_inputdir = os.path.join(settings.data_dir,settings.augdata_dname,"yolo_inputs")
        if not os.path.exists(yolo_inputdir):
                os.mkdir(yolo_inputdir)
        
        starttime_train = time.time()
        
        #region create data classes text files
        try:
            with open(os.path.join(settings.data_dir,settings.augdata_dname, "dataset_dict.json"), "r") as fp:
                    json_dict = json.load(fp)
                    yolo_objnames = open(os.path.join(yolo_inputdir,"data_classes.txt"), 'a')
                    yolo_objnames.write(json_dict['labels']['1'])
        except IOError as e:
            print("Could not read class name(s). File {} doesn't exist in {} ".format("dataset_dict.json", os.path.join(settings.data_dir,settings.augdata_dname)))
        #endregion create data classes text files
        
        no_files    = len(annot_list)
        no_trainfiles= int(no_files*settings.train_test_split)
        no_valfiles  = int((no_files - no_trainfiles)*settings.validate_test_split)
        
        #print(no_files)
        #print(no_trainfiles, no_valfiles)
        
        file_idxs = shuffle(np.arange(0,no_files))
        train_idxs = file_idxs[:no_trainfiles]
        
        val_test_idxs = file_idxs[no_trainfiles:]
        val_idxs = val_test_idxs[:no_valfiles]
        test_idxs= val_test_idxs[no_valfiles:]
        
        train_idxs = np.concatenate((train_idxs, val_idxs), axis=None)
        
        train_list= np.take(annot_list, train_idxs)
        test_list = np.take(annot_list, test_idxs)
        

        with open(os.path.join(yolo_inputdir,'annotations.csv'), 'a') as file:
                for line in train_list: file.write(line+'\n')
                
        with open(os.path.join(yolo_inputdir,'data_train.txt'), 'a') as file:
                for line in train_list: file.write(line.split(",")[0] +'\n')
                
        with open(os.path.join(yolo_inputdir,'data_test.txt'), 'a') as file:
                for line in test_list: file.write(line.split(",")[0] +'\n')
        
        #region create yolo input lists
        
        endtime_train = time.time()
        print("- Network {}: creating input lists for yolo complete.".format(net, endtime_train-starttime_train))
        #endregion create yolo input lists
           
def preprocess_data(net, data_dir):
        
        #region initialize directories and paths
        datanet1_dir = os.path.join(data_dir,"data_net1_loc")
        datanet2_dir = os.path.join(data_dir,"data_net2_seg")
        
        if not os.path.exists(datanet1_dir):
                os.mkdir(datanet1_dir)
                os.mkdir(os.path.join(datanet1_dir,"brains"))
                if settings.labels_available:
                        os.mkdir(os.path.join(datanet1_dir,"target_labels"))
                
        if not os.path.exists(datanet2_dir):
                os.mkdir(datanet2_dir)
                os.mkdir(os.path.join(datanet2_dir,"brains"))
                if settings.labels_available:
                        os.mkdir(os.path.join(datanet2_dir,"target_labels"))
                        
        if settings.yolo_localize and not os.path.exists(os.path.join(settings.data_dir,settings.augdata_dname,"yolo_inputs")):
                os.mkdir(os.path.join(settings.data_dir,settings.augdata_dname,"yolo_inputs"))
        #endregion initialize directories and paths
        
        files_list = get_fileslist(os.path.join(data_dir,settings.augdata_dname))
        annot_list = list()
        #region prepare inputs for net 1 -localize and net 2 - segment
        for idx in range(0,len(files_list)):
                
                img_fname = files_list[idx][0]
                
                if not(args.mode == "test") or settings.labels_available:
                        msk_fname = files_list[idx][1]
                        if not (os.path.exists(img_fname) or os.path.exists(msk_fname)): continue
                else:
                        msk_fname == None
        
                img_file, msk_file = normalize_images(img_fname, msk_fname)
                
                #region split and flip images only
                mid_x = int(img_file.GetSize()[0]/2)
                
                img_file_L = img_file[0:mid_x,:,:]
                img_file_Rpf = img_file[mid_x:(mid_x*2),:,:]
                
                img_arr_R = sitk.GetArrayFromImage(img_file_Rpf)
                img_arr_R = img_arr_R[:,:,::-1]
                img_file_R = sitk.GetImageFromArray(img_arr_R)
                img_file_R.CopyInformation(img_file_Rpf)
                #endregion split and flip images only
                
                if not(args.mode == "test") or settings.labels_available:
                        #region split and flip labels and get roi
                        msk_file_L = msk_file[0:mid_x,:,:]
                        msk_file_Rpf = msk_file[mid_x:(mid_x*2),:,:]
                
                        msk_arr_R = sitk.GetArrayFromImage(msk_file_Rpf)
                        msk_arr_R = msk_arr_R[:,:,::-1]
                        msk_file_R = sitk.GetImageFromArray(msk_arr_R)
                        msk_file_R.CopyInformation(msk_file_Rpf)
                        
                        [bb_Lbot, bb_Lup] = get_roi(msk_file_L, reserve=4)
                        [bb_Rbot, bb_Rup] = get_roi(msk_file_R, reserve=4)
                        #endregion split and flip labels only and get roi
                        
                        #region create annotations for yolo 
                        if settings.yolo_localize:
                                annotL = np.array([os.path.basename(img_fname).replace("_t1_","_labels_L_"),bb_Lbot[0],bb_Lbot[1],bb_Lbot[2],bb_Lup[0],bb_Lup[1],bb_Lup[2],int(1)])
                                annotR = np.array([os.path.basename(img_fname).replace("_t1_","_labels_R_"),bb_Rbot[0],bb_Rbot[1],bb_Rbot[2],bb_Rup[0],bb_Rup[1],bb_Rup[2],int(1)])
                                
                                annot_list.append(np.array2string(annotL, separator=',',max_line_width=10000).strip("[]").replace("'",""))
                                annot_list.append(np.array2string(annotR, separator=',',max_line_width=10000).strip("[]").replace("'","")) 
                        #endregion create annotations for yolo
                
                        #region generate and save net2 labels
                        img_file_Lcrp = img_file_L[bb_Lbot[0]:bb_Lup[0], bb_Lbot[1]: bb_Lup[1], bb_Lbot[2]: bb_Lup[2]]
                        msk_file_Lcrp = msk_file_L[bb_Lbot[0]:bb_Lup[0], bb_Lbot[1]: bb_Lup[1], bb_Lbot[2]: bb_Lup[2]]
                
                        img_file_Rcrp = img_file_R[bb_Rbot[0]:bb_Rup[0], bb_Rbot[1]: bb_Rup[1], bb_Rbot[2]: bb_Rup[2]]
                        msk_file_Rcrp = msk_file_R[bb_Rbot[0]:bb_Rup[0], bb_Rbot[1]: bb_Rup[1], bb_Rbot[2]: bb_Rup[2]]
                        
                        img_size_crp = settings.img_size #tuple(int(val/6) for val in settings.img_size)
                        
                        img_file_Lcrp, msk_file_Lcrp = resample_img(img_size_crp, img_file_Lcrp, msk_file_Lcrp)
                        img_file_Rcrp, msk_file_Rcrp = resample_img(img_size_crp, img_file_Rcrp, msk_file_Rcrp)
                        
                        #region threshold cropped labels and close holes
                        msk_arr_Lcrp = sitk.GetArrayFromImage(msk_file_Lcrp)
                        msk_arr_Rcrp = sitk.GetArrayFromImage(msk_file_Rcrp)
                        
                        msk_arr_Lcrp[msk_arr_Lcrp>0] =1
                        msk_arr_Rcrp[msk_arr_Rcrp>0] =1
                        
                        msk_arr_Lcrp = ndimage.binary_closing(msk_arr_Lcrp).astype(int)
                        msk_arr_Rcrp = ndimage.binary_closing(msk_arr_Rcrp).astype(int)
                        
                        msk_file_Lcrpt = sitk.GetImageFromArray(msk_arr_Lcrp)
                        msk_file_Rcrpt = sitk.GetImageFromArray(msk_arr_Rcrp)
                        
                        msk_file_Lcrpt.CopyInformation(msk_file_Lcrp)
                        msk_file_Rcrpt.CopyInformation(msk_file_Rcrp)
                        #endregion threshold cropped labels and close holes
                        
                        sitk.WriteImage(img_file_Lcrp ,os.path.join(datanet2_dir,"brains",os.path.basename(img_fname).replace("_t1_","_t1_L_")))
                        sitk.WriteImage(msk_file_Lcrpt,os.path.join(datanet2_dir,"target_labels",os.path.basename(img_fname).replace("_t1_","_labels_L_")))
                
                        sitk.WriteImage(img_file_Rcrp ,os.path.join(datanet2_dir,"brains",os.path.basename(img_fname).replace("_t1_","_t1_R_")))
                        sitk.WriteImage(msk_file_Rcrpt,os.path.join(datanet2_dir,"target_labels",os.path.basename(img_fname).replace("_t1_","_labels_R_")))
                        #endregion generate and save net2 labels
                
                        #region generate and save net1 labels
                        msk_file_L[bb_Lbot[0]:bb_Lup[0], bb_Lbot[1]: bb_Lup[1], bb_Lbot[2]: bb_Lup[2]] =1
                        msk_file_R[bb_Rbot[0]:bb_Rup[0], bb_Rbot[1]: bb_Rup[1], bb_Rbot[2]: bb_Rup[2]] =1
                        
                        img_file_L, msk_file_L = resample_img(settings.img_size,img_file_L,msk_file_L)
                        img_file_R, msk_file_R = resample_img(settings.img_size,img_file_R,msk_file_R )
                        
                        sitk.WriteImage(img_file_L,os.path.join(datanet1_dir,"brains",os.path.basename(img_fname).replace("_t1_","_t1_L_")))
                        sitk.WriteImage(msk_file_L,os.path.join(datanet1_dir,"target_labels",os.path.basename(img_fname).replace("_t1_","_labels_L_")))
                
                        sitk.WriteImage(img_file_R,os.path.join(datanet1_dir,"brains",os.path.basename(img_fname).replace("_t1_","_t1_R_")))
                        sitk.WriteImage(msk_file_R,os.path.join(datanet1_dir,"target_labels",os.path.basename(img_fname).replace("_t1_","_labels_R_")))  
                        #endregion generate and save net1 labels
                     
        #endregion prepare inputs for net 1 -localize and net 2 - segment
        
        #region prepare yolo network inputs
        #print(annot_list)
        create_yolo_inputs(net,annot_list)
                                
        #endregion prepare yolo network inputs
