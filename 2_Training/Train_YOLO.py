"""
MODIFIED FROM keras-yolo3 PACKAGE, https://github.com/qqwweee/keras-yolo3
Retrain the YOLO model for your own dataset.
"""

import os
import sys
import argparse
import numpy as np

def get_parent_dir(n=1):
    """ 
    # Returns n-th parent dicrectory of current working directory 
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n): current_path = os.path.dirname(current_path)
    return current_path

#region Initialize paths and directories

keras_path  = os.path.join("/data/chamal/projects/swapna/YOLO_22v2/hipp-YOLO/2_Training/src/keras_yolo3")

src_path = os.path.join(get_parent_dir(0), "src")
sys.path.append(src_path)

utils_path = os.path.join(get_parent_dir(1), "Utils")
sys.path.append(utils_path)

Data_Folder = os.path.join(get_parent_dir(1), "Data")
Image_Folder = os.path.join(Data_Folder, "Source_Images", "Training_Images")
Annot_Folder  = os.path.join(Image_Folder, "annotations-export.csv")
YOLO_filename = os.path.join(Annot_Folder, "data_train.txt")

Model_Folder = os.path.join(Data_Folder, "Model_Weights")
YOLO_classname = os.path.join(Model_Folder, "data_classes.txt")

log_dir = Model_Folder
anchors_path = os.path.join(keras_path, "model_data", "yolo_anchors.txt")
weights_path = os.path.join(keras_path, "yolo.h5")

from Train_Utils  import get_classes
from calc_anchors import calc_anchors 
#endregion Initialize paths and directories

if __name__ == "__main__":
        
        #region argsparser
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        
        """
        Command line options
        """
        parser.add_argument( "--annotation_file",type=str, default=YOLO_filename,
                            help="Path to annotation file for Yolo. Default: " + YOLO_filename)
        
        parser.add_argument("--classes_file", type=str, default=YOLO_classname,
                            help="Path to YOLO classnames. Default is " + YOLO_classname)
        
        parser.add_argument("--log_dir", type=str, default=log_dir,
        help="Folder to save training logs and trained weights to. Default is "+ log_dir)
        
        parser.add_argument( "--anchors_path", type=str, default=anchors_path, 
                            help="Path to YOLO anchors. Default is " + anchors_path )
        
        parser.add_argument("--weights_path", type=str, default=weights_path,
        help="Path to pre-trained YOLO weights. Default is " + weights_path)
        
        parser.add_argument("--val_split", type=float, default=0.1,
        help="Percentage of training set to be used for validation. Default is 10%.")
        
        parser.add_argument("--is_tiny", default=False, action="store_true",
        help="Use the tiny Yolo version for better performance and less accuracy. Default is False.")
        
        parser.add_argument("--random_seed", type=float, default=None,
        help="Random seed value to make script deterministic. Default is 'None', i.e. non-deterministic.")
        
        parser.add_argument("--epochs", type=float,  default=51,
                            help="Number of epochs for training last layers and number of epochs for fine-tuning layers. Default is 51.")
        
        FLAGS = parser.parse_args()
        #endregion argsparser
        
        np.random.seed(FLAGS.random_seed)
        log_dir = FLAGS.log_dir
        
        class_names = get_classes(FLAGS.classes_file) 
        num_classes = len(class_names)
        
        calc_anchors(None)
        #anchors = get_anchors(FLAGS.anchors_path) 
        #weights_path = FLAGS.weights_path
        
        