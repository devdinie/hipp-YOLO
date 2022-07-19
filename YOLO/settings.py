
import os

#region define mode and main directory paths
"""
# Define directory paths with respect to current working directory
# Setting if data augmentation is required (in train mode only)
"""

mode = "train"
#mode = "test"
augment = False
is_overwrite = True
yolo_localize= True
visualize_training = True

root_dir = os.path.join(os.path.dirname(__file__),"..")
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/")
testdata_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)),"testdata/")
visualizations_dir =  os.path.join(os.path.dirname(os.path.dirname(__file__)),"visualizations")

labels_available = os.path.exists(os.path.join(data_dir,"target_labels"))
augdata_dname = "data_input" 
net1data_dname= "data_net1_loc"
net2data_dname= "data_net2_seg"
#endregion define mode and main directory paths

#region image related settings
img_size = (160,160,160)
imgio_type = "NiftiImageIO"
#endregion image related settings

#region data and model related settings
"""
# Names used to save the models
# Ratio to split dataset for training and testing (train_test_split)
# From the remaining test dataset, the ratio to split between
# validation and testing data (test dataset = no. files - train data)
# test dataset = {validation data, test data}
"""
random_seed= 816

batch_size = 2
train_test_split = 0.6
validate_test_split = 0.5

net1_loc_modelname = "net1_model_localize"
net2_seg_modelname = "net2_model_segment"

epochs = 40
filters = 16
use_upsampling = True #Use upsampling instead of transposed convolution

no_input_classes = 1
no_output_classes= 1

#endregion data and model related settings
