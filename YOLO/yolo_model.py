from fileinput import filename
import os
import io
import json
import time

import argparse

from sklearn import cluster
import settings
import numpy as np
import configparser

import tensorflow as tf
import SimpleITK as sitk
                    
from argparser import args
from augment import resample_img
from sklearn.utils import shuffle
from collections import defaultdict

from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization    
from keras.layers import Input, ZeroPadding3D, Conv3D, Concatenate, MaxPooling3D, Add, UpSampling3D  

from tensorflow.python.keras.utils.vis_utils import plot_model as plot
from calculate_anchors import get_anchors

save_weights = True

def unique_config_sections(config_file):
    """
    Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith("["):
                section = line.strip().strip("[]")
                _section = section + "_" + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    
    return output_stream

        
def init_yolo_model(config_file, weights_file,outmodel_fname):
        
        #region parsing darknet
        print("Parsing Darknet config...")
        unique_config_file = unique_config_sections(config_file)
        cfg_parser = configparser.ConfigParser()
        cfg_parser.read_file(unique_config_file)
        print("Parsing Darknet config complete.")
        #endregion parsing darknet
        
        #region creating keras model
        print("Creating Keras model...")
        input_layer = Input(shape=(None, None, None, 1))
        prev_layer = input_layer
        all_layers = []
        print("Creating Keras model complete.")
        #endregion creating keras model
        
        weight_decay = (float(cfg_parser["net_0"]["decay"]) if "net_0" in cfg_parser.sections() else 5e-4)
        """
        out_index = []
        for section in cfg_parser.sections():
                print("Parsing section {}".format(section))
                
                #region convolutional
                if section.startswith("convolutional"):
                        filters = int(cfg_parser[section]["filters"])
                        size = int(cfg_parser[section]["size"])
                        stride = int(cfg_parser[section]["stride"])
                        pad = int(cfg_parser[section]["pad"])
                        activation = cfg_parser[section]["activation"]
                        batch_normalize = "batch_normalize" in cfg_parser[section]
                        
                        #region convolutional: padding activation, stride
                        padding = "same" if pad == 1 and stride == 1 else "valid"
                        prev_layer_shape = K.int_shape(prev_layer)
                        print(prev_layer.shape)
                        act_fn = None
                        if activation == "leaky": 
                                pass  # Add advanced activation later.
                        elif activation != "linear":
                                raise ValueError( "Unknown activation function `{}` in section {}".format(activation, section))
                        
                        if stride > 1:
                                # Darknet uses left and top padding instead of 'same' mode
                                prev_layer = ZeroPadding3D(((1, 0),(1, 0),(1, 0)))(prev_layer)
                        #endregion convolutional: padding activation, stride
                        
                        #region create convolutional layer
                        conv_layer = (Conv3D(filters, (size, size,size), strides=(stride, stride, stride), 
                                      kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                      kernel_regularizer=l2(weight_decay), use_bias=not batch_normalize, 
                                      activation=act_fn, padding=padding))(prev_layer)
            
                        if batch_normalize:
                                conv_layer = (BatchNormalization(beta_initializer="zeros", gamma_initializer="ones",
                                        moving_mean_initializer="zeros", moving_variance_initializer="ones",))(conv_layer)
                        prev_layer = conv_layer
   
                        if activation == "linear": 
                                all_layers.append(prev_layer)
                        elif activation == "leaky":
                                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                                prev_layer = act_layer
                                all_layers.append(act_layer)
                                
                        
                        print("Parsing section {} complete".format(section))
                        #endregion create convolutional layer
                #endregion convolutional
                
                #region route
                elif section.startswith("route"):
                        
                        ids = [int(i) for i in cfg_parser[section]["layers"].split(",")]
                        layers = [all_layers[i] for i in ids]
                        [print(all_layers[i]) for i in ids]
                
                        if len(layers) > 1:
                                print("Concatenating route layers:", layers)
                                concatenate_layer = Concatenate()(layers)
                                all_layers.append(concatenate_layer)
                                prev_layer = concatenate_layer
                
                        else:
                                skip_layer = layers[0]  # only one layer to route
                                all_layers.append(skip_layer)
                                prev_layer = skip_layer
                #endregion route
                
                #region maxpool
                elif section.startswith("maxpool"):
                        size = int(cfg_parser[section]["size"])
                        stride = int(cfg_parser[section]["stride"])
                        
                        all_layers.append(MaxPooling3D(pool_size=(size, size, size), 
                                        strides=(stride, stride,stride), padding="same")(prev_layer))
                        prev_layer = all_layers[-1]
                #endregion maxpool
                
                #region shortcut
                elif section.startswith("shortcut"):
                        index = int(cfg_parser[section]["from"])
                        activation = cfg_parser[section]["activation"]
                        assert activation == "linear", "Only linear activation supported."
                
                        all_layers.append(Add()([all_layers[index], prev_layer]))
                        prev_layer = all_layers[-1]
                #endregion shortcut
                
                #region upsample
                elif section.startswith("upsample"):
                        stride = int(cfg_parser[section]["stride"])
                        assert stride == 2, "Only stride=2 supported."
                
                        all_layers.append(UpSampling3D(stride)(prev_layer))
                        prev_layer = all_layers[-1]
                #endregion upsample
                
                #region yolo layer
                elif section.startswith("yolo"):
                        out_index.append(len(all_layers) - 1)
                        all_layers.append(None)
                        prev_layer = all_layers[-1]        
                #endregion yolo layer
                
                elif section.startswith("net"): pass
                else: raise ValueError("Unsupported section header type: {}".format(section))
                
        #region create and save model
        if len(out_index) == 0:
                out_index.append(len(all_layers) - 1)
                print("*",out_index)
        model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
        print(model.summary())
        
        if save_weights:
                model.save_weights("{}".format(outmodel_fname.replace(".h5",".weights")))
                print("Saved Keras weights to {}".format(outmodel_fname.replace(".h5",".weights")))
                
        else:
                model.save("{}".format(outmodel_fname))
                print("Saved Keras model to {}".format(outmodel_fname))
                
        plot(model, to_file="{}".format(outmodel_fname.replace(".h5",".png")), show_shapes=True)
        print("Saved model plot to {}".format(outmodel_fname.replace(".h5",".png")))
        #endregion create and save model
        """
        #region get anchors
        get_anchors(os.path.join(settings.data_dir,settings.augdata_dname,"yolo_inputs", "annotations.csv"))
        #endregion get anchors
        
