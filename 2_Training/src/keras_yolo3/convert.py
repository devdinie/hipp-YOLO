#! /usr/bin/env python
"""
Reads Darknet config and weights and creates Keras model with TF backend.

"""

import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import (Input, ZeroPadding3D, Conv3D, Add, UpSampling3D, MaxPooling3D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from tensorflow.keras import initializers
from tensorflow.python.keras.utils.vis_utils import plot_model as plot

parser = argparse.ArgumentParser(description="Darknet To Keras Converter.")
parser.add_argument("config_path", help="Path to Darknet cfg file.")
parser.add_argument("weights_path", help="Path to Darknet weights file.")
parser.add_argument("output_path", help="Path to output Keras model file.")

parser.add_argument("-p", "--plot_model",
                    help="Plot generated Keras model and save as image.",action="store_true")

parser.add_argument("-w", "--weights_only",
                    help="Save as Keras weights file instead of model file.",
                    action="store_true")


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


# %%
def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    assert config_path.endswith(".cfg"), "{} is not a .cfg file".format(config_path)
    #assert weights_path.endswith(".weights"), "{} is not a .weights file".format(weights_path)
    
    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith(".h5"), "output path {} is not a .h5 file".format(output_path)
    output_root = os.path.splitext(output_path)[0]
    
    #region Load weights and config.
    """
    print("Loading weights.")
    weights_file = open(weights_path, "rb")
    major, minor, revision = np.ndarray(shape=(3,), dtype="int32", buffer=weights_file.read(12))
    
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype="int64", buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype="int32", buffer=weights_file.read(4))
    print("Weights Header: ", major, minor, revision, seen)
    """
    #endregion Load weights.
    
    print("Parsing Darknet config.")
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    
    print("Creating Keras model.")
    input_layer = Input(shape=(None, None, None, 1))
    prev_layer = input_layer
    all_layers = []
    
    weight_decay = (float(cfg_parser["net_0"]["decay"])
                    if "net_0" in cfg_parser.sections() else 5e-4)
       
    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print("Parsing section {}".format(section))
        
        if section.startswith("convolutional"):
        #reigon convolutional
            filters = int(cfg_parser[section]["filters"])
            size = int(cfg_parser[section]["size"])
            stride = int(cfg_parser[section]["stride"])
            pad = int(cfg_parser[section]["pad"])
            activation = cfg_parser[section]["activation"]
            batch_normalize = "batch_normalize" in cfg_parser[section]
            
            padding = "same" if pad == 1 and stride == 1 else "valid"
            prev_layer_shape = K.int_shape(prev_layer)

            #region Handle activation.
            act_fn = None
            if activation == "leaky":
                pass  # Add advanced activation later.
            elif activation != "linear":
                raise ValueError( "Unknown activation function `{}` in section {}".format(activation, section))
            #endregion Handle activation.
            
            if stride > 1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = ZeroPadding3D(((1, 0),(1, 0),(1, 0)))(prev_layer)
            
            #region Create Conv3D layer    
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
            #endregion Create Conv3D layer       
        
        #endreigon convolutional
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
            
        elif section.startswith("maxpool"):
            
            size = int(cfg_parser[section]["size"])
            stride = int(cfg_parser[section]["stride"])
            
            all_layers.append(MaxPooling3D(pool_size=(size, size, size), 
                                           strides=(stride, stride,stride), 
                                           padding="same")(prev_layer))
            prev_layer = all_layers[-1]
            
        elif section.startswith("shortcut"):
            
            index = int(cfg_parser[section]["from"])
            activation = cfg_parser[section]["activation"]
            assert activation == "linear", "Only linear activation supported."
            
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]
            
        elif section.startswith("upsample"):
            
            stride = int(cfg_parser[section]["stride"])
            assert stride == 2, "Only stride=2 supported."
            
            all_layers.append(UpSampling3D(stride)(prev_layer))
            prev_layer = all_layers[-1]
            
        elif section.startswith("yolo"):
           
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]
            
        elif section.startswith("net"): pass
        
        else: raise ValueError("Unsupported section header type: {}".format(section))
    
    #region Create and save model.
    if len(out_index) == 0:
        out_index.append(len(all_layers) - 1)
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    print(model.summary())
    
    if args.weights_only:
        model.save_weights("{}".format(output_path))
        print("Saved Keras weights to {}".format(output_path))
    else:
        model.save("{}".format(output_path))
        print("Saved Keras model to {}".format(output_path))

    #if args.plot_model:
    plot(model, to_file="{}.png".format(output_root), show_shapes=True)
    print("Saved model plot to {}.png".format(output_root))
    #endregion Create and save model.
    
if __name__ == "__main__":
    _main(parser.parse_args())
