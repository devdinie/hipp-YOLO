from fileinput import filename
from itertools import count
import os
import io
import json
import time

import argparse


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
from keras.layers import Input, ZeroPadding3D, Conv3D, Concatenate, MaxPooling3D, Add, UpSampling3D, Lambda  

from utils import compose
from tensorflow.python.keras.utils.vis_utils import plot_model as plot
from calculate_anchors import get_anchors
from darknet_model import darknet_body, make_last_layers, DarknetConv3D_BN_Leaky

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

        c=0 ; out_index = []
        for section in cfg_parser.sections():
                print("Parsing section {}...".format(section))
                c +=1 # Quick check 
                if c>=5: break
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
                                
                        #print("Parsing section {} complete".format(section))
                        #endregion create convolutional layer
                #endregion convolutional
              
                #region route
                elif section.startswith("route"):
                        
                        ids = [int(i) for i in cfg_parser[section]["layers"].split(",")]
                        layers = [all_layers[i] for i in ids]
                        #[print(all_layers[i]) for i in ids]
                
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
        if len(out_index) == 0: out_index.append(len(all_layers) - 1)
        
        model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
        #print(model.summary())
        
        if save_weights:
                model.save_weights("{}".format(outmodel_fname.replace(".h5",".weights")))
                print("Saved Keras weights to {}".format(outmodel_fname.replace(".h5",".weights"))) 
        else:
                model.save("{}".format(outmodel_fname))
                print("Saved Keras model to {}".format(outmodel_fname))
        
        """        
        plot(model, to_file="{}".format(outmodel_fname.replace(".h5",".png")), show_shapes=True)
        print("Saved model plot to {}".format(outmodel_fname.replace(".h5",".png")))
        #endregion create and save model
        """
        #endregion create and save model
        
        #region generate anchors 
        if not os.path.exists(os.path.join("model_files", "anchors.txt")):
                print("Anchors file 'anchors.txt' not found. Generating custom anchors...")
                get_anchors(os.path.join(settings.data_dir,settings.augdata_dname,"yolo_inputs", "annotations.csv"))
        #endregion generate anchors
       
def yolo_body(inputs, no_anchors, no_classes):
        """Create YOLO_V3 model CNN body in Keras."""
        
        darknet = Model(inputs, darknet_body(inputs))
        x, y1 = make_last_layers(darknet.output, 512, no_anchors * (no_classes + 7))
        x = compose(DarknetConv3D_BN_Leaky(256, (1, 1, 1)), UpSampling3D(2))(x)
        x = Concatenate()([x, darknet.layers[51].output])
        
        x, y2 = make_last_layers(x, 256, no_anchors * (no_classes + 7))
        
        x = compose(DarknetConv3D_BN_Leaky(128, (1, 1, 1)), UpSampling3D(2))(x)
        x = Concatenate()([x, darknet.layers[29].output])
        x, y3 = make_last_layers(x, 128, no_anchors * (no_classes + 7))

        return Model(inputs, [y1, y2, y3])

def yolo_head(feats, anchors, no_classes, input_shape, calc_loss=False):
        
        no_anchors = len(anchors)
        
        # Reshape to batch, height, width, depth, no_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, no_anchors, 3])
        
        grid_shape = K.shape(feats)[1:4]  # height, width, depth
        
        grid_xy = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1,-1, 1, 1]), [grid_shape[0], 1, 1, 1])
        grid_xz = K.tile(K.reshape(K.arange(0, stop=grid_shape[2]), [1, 1,-1, 1]), [grid_shape[0], 1, 1, 1])
        grid_yz = K.tile(K.reshape(K.arange(0, stop=grid_shape[2]), [1, 1,-1, 1]), [1, grid_shape[1], 1, 1])
        grid = K.concatenate([grid_xy, grid_xz, grid_yz])
        
        grid = K.cast(grid, K.dtype(feats))
        
        feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], grid_shape[2], no_anchors, no_classes + 7])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xyz = (K.sigmoid(feats[..., :3]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        
        box_whd = (K.exp(feats[..., 3:6])* anchors_tensor/ K.cast(input_shape[::-1], K.dtype(feats)))
       
        box_confidence  = K.sigmoid(feats[..., 5:6])
        box_class_probs = K.sigmoid(feats[...,7:])
        
        if calc_loss == True:
                return grid, feats, box_xyz, box_whd
        return box_xyz, box_whd, box_confidence, box_class_probs

def box_iou(b1, b2):
        # Expand dim to apply broadcasting.
        b1 = K.expand_dims(b1, -2)
        b1_xyz = b1[..., :3]
        b1_whd = b1[..., 3:6]
        b1_whd_half = b1_whd / 2.0
        b1_mins = b1_xyz - b1_whd_half
        b1_maxes = b1_xyz + b1_whd_half
        
        # Expand dim to apply broadcasting.
        b2 = K.expand_dims(b2, 0)
        b2_xyz = b2[..., :3]
        b2_whd = b2[..., 3:6]
        b2_whd_half = b2_whd / 2.0
        b2_mins = b2_xyz - b2_whd_half
        b2_maxes= b2_xyz + b2_whd_half

        intersect_mins  = K.maximum(b1_mins, b2_mins)
        intersect_maxes = K.minimum(b1_maxes, b2_maxes)
        intersect_whd = K.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_whd[..., 0] * intersect_whd[..., 1] * intersect_whd[..., 2]
        b1_area = b1_whd[..., 0] * b1_whd[..., 1] * b1_whd[..., 2]
        b2_area = b2_whd[..., 0] * b2_whd[..., 1] * b2_whd[..., 2]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

        return iou
                

def yolo_loss(args, anchors, no_classes, ignore_thresh=0.5, print_loss=False):
        
        no_layers = len(anchors) // 3  # default setting
        yolo_outputs = args[:no_layers]
        y_true = args[no_layers:]
        
        anchors = np.array(anchors)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if no_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        input_shape = K.cast(K.shape(yolo_outputs[0])[1:4] * 16, K.dtype(y_true[0]))
    
        grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:4], K.dtype(y_true[0])) for l in range(no_layers)]
        loss = 0
        m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
        mf = K.cast(m, K.dtype(yolo_outputs[0]))
        
        for l in range(no_layers):
                object_mask = y_true[l][..., 6:7]
                true_class_probs = y_true[l][..., 7:]

                grid, raw_pred, pred_xyz, pred_whd = yolo_head(yolo_outputs[l], anchors[anchor_mask[l]],
                                                               no_classes, input_shape, calc_loss=True)
                
                pred_box = K.concatenate([pred_xyz, pred_whd])
                raw_true_xyz = y_true[l][..., :3] * grid_shapes[l][::-1] - grid
                raw_true_whd = K.log(y_true[l][..., 3:6] / anchors[anchor_mask[l]] * input_shape[::-1])
                
                raw_true_whd = K.switch(object_mask, raw_true_whd, K.zeros_like(raw_true_whd))  # avoid log(0)=-inf
                box_loss_scale = 2 - y_true[l][..., 3:4] * y_true[l][..., 4:5] * y_true[l][..., 5:6]
                
                # Find ignore mask, iterate over each of batch.
                ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
                object_mask_bool = K.cast(object_mask, "bool")
                
                def loop_body(b, ignore_mask):
                        true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                        iou = box_iou(pred_box[b], true_box)
                        best_iou = K.max(iou, axis=-1)
                        ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
                        return b + 1, ignore_mask
        
                _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
                ignore_mask = ignore_mask.stack()
                ignore_mask = K.expand_dims(ignore_mask, -1)
                
                xyz_loss = (object_mask * box_loss_scale
                           * K.binary_crossentropy(raw_true_xyz, raw_pred[..., 0:3], from_logits=True))
                
                whd_loss = (object_mask * box_loss_scale * 0.5 * K.square(raw_true_whd - raw_pred[..., 3:6]))
                
                confidence_loss = (object_mask
                                   * K.binary_crossentropy(object_mask, raw_pred[..., 6:7], from_logits=True)
                                   + (1 - object_mask)
                                   * K.binary_crossentropy(object_mask, raw_pred[..., 6:7], from_logits=True)
                                   * ignore_mask)
                
                class_loss = object_mask * K.binary_crossentropy(
                        true_class_probs, raw_pred[..., 7:], from_logits=True)
                
                xyz_loss = K.sum(xyz_loss) / mf
                whd_loss = K.sum(whd_loss) / mf
                confidence_loss = K.sum(confidence_loss) / mf
                class_loss = K.sum(class_loss) / mf
                loss += xyz_loss + whd_loss + confidence_loss + class_loss
                
                if print_loss:
                        loss = tf.Print(loss,[loss, xyz_loss, whd_loss, confidence_loss,
                                        class_loss, K.sum(ignore_mask)],message="loss: ")
        return loss


def create_yolo_model(input_shape, anchors, no_classes, load_pretrained=False, 
                      freeze_body=2, weights_path=os.path.join("model_files","yolov3.weights")):
        
        K.clear_session()
        
        image_input = Input(shape=(None, None, None, 1))
        h, w, d = input_shape
        no_anchors = len(anchors)
        
        y_true = [Input(shape=( h // {0: 16, 1: 8, 2: 4}[l], 
                                w // {0: 16, 1: 8, 2: 4}[l],
                                d // {0: 16, 1: 8, 2: 4}[l],
                                no_anchors//3, no_classes+7))
                  for l in range(3)]
        
        model_body = yolo_body(image_input, no_anchors//3, no_classes)
        print("Create YOLOv3 model with {} anchors and {} classes.".format(no_anchors, no_classes))
        
        #TODO: Load pretrained=True not tested
        """
        if load_pretrained: 
                print(model_body.weights)
                model_body.load_weights(weights_path, by_name=True, skip_mismatch=True) 
                #print("Load weights {}.".format(weights_path)) 
               
                if freeze_body in [1, 2]:
                        # Freeze darknet53 body or freeze all but 3 output layers.
                        num = (59, len(model_body.layers) - 3)[freeze_body - 1]
                        print(num)
                        for i in range(num): model_body.layers[i].trainable = False
                        print("Freeze the first {} layers of total {} layers.".format(num, len(model_body.layers)))
        """
        
        #print(model_body.summary(line_length = 150))
        model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss",
                            arguments={"anchors": anchors, "no_classes": no_classes, 
                                       "ignore_thresh": 0.5},)([*model_body.output, *y_true])
        
        model = Model([model_body.input, *y_true], model_loss)
        
        return model