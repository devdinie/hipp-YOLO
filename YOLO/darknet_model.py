
import os
from functools import wraps

from utils import compose
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization    
from keras.layers import Conv3D, ZeroPadding3D, Add


@wraps(Conv3D)
def DarknetConv3D(*args, **kwargs):
        """Wrapper to set Darknet parameters for Convolution2D."""
        darknet_conv_kwargs = {"kernel_regularizer": l2(5e-4)}
        darknet_conv_kwargs["padding"] = ("valid" if kwargs.get("strides") == (2, 2, 2) else "same")
        darknet_conv_kwargs.update(kwargs)
        
        return Conv3D(*args, **darknet_conv_kwargs)


def DarknetConv3D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    
    no_bias_kwargs = {"use_bias": False}
    no_bias_kwargs.update(kwargs)
    
    return compose(DarknetConv3D(*args, **no_bias_kwargs), 
                   BatchNormalization(), LeakyReLU(alpha=0.1))


def resblock_body(x, no_filters, no_blocks):
        
        # Darknet uses left and top padding instead of 'same' mode
        x = ZeroPadding3D(((1,0), (1, 0), (1, 0)))(x)
        x = DarknetConv3D_BN_Leaky(no_filters, (3, 3, 3), strides=(2, 2, 2))(x)
        
        for i in range(no_blocks):
                y = compose(DarknetConv3D_BN_Leaky(no_filters // 2, (1, 1, 1)),
                            DarknetConv3D_BN_Leaky(no_filters, (3, 3, 3)))(x)
        x = Add()([x, y])
        return x


def make_last_layers(x, no_filters, out_filters):
        
        x = compose(DarknetConv3D_BN_Leaky(no_filters  , (1, 1, 1)),
                    DarknetConv3D_BN_Leaky(no_filters*2, (3, 3, 3)),
                    DarknetConv3D_BN_Leaky(no_filters  , (1, 1, 1)),
                    DarknetConv3D_BN_Leaky(no_filters*2, (3, 3, 3)),
                    DarknetConv3D_BN_Leaky(no_filters  , (1, 1, 1)))(x)
        
        y = compose(DarknetConv3D_BN_Leaky(no_filters*2, (3, 3, 3)),
                    DarknetConv3D(out_filters, (1, 1, 1)))(x)
        
        return x, y

def darknet_body(x): 
        x = DarknetConv3D_BN_Leaky(32, (3, 3, 3))(x)
        x = resblock_body(x, 64, 1)
        x = resblock_body(x, 128, 2)
        x = resblock_body(x, 256, 8)
        x = resblock_body(x, 512, 8)
        x = resblock_body(x, 1024, 4)
        
        return x