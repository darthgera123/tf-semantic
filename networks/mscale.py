from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, Concatenate, BatchNormalization, Activation
from tensorflow.keras.applications import ResNet50
from deeplabv3plus import ASPP

def MScaleV3Plus(input_shape, num_classes, weights='imagenet',backbone='resnet50', attn_2b=False):
    base_model = ResNet50(include_top=False,weights=weights,input_shape=input_shape)
    image_features = base_model.get_layer('conv4_block6_out').output
    s2_features = None #to check, from another layer of resnet
    aspp = ASPP(image_features)
    #something here, pytorch code:
    # if self.fuse_aspp and \
    #     aspp_lo is not None and aspp_attn is not None:
    #     aspp_attn = scale_as(aspp_attn, aspp)
    #     aspp_lo = scale_as(aspp_lo, aspp)
    #     aspp = aspp_attn * aspp_lo + (1 - aspp_attn) * aspp
    conv_aspp = Conv2D(filters=256, kernel_size=1,use_bias=False)(aspp)
    conv_s2 = Conv2D(filters=48, kernel_size=1, use_bias=False)(s2_features)
    conv_aspp = #need to upsample conv_aspp to to size of s2_features, a nearest neighbor interpolation
    cat_s4 = Concatenate(axis=1)([conv_s2, conv_aspp])
    cat_s4_attn = Concatenate(axis=1)([conv_s2, conv_aspp])
    #prediction head
    final = Conv2D(filters=256,kernel_size=3,use_bias=False)(cat_s4)
    final = BatchNormalization()(final)
    final = ReLU()(final)
    final = Conv2D(filters=256, kernel_size=3, use_bias=False)(final)
    final = BatchNormalization()(final)
    final = ReLU()(final)
    final = (filters=num_classes, kernel_size=1, use_bias=False)
    scale_attn = Conv2D(filters=256, kernel_size=3, use_bias=False)(cat_s4_attn)
    scale_attn = BatchNormalization()(scale_attn)
    scale_attn = ReLU()(scale_attn)
    if attn_2b:
        attn_ch = 2
    else:
        attn_ch = 1
    scale_attn = Conv2D(filters=attn_ch, kernel_size=1, use_bias=False)(scale_attn)
    scale_attn = Activation('sigmoid')(scale_attn)

    #interpolate final to input_shape[2:]
    #interpolate scale_attn to input_shape[2:]

    #pytorch code:
    # if self.attn_2b:
    #     logit_attn = scale_attn[:, 0:1, :, :]
    #     aspp_attn = scale_attn[:, 1:, :, :]
    # else:
    #     logit_attn = scale_attn
    #     aspp_attn = scale_attn

    # return out, logit_attn, aspp_attn, aspp
    return None

