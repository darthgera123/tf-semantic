from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, Concatenate, BatchNormalization, Activation
from tensorflow.keras.applications import ResNet50
from deeplabv3plus import ASPP
import tensorflow as tf
from tensorflow import keras

def MScaleV3Plus(input_shape, num_classes, weights='imagenet',backbone='resnet50'):
    base_model = ResNet50(include_top=False,weights=weights,input_shape=input_shape)
    image_features = base_model.get_layer('conv4_block6_out').output
    s2_features = base_model.get_layer('conv2_block1_1_bn').output
    aspp = ASPP(image_features)
    # concatenate onto image features output of other features
    # before concat [bs,256,144,192] after concat [bs,1280,144,192]
    conv_aspp = Conv2D(filters=256, kernel_size=1,use_bias=False)(aspp)
    conv_s2 = Conv2D(filters=48, kernel_size=1, use_bias=False)(s2_features)
    conv_aspp = UpSampling2D(size=(4,4))(conv_aspp)
    cat_s4 = Concatenate(axis=-1)([conv_s2, conv_aspp])
    cat_s4_attn = Concatenate(axis=-1)([conv_s2, conv_aspp])
    #prediction head
    final = Conv2D(filters=256,kernel_size=3,use_bias=False)(cat_s4)
    final = BatchNormalization()(final)
    final = ReLU()(final)
    final = Conv2D(filters=256, kernel_size=3, use_bias=False)(final)
    final = BatchNormalization()(final)
    final = ReLU()(final)
    final = Conv2D(filters=num_classes, kernel_size=1, use_bias=False)(final)
    # ##
    scale_attn = Conv2D(filters=256, kernel_size=3, use_bias=False)(cat_s4_attn)
    scale_attn = BatchNormalization()(scale_attn)
    scale_attn = ReLU()(scale_attn)
    scale_attn = Conv2D(filters=2, kernel_size=1, use_bias=False)(scale_attn)
    scale_attn = Activation('sigmoid')(scale_attn)

    # what to do with scale_attn?

    model = Model(inputs=base_model.input, outputs=final, name="mscalev3plus")
    return model

if __name__ == '__main__':
    model = MScaleV3Plus((32,32,3),1000)
    print(model.summary())
#     res = ResNet50()
#     layer0 = res.layers[0]
#     # res.layers[0] = tf.keras.Sequential()
#     layers = [l.name for l in res.layers]
#     # for layer in tuple(res.layers):
#     #     layer_type = type(layer).__name__
#     #     if hasattr(layer, 'activation') and layer.activation.__name__ == 'relu':
#     #         # Set activation to linear, add PReLU
#     #     #     prelu_name = layer.name + "_prelu"
#     #     #     prelu = PReLU(shared_axes=(1, 2), name=prelu_name) \ 
#     #     #         if layer_type == "Conv2D" else PReLU(name=prelu_name)
#     #     #     layer.activation = linear_activation
#     #     #     new_model.add(layer)
#     #     #     new_model.add(prelu)
#     #     # else:
#     #     #     new_model.add(layer)    
#     #         print(layer.name)
#     layer1 = tf.keras.Sequential()
#     layer2 = tf.keras.Sequential()
#     layer3 = tf.keras.Sequential()
#     layer4 = tf.keras.Sequential()
#     layer5 = tf.keras.Sequential()
#     for layer in tuple(res.layers):
#         if 'conv1_conv' in layer.name:
#             layer1.add(layer)
#         if 'conv1_bn' in layer.name:
#             layer1.add(layer)
#         if 'conv1_relu' in layer.name:
#             layer1.add(layer)
#         if 'poo1_pool' in layer.name:
#             layer1.add(layer)
#         if 'conv2' in layer.name:
#             layer2.add(layer)
#         if 'conv3' in layer.name:
#             layer3.add(layer)
#         if 'conv4' in layer.name:
#             layer4.add(layer)
#         if 'conv5' in layer.name:
#             layer5.add(layer)
#     # print([l.stride for l in layer2.layers])
#     # print(layer2.layers)
#     for layer in layer2.layers:
#         if hasattr(layer,'strides') and '_conv':
#             print(layer.kernel_size,layer.strides,layer.name)
    