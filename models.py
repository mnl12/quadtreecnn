import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix


class const_multiply(tf.keras.layers.Layer):
    def __init__(self, const=.3):
        super(const_multiply, self).__init__()
        self.const=tf.Variable([[[[const]]]], shape=(1,1,1,1))
    def call(self, inputs):
        return tf.multiply(self.const, inputs)

class attention_sp(tf.keras.layers.Layer):
    def __init__(self, const=.3):
        super(attention_sp, self).__init__()


    def call(self, B,C, D):
        [b_size,H,W,ch]=B.shape
        B_re=tf.reshape(B, (-1, tf.multiply(H,W), ch))
        C_re = tf.reshape(C, (-1, tf.multiply(H, W), ch))
        D_re = tf.reshape(D, (-1, tf.multiply(H, W), ch))
        prod_mat=tf.matmul(B_re,tf.transpose(C_re, perm=[0, 2, 1]))
        soft_mat=tf.nn.softmax(prod_mat)
        out_re=tf.matmul(soft_mat,D_re)
        out=tf.reshape(out_re, (-1, H,W, ch))
        return out


def attention_layer(Input):
    dim=Input.shape[-1]

    B=tf.keras.layers.Conv2D(dim,1, activation=None)(Input)
    C = tf.keras.layers.Conv2D(dim, 1, activation=None)(Input)
    D = tf.keras.layers.Conv2D(dim, 1, activation=None)(Input)
    out_a=attention_sp()(B,C, D)
    out_a = tf.keras.layers.Conv2D(dim, 1, activation=None)(out_a)
    out=tf.keras.layers.Add()([Input, out_a])
    return out





def Upsample(tensor, size):
    '''bilinear upsampling'''
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = tf.keras.layers.Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y


def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = tf.keras.backend.int_shape(tensor)

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = tf.keras.layers.Activation('relu', name=f'relu_1')(y_pool)

   # y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])
    y_pool = tf.keras.layers.UpSampling2D((dims[1], dims[2]), interpolation='bilinear')(y_pool)

    y_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = tf.keras.layers.BatchNormalization(name=f'bn_2')(y_1)
    y_1 = tf.keras.layers.Activation('relu', name=f'relu_2')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = tf.keras.layers.BatchNormalization(name=f'bn_3')(y_6)
    y_6 = tf.keras.layers.Activation('relu', name=f'relu_3')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = tf.keras.layers.BatchNormalization(name=f'bn_4')(y_12)
    y_12 = tf.keras.layers.Activation('relu', name=f'relu_4')(y_12)

    y_18 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = tf.keras.layers.BatchNormalization(name=f'bn_5')(y_18)
    y_18 = tf.keras.layers.Activation('relu', name=f'relu_5')(y_18)

    y = tf.keras.layers.concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization(name=f'bn_final')(y)
    y = tf.keras.layers.Activation('relu', name=f'relu_final')(y)
    return y



def segmentation_network (base_model_name,decoder_name, n_classes, IMAGE_SIZE):
    OUTPUT_CHANNELS = n_classes
    if base_model_name=='mobilenet':
        base_model = tf.keras.applications.MobileNetV2(input_shape=[*IMAGE_SIZE, 3], include_top=False)
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
    elif base_model_name=='vgg':
        base_model = tf.keras.applications.VGG16(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        layer_names = [
            'block1_pool',  # 64
            'block2_pool',
            'block3_pool',
            'block4_pool',
            'block5_pool'
        ]
    elif base_model_name == 'resnet':
        base_model = tf.keras.applications.ResNet50(input_shape=[*IMAGE_SIZE, 3], include_top=False, weights='imagenet')
        layer_names = ['conv2_block3_out', 'conv4_block6_out']



    layers = [base_model.get_layer(name).output for name in layer_names]
    #for mlayer in base_model.layers:
    #    mlayer.trainable = False

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = True

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]
    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3])
    enc_outs = down_stack(inputs)



    if decoder_name=='class_att_bias':
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3, strides=2, padding='same', activation=None)
        x = enc_outs[-1]
        skips=reversed(enc_outs[:-1])
        z = x
        z = tf.keras.layers.Flatten()(z)
        # z=tf.keras.layers.GlobalAveragePooling2D()(z)
        z = tf.keras.layers.Dense(64, activation='relu')(z)
        z = tf.keras.layers.Dense(1, activation=None)(z)
        xg = tf.keras.layers.Dense(1, activation=None)(z)
        z = tf.keras.layers.Activation('sigmoid', name='class_out')(z)

        xg = tf.keras.backend.expand_dims(xg, axis=1)
        xg = tf.keras.backend.expand_dims(xg, axis=1)
        xg = tf.keras.layers.UpSampling2D(size=IMAGE_SIZE, interpolation='nearest')(xg)

        for up, skip in zip(up_stack, skips):
            x = up(x)
            cancat = tf.keras.layers.Concatenate()
            x = cancat([x, skip])

        x = last(x)
        xg=tf.keras.layers.Multiply()([x,xg])
        xg=tf.keras.layers.Conv2D(1,1,use_bias=False, kernel_initializer='zeros')(xg)
        x = tf.keras.layers.Add()([x, xg])
        x = tf.keras.layers.Activation('sigmoid', name='seg_out')(x)

    elif decoder_name=='class_att_sigmoid':
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3, strides=2, padding='same', activation=None)
        x = enc_outs[-1]
        x = attention_layer(x)
        skips=reversed(enc_outs[:-1])
        z = x
        #z = tf.keras.layers.Flatten()(z)
        z=tf.keras.layers.GlobalAveragePooling2D()(z)
        z = tf.keras.layers.Dense(64, activation='relu')(z)
        z = tf.keras.layers.Dense(1, activation=None)(z)
        z = tf.keras.layers.Activation('sigmoid', name='class_out')(z)
        xg=z

        xg = tf.keras.backend.expand_dims(xg, axis=1)
        xg = tf.keras.backend.expand_dims(xg, axis=1)
        xg = tf.keras.layers.UpSampling2D(size=IMAGE_SIZE, interpolation='nearest')(xg)

        for up, skip in zip(up_stack, skips):
            x = up(x)
            cancat = tf.keras.layers.Concatenate()
            x = cancat([x, skip])

        x = last(x)
        xg=tf.keras.layers.Multiply()([x,xg])
        #xg=const_multiply(const=10.0)(xg)
        xg=tf.keras.layers.Conv2D(1,1,use_bias=False, kernel_initializer='zeros')(xg)
        x = tf.keras.layers.Add()([x, xg])
        x = tf.keras.layers.Activation('sigmoid', name='seg_out')(x)

    elif decoder_name == 'deep_lab_att':
        y = ASPP(enc_outs[-1])
        y=attention_layer(y)
        z = y
        z=tf.keras.layers.GlobalAveragePooling2D()(z)
        z = tf.keras.layers.Dense(64, activation='relu')(z)
        z = tf.keras.layers.Dense(1, activation=None)(z)
        z = tf.keras.layers.Activation('sigmoid', name='class_out')(z)

        y = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 1, padding='same', activation=None, name='final_logits')(y)
        y = tf.keras.layers.UpSampling2D((16, 16), interpolation='bilinear', name='upsampled')(y)
        x = tf.keras.layers.Activation('sigmoid', name='seg_out')(y)

    elif decoder_name=='deep_lab_att_2':
        y = ASPP(enc_outs[-1])
        y = attention_layer(y)

        # classification branch
        z = y
        z = tf.keras.layers.GlobalAveragePooling2D()(z)
        z = tf.keras.layers.Dense(64, activation='relu')(z)
        z = tf.keras.layers.Dense(1, activation=None)(z)
        z = tf.keras.layers.Activation('sigmoid', name='class_out')(z)

        # Channel attention
        xg = z
        xg = tf.keras.backend.expand_dims(xg, axis=1)
        xg = tf.keras.backend.expand_dims(xg, axis=1)
        xg = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='nearest')(xg)
        xg = tf.keras.layers.Multiply()([y, xg])
        # xg=const_multiply(const=10.0)(xg)
        xg = tf.keras.layers.Conv2D(1, 1, use_bias=False, kernel_initializer='zeros')(xg)
        y = tf.keras.layers.Add()([y, xg])


        y = tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 1, padding='same', activation=None, name='final_logits')(y)
        y = tf.keras.layers.UpSampling2D((16, 16), interpolation='bilinear', name='upsampled')(y)

        x = tf.keras.layers.Activation('sigmoid', name='seg_out')(y)

    return tf.keras.Model(inputs=inputs, outputs=[z,x])



