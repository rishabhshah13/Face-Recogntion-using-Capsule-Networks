import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks

K.set_image_data_format('channels_last')


def capsModel(input_shape, n_class, num_routing):

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=512, kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(x)
    BN1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv2')(BN1)
    BN2 = layers.BatchNormalization()(conv2)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv3')(BN2)
    BN3 = layers.BatchNormalization()(conv3)
    conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv4')(BN3)
    BN4 = layers.BatchNormalization()(conv4)
    conv5 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv5')(BN4)
    BN5 = layers.BatchNormalization()(conv5)
    conv6 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv6')(BN5)
    
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv6, dim_capsule=24, n_channels=32, kernel_size=12, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, num_routing=num_routing,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model



