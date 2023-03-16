#!/usr/bin/env python3
import tensorflow as tf

def VDSR(input_shape, filters=64, kernel_initializer='he_normal'):
    """VDSR model architecture (Very Deep Super-Resolution Neural Network).

    - 'he_normal' weights initializer
    - 64 filters per layer
    - 20 convolutional layers
    - parametric rectifying linear unit (PReLU) as activation

    Reference: 
    J. Kim, J. K. Lee, and K. M. Lee, 
    “Accurate Image Super-Resolution Using Very Deep Convolutional Networks,” 
    in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2016, pp. 1646–1654. 
    doi: 10.1109/CVPR.2016.182.

    Parameters
    ----------
    input_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    filters : int
        Number of filters per layer
    kernel_initializer : string
        Kernel initializer to be used as defined by keras.initializers
    
    Returns
    -------
    keras.Model
    """

    # Initialize a parametric linear rectifier unit
    para_relu = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.constant(0.25))

    # Create the neural network
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, activation=para_relu, kernel_initializer=kernel_initializer, padding='same') (input)

    for _ in range(19):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, activation=para_relu, kernel_initializer=kernel_initializer, padding='same') (x)

    x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, kernel_initializer=kernel_initializer, padding='same') (x)
    model = tf.keras.Model(input, x, name="VDSR")

    return model

def IRUNet(input_shape, filters=64, kernel_initializer='he_normal'):
    """IRUNet model architecture.

    - 'he_normal' weights initializer
    - 64 filters per layer
    - rectifying linear unit (ReLU) as activation

    Reference:
    F. H. Gil Zuluaga, F. Bardozzo, J. I. Rios Patino, and R. Tagliaferri, 
    “Blind microscopy image denoising with a deep residual and multiscale encoder/decoder network,” 
    in 2021 43rd Annual International Conference of the IEEE Engineering in Medicine Biology Society (EMBC), Nov. 2021, pp. 3483–3486. 
    doi: 10.1109/EMBC46164.2021.9630502.


    Parameters
    ----------
    input_shape : tuple[int]
        Input shape in the form of (# pixels in x, # pixels in y, 1)
    filters : int
        Number of filters per layer
    kernel_initializer : string
        Kernel initializer to be used as defined by keras.initializers
    
    Returns
    -------
    keras.Model
    """

    # Define the activiation function
    activation_function = "relu"

    # Define the inputs
    inputs = tf.keras.Input(shape=input_shape)

    # Define the inception blocks
    def inception_block(inputVector, nf):
        branch_a = tf.keras.layers.Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (inputVector)
        branch_b = tf.keras.layers.Conv2D(filters=nf*2, kernel_size=3, strides=1, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (inputVector)
        branch_c = tf.keras.layers.Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', activation=activation_function, kernel_initializer=kernel_initializer, dilation_rate=2) (inputVector)
        
        concat = tf.keras.layers.Concatenate() ([branch_a, branch_b, branch_c])
        filter_reduction = tf.keras.layers.Conv2D(filters=nf, kernel_size=1, strides=1, padding='same') (concat)
        shortcut = tf.keras.layers.Add() ([inputVector, filter_reduction])
        return shortcut

    def inception_block_reduction(inputVector, nf):
        shortcut = tf.keras.layers.Conv2D(filters=nf, kernel_size=2, strides=2, padding='same') (inputVector)
        branch_a = tf.keras.layers.Conv2D(filters=nf, kernel_size=3, strides=2, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (inputVector)
        branch_b = tf.keras.layers.Conv2D(filters=nf*2, kernel_size=3, strides=2, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (inputVector)
        branch_c = tf.keras.layers.AveragePooling2D(padding='same') (inputVector)

        concat = tf.keras.layers.Concatenate() ([branch_a, branch_b, branch_c])
        filter_reduction = tf.keras.layers.Conv2D(filters=nf, kernel_size=1, strides=1, padding='same') (concat)
        shortcut = tf.keras.layers.Add() ([shortcut, filter_reduction])
        return shortcut

    # Define the encoder
    head = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (inputs)
    conv1 = inception_block_reduction(head, filters)
    conv1 = inception_block(conv1, filters)
    conv2 = inception_block_reduction(conv1, filters)
    conv2 = inception_block(conv2, filters)
    conv3 = inception_block_reduction(conv2, filters)
    conv3 = inception_block(conv3, filters)
    body = inception_block_reduction(conv3, filters)
    body = inception_block(body, filters)

    # Define the decoder
    deconv3 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (body)
    deconv3 = inception_block(deconv3, filters)
    deconv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (deconv3)
    deconv2 = inception_block(deconv2, filters)
    deconv2 = tf.keras.layers.Add() ([conv2, deconv2])
    deconv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (deconv2)
    deconv1 = inception_block(deconv1, filters)
    deconv1 = tf.keras.layers.Add() ([conv1, deconv1])
    tail = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same', activation=activation_function, kernel_initializer=kernel_initializer) (deconv1)
    tail = inception_block(tail, filters)
    tail = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid') (tail)

    return tf.keras.Model(inputs, tail, name="IRUNet")
