#! /usr/bin/env python
# coding=utf-8


import tensorflow as tf


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):

    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01), regularizer=tf.contrib.layers.l2_regularizer(5e-4))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)


        if bn:
            conv = tf.layers.batch_normalization(conv, momentum =0.995, epsilon = 1e-3, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):

    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1,   filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output



def route(name, previous_output, current_output):

    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)

    return output


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            # resize images to size using nearest neighbor interpolation
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())

    return output

def separable_conv(name, input_data, input_c, output_c, trainable, downsample=False):
    with tf.variable_scope(name):
        with tf.variable_scope('depthwise'):
            if downsample:
                pad_data = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
                input_data = tf.pad(input_data, pad_data, 'CONSTANT')
                strides = (1, 2, 2, 1)
                padding = 'VALID'
            else:
                strides = (1, 1, 1, 1)
                padding = "SAME"
            dwise_weight = tf.get_variable(name='depthwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(3, 3, input_c, 1),
                                           initializer=tf.random_normal_initializer(stddev=0.01),regularizer=tf.contrib.layers.l2_regularizer(5e-4))

            dwise_conv = tf.nn.depthwise_conv2d(input=input_data, filter=dwise_weight, strides=strides, padding=padding)
            dwise_conv = tf.layers.batch_normalization(dwise_conv, momentum = 0.995, epsilon = 1e-3,beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
            # dwise_conv = batch_normalization(input_data=dwise_conv, input_c=input_c, trainable=trainable)
            dwise_conv = tf.nn.leaky_relu(dwise_conv, alpha=0.1)

        with tf.variable_scope('pointwise'):
            pwise_weight = tf.get_variable(name='pointwise_weights', dtype=tf.float32, trainable=True,
                                           shape=(1, 1, input_c, output_c),
                                           initializer=tf.random_normal_initializer(stddev=0.01),regularizer=tf.contrib.layers.l2_regularizer(5e-4))
            pwise_conv = tf.nn.conv2d(input=dwise_conv, filter=pwise_weight, strides=(1, 1, 1, 1), padding="SAME")
            # pwise_conv = batch_normalization(input_data=pwise_conv, input_c=output_c, trainable=trainable)
            pwise_conv = tf.layers.batch_normalization(pwise_conv, momentum =0.995, epsilon = 1e-3, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
            pwise_conv =tf.nn.leaky_relu(pwise_conv, alpha=0.1)
        return pwise_conv


def cbam_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        # print("CBAM Hello")
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat

def BasicSepConv(input_data, filters_shape, rate_1, trainable , name, activate=True, bn=True):

    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01),regularizer=tf.contrib.layers.l2_regularizer(5e-4))
        conv = tf.nn.atrous_conv2d(input_data, weight, rate_1, padding = "SAME")

        if bn:
            conv = tf.layers.batch_normalization(conv,momentum = 0.995, epsilon = 1e-3, beta_initializer=tf.zeros_initializer(),
                                                     gamma_initializer=tf.ones_initializer(),
                                                     moving_mean_initializer=tf.zeros_initializer(),
                                                     moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                       dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv

def RFB_net(input_data ,trainable, name, stride=1,scale =0.1, activate=True, bn=True):

    with tf.variable_scope(name):

        conv = convolutional(input_data, (1, 1, 1024, 256), trainable, 'conv_rfb_1')

        conv_1 = separable_conv('conv_rfb_2', conv, 256, 256, trainable)

        conv_1_1 = BasicSepConv(conv_1, (3,3,256,256), 3, trainable,'BasicSepConv_1')

        conv_2_1 = BasicSepConv(conv_1,(3,3,256,256), 5, trainable,'BasicSepConv_2')

        output = tf.concat([conv_1, conv_1_1, conv_2_1, conv], axis=3)

        output = convolutional(output, (1, 1, 1024, 1024), trainable,'conv_rfb_out', activate=False)

        output = output + input_data

        output  = tf.nn.leaky_relu(output, alpha=0.1)


    return output





