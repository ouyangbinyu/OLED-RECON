import tensorflow as tf
import numpy as np
# from keras import *
# import keras

def model(inpt):
    with tf.variable_scope('inpute_layer'):
        conv_in = conv_relu(inpt, [3, 3, 2, 64], 1)
        # num = tf.shape(inpt)[2]

    # ############# jiangwei ###################

    with tf.variable_scope('u_net'):
        # conv1 = conv_relu(conv_in, [3, 3, 64, 64], 1)
        # conv1 = conv_relu(conv1, [3, 3, 64, 64], 1)
        conv1 = residual_block(conv_in, 64)
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')  ##  128*128

        # conv2 = conv_relu(pool1, [3, 3, 64, 64], 1)
        # conv2 = conv_relu(conv2, [3, 3, 64, 64], 1)
        conv21 = conv_relu(pool1, [3, 3, 64, 128], 1)
        conv2 = residual_block(conv21, 128)
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##  64*64

        # conv3 = conv_relu(pool2, [3, 3, 64, 64], 1)
        # conv3 = conv_relu(conv3, [3, 3, 64, 64], 1)
        conv31 = conv_relu(pool2, [3, 3, 128, 256], 1)
        conv3 = residual_block(conv31, 256)
        pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##  32*32

        # conv4 = conv_relu(pool3, [3, 3, 64, 64], 1)
        # conv4 = conv_relu(conv4, [3, 3, 64, 64], 1)
        conv41 = conv_relu(pool3, [3, 3, 256, 512], 1)
        conv4 = residual_block(conv41, 512)

        # num1 = np.zeros(1, dtype=np.float32)
        # num1 = num / 4
        up3 = deconv2(conv4, 512, 256, 2, 2)
        # up3 = deconv2(up3, 64, 64, 2, 2)
        # up3 = tf.nn.conv2d_transpose(conv4, [64, 64, 64, 64], [64, 2, 2, 64], [1, 2, 2, 1])
        up3 = tf.concat((conv3, up3), 3)
        conv5 = conv_relu(up3, [3, 3, 512, 256], 1)
        # conv5 = conv_relu(conv5, [3, 3, 64, 64], 1)
        conv5 = residual_block(conv5, 256)

        # num2 = np.zeros(1, dtype=np.float32)
        # num2 = num / 2
        up2 = deconv2(conv5, 256, 128, 2, 2)
        # up3 = deconv2(up3, 64, 64, 2, 2)
        # up3 = tf.nn.conv2d_transpose(conv4, [64, 64, 64, 64], [64, 2, 2, 64], [1, 2, 2, 1])
        up2 = tf.concat((conv2, up2), 3)
        conv6 = conv_relu(up2, [3, 3, 256, 128], 1)
        # conv6 = conv_relu(conv6, [3, 3, 64, 64], 1)
        conv6 = residual_block(conv6, 128)

        up1 = deconv2(conv6, 128, 64, 2, 2)
        # up3 = deconv2(up3, 64, 64, 2, 2)
        # up3 = tf.nn.conv2d_transpose(conv4, [64, 64, 64, 64], [64, 2, 2, 64], [1, 2, 2, 1])
        up1 = tf.concat((conv1, up1), 3)
        conv7 = conv_relu(up1, [3, 3, 128, 64], 1)
        # conv7 = conv_relu(conv7, [3, 3, 64, 64], 1)
        conv7 = residual_block(conv7, 64)
        out = conv_relu(conv7, [3, 3, 64, 1], 1)
        # out = conv_relu(conv, [3, 3, 64, 1], 1)

    return out

        # kernel = []
        # dconv1 = tf.nn.conv2d_transpose(conv8, [3, 3, 64, 64], [64, 48, 48, 64], [1, 2, 2, 1], padding = 'SAME')


def deconv2(x, input_filter, output_filter, kernel, strides):
    with tf.variable_scope('conv_transpose'):
        shape = [kernel, kernel, output_filter, input_filter]
        weight = tf.Variable(tf.truncated_normal(shape, stddev = 0.1), name = 'weight')
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]*strides
        width = tf.shape(x)[2]*strides
        output_shape = tf.stack([batch_size, height, width, output_filter])
    return tf.nn.conv2d_transpose(x, weight, output_shape, strides = [1, strides, strides,1], name = 'conv_transpose')









####################################################下面是函数定义######################################################
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv_prelu(inpt, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    biases = bias_variable([filter_shape[3]])
    pre_relu = tf.nn.bias_add(conv, biases)
    out = prelu(pre_relu)

    return out


def conv_bn_relu(inpt, filter_shape, stride, is_training):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    batch_norm = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)
    out = tf.nn.relu(batch_norm)

    return out


def conv_bn(inpt, filter_shape, stride, is_training):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    batch_norm = tf.contrib.layers.batch_norm(conv, scale=True, is_training=is_training, updates_collections=None)

    return batch_norm


def conv_relu(inpt, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    biases = bias_variable([filter_shape[3]])
    pre_relu = tf.nn.bias_add(conv, biases)
    out = tf.nn.relu(pre_relu)

    return out


def conv(inpt, filter_shape, stride):
    filter_ = weight_variable(filter_shape)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    biases = bias_variable([filter_shape[3]])
    out = tf.nn.bias_add(conv, biases)

    return out


def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1], initializer=tf.constant_initializer(0.25), dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg


def residual_block(inpt, output_depth):
    input_depth = inpt.get_shape().as_list()[3]

    with tf.variable_scope('conv1_in_block'):
        conv1 = conv_relu(inpt, [3, 3, input_depth, output_depth], 1)

    with tf.variable_scope('conv2_in_block'):
        conv2 = conv(conv1, [3, 3, output_depth, output_depth], 1)

    res = conv2 + inpt
    out = tf.nn.relu(res)

    return out
