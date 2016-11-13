from __future__ import print_function
from six.moves import xrange
import tensorflow.contrib.slim as slim

import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as ly

def main():
    unrolled_step = 3
    unrolled_rate = 0.01
    batch_size = 32
    xx_raw = tf.placeholder(dtype=tf.float32, shape=(64, 28,28,1))
    xx = xx_raw
    size_li = [1, 64, 128, 256]
    with tf.variable_scope('discriminator0'):
        for j in xrange(3):
            with tf.variable_scope('Conv' if j==0 else 'Conv_{}'.format(j)):
                # print([3,3, size_li[j], size_li[j+1]])
                weights = slim.model_variable('weights', shape=[3,3, size_li[j], size_li[j+1]],
                                    initializer=ly.xavier_initializer())
                xx = conv_lrelu_bn(xx, weights, stride=2)
        # xx = ly.conv2d(xx, 128, 3, stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        # xx = ly.conv2d(xx, 256, 3, stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        xx = tf.reshape(xx, (2*batch_size, -1))
        with tf.variable_scope('fully_connected'):
            weights = slim.model_variable('weights', shape=[xx.get_shape().as_list()[-1], 1],
                                        initializer=ly.xavier_initializer())
            bias = slim.model_variable('bias', shape=(1,), initializer=tf.zeros_initializer)
        xx = tf.squeeze(fully_connected(xx, weights, bias))
        f = tf.identity(tf.reduce_mean(tf.log(1-xx[:batch_size])) + tf.reduce_mean(tf.log(xx[:batch_size])), name='f0')
    va_li = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator0')
    
    print([va.name for va in va_li])
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # grad_var = opt.compute_gradients(f, va_li)
    # print(grad_var)  
    # new_var = [var+unrolled_rate*grad for grad, var in grad_var]  
    # print(new_var)
    for i in xrange(unrolled_step):
        xx = xx_raw
        with tf.variable_scope('discriminator{}'.format(i+1)):
            grad_var = list(zip(tf.gradients(f, va_li), va_li))
            new_var = [var+unrolled_rate*grad if grad is not None else 1 for grad, var in grad_var]
            for j in xrange(3):
                with tf.variable_scope('Conv' if j==0 else 'Conv_{}'.format(j)):
                    va_li[2*j] = add_model_variable_wrapper(tf.identity(new_var[2*j], name='weights'))
                    # tf.get_variable('weights', initializer=tf.contrib.layers.xavier_initializer(), shape=(3, 3, 1, 64))
                    # print([c.name for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator1')])
                    # print(va_li[2*j].name)
                    with tf.variable_scope('BatchNorm'):
                        va_li[2*j+1] = add_model_variable_wrapper(tf.identity(new_var[2*j+1], name='beta'))
            with tf.variable_scope('fully_connected'):
                va_li[-2] = add_model_variable_wrapper(tf.identity(va_li[-2], name='weights'))
                va_li[-1] = add_model_variable_wrapper(tf.identity(va_li[-1], name='biases'))
            xx = conv_lrelu_bn(xx, va_li[0], 2)
            xx = conv_lrelu_bn(xx, va_li[2], 2)
            xx = conv_lrelu_bn(xx, va_li[4], 2)
            xx = tf.reshape(xx, (2*batch_size, -1))
            xx = tf.squeeze(fully_connected(xx, va_li[-2], va_li[-1]))
            f = tf.identity(tf.reduce_mean(tf.log(1-xx[:batch_size])) + tf.reduce_mean(tf.log(xx[:batch_size])), name='f{}'.format(i+1))

    print(f.name)
    # print([c.name for c in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator1')])

def conv_lrelu_bn(xx, kernel, stride=2, padding = 'SAME', name='conv_relu_bn'):
    output = tf.nn.conv2d(xx, kernel, [1, stride, stride, 1], padding)
    output = ly.batch_norm(output, activation_fn=lrelu)
    return output
# ['discriminator0/Conv/weights:0', 'discriminator0/Conv/BatchNorm/beta:0', 'discriminator0/Conv_1/weights:0', 'discriminator0/Conv_1/BatchNorm/beta:0', 'discriminator0/Conv_2/weights:0', 'discriminator0/Conv_2/BatchNorm/beta:0', 'discriminator0/fully_connected/weights:0', 'discriminator0/fully_connected/biases:0']

def fully_connected(xx, weights, bias):
    output = tf.nn.bias_add(tf.matmul(xx, weights), bias)
    output = lrelu(output)
    return output

def add_model_variable_wrapper(x):
    slim.add_model_variable(x)
    return x

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

if __name__ == '__main__':
    main()
