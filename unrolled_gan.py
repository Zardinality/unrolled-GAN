from __future__ import print_function


import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as ly

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "256", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "1e-3",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_bool("pt", "False", "Include pull away loss term")
tf.flags.DEFINE_integer("unrolled_step", "1",
                        "unrolled step during optimization")
tf.flags.DEFINE_string("mode", "train", "train - visualize")

z = tf.random_normal((FLAGS.batch_size, FLAGS.z_dim))
with tf.name_scope('generator'):
    train = ly.fully_connected(z, 4 * 4 * 512)
    train = tf.reshape(train, (-1, 4, 4, 512))
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                      activation_fn=lrelu, normalizer_fn='batch_norm')
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                      activation_fn=lrelu, normalizer_fn='batch_norm')
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                      activation_fn=lrelu, normalizer_fn='batch_norm')
    train = ly.conv2d_transpose(train, 1, 3, stride=1,
                      activation_fn=lrelu, normalizer_fn='batch_norm')
real_data = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, 28, 28))
xx = tf.concat(0, [train, real_data])
with tf.name_scope('discriminator'):
    xx = ly.conv2d(xx, 64, 3, stride=2, activation_fn=lrelu, normalizer_fn='batch_norm')
    xx = ly.conv2d(xx, 128, 3, stride=2, activation_fn=lrelu, normalizer_fn='batch_norm')
    xx = ly.conv2d(xx, 256, 3, stride=2, activation_fn=lrelu, normalizer_fn='batch_norm')
    xx = tf.reshape(xx, (-1, ))
    xx = ly.fully_connected(xx, 1)

d_loss = tf.reduce_mean((1-xx[:FLAGS.batch_size])) + 


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

if __name__ == "__main__":
    tf.app.run()
