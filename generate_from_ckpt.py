import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as ly
# import matplotlib.pyplot as plt
# import scipy.misc
import cv2

from utils import *

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "256", "size of input vector to generator")
tf.flags.DEFINE_string("ckpt_dir", "./ckpt", "dir to store checkpoint")


def generator(z):
    weights = slim.model_variable(
        'fn_weights', shape=(FLAGS.z_dim, 4 * 4 * 512), initializer=ly.xavier_initializer())
    bias = slim.model_variable(
        'fn_bias', shape=(4 * 4 * 512, ), initializer=tf.zeros_initializer)
    train = tf.nn.relu(fully_connected(z, weights, bias))
    train = tf.reshape(train, (-1, 4, 4, 512))
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 1, 3, stride=1,
                                activation_fn=None, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02), biases_initializer=None)
    bias = slim.model_variable('bias', shape=(
        1, ), initializer=tf.zeros_initializer)
    train += bias
    train = tf.nn.tanh(train)
    return train


def build_graph():
    z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
    with tf.variable_scope('generator'):
        train = generator(z)
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return z, train


def main():
    with tf.device('/cpu:0'):
        z, train = build_graph()
    if FLAGS.ckpt_dir != None:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            batch_z = np.random.normal(0, 1.0, [FLAGS.batch_size, FLAGS.z_dim]) \
                .astype(np.float32)
            rs = train.eval(feed_dict={z:batch_z})
    print(rs[0].shape)
    overall = []
    for i in range(8):
        temp = []
        for j in range(8):
            temp.append(rs[i * 8 + j])

        overall.append(np.concatenate(temp, axis=1))
    res = np.concatenate(overall, axis=0)
    res = cv2.cvtColor((res)*255, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('sample.png', res)

if __name__ == '__main__':
    main()
