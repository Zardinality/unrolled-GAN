import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as ly
from utils import *
import cv2


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "80", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_string("ckpt_dir", "./dc_ckpt", "dir to store checkpoint")


def generator(z, label):
    z = tf.concat(1, [z,label])
    train = ly.fully_connected(
        z, 1024, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm)
    train = tf.concat(1, [train, label])
    train = ly.fully_connected(
        z, 4*4*512, activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4, 512))
    yb = tf.ones([FLAGS.batch_size, 4, 4, 10])*tf.reshape(label, [FLAGS.batch_size, 1, 1, 10]) 
    train = tf.concat(3, [train, yb])
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 1, 3, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    return train

def build_graph():
    z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
    label = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 10])
    with tf.variable_scope('generator'):
        train = generator(z, label)
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return z, label, train



def main():
    with tf.device('/cpu:0'):
        z, label, train = build_graph()
    temp = np.repeat(np.arange(10), 8)
    lb = np.zeros((80, 10))
    lb[np.arange(80), temp] = 1
    if FLAGS.ckpt_dir != None:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            batch_z = np.random.normal(0, 1.0, [FLAGS.batch_size, FLAGS.z_dim]) \
                .astype(np.float32)
            rs = train.eval(feed_dict={z:batch_z, label:lb})
    print(rs[0].shape)
    overall = []
    for i in range(10):
        temp = []
        for j in range(8):
            temp.append(rs[i * 8 + j])

        overall.append(np.concatenate(temp, axis=1))
    res = np.concatenate(overall, axis=0)
    res = cv2.cvtColor((res)*255, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('sample.png', res)

if __name__ == '__main__':
    main()
