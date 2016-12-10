from __future__ import print_function
from six.moves import xrange
from utils import *
import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as ly

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "100", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate_dis", "2e-4",
                      "Learning rate for Adam Optimizer for discriminator")
tf.flags.DEFINE_float("learning_rate_ger", "1e-4",
                      "Learning rate for Adam Optimizer for generator")
tf.flags.DEFINE_integer('max_iter_step', "100000", "the name says itself")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")

tf.flags.DEFINE_string("mode", "train", "train - visualize")
tf.flags.DEFINE_string("log_dir", "./dc_log", "dir to store summary")
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


def discriminator(img, label):
    size = 64
    label = tf.tile(label, [2, 1])
    yb = tf.reshape(label, [2*FLAGS.batch_size, 1, 1, 10])
    img = tf.concat(3, [img, yb*tf.ones([2*FLAGS.batch_size, 32, 32, 10])])
    img = ly.conv2d(img, num_outputs=size, kernel_size=3,
                    stride=2, activation_fn=lrelu)
    img = tf.concat(3, [img, yb*tf.ones([2*FLAGS.batch_size, 16, 16, 10])])
    img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
                    stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
                    stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    logit = ly.fully_connected(tf.reshape(
        img, [2 * FLAGS.batch_size, -1]), 1, activation_fn=None)
    return logit

def build_graph():
    z = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.z_dim))
    real_img = tf.placeholder(
        dtype=tf.float32, shape=(FLAGS.batch_size, 32, 32, 1))
    real_label = tf.placeholder(
        dtype=tf.float32, shape=(FLAGS.batch_size, 10))
    real_data = [real_img, real_label]
    with tf.variable_scope('generator'):
        train = generator(z, real_label)
    xx = tf.concat(0, [train, real_img])
    with tf.variable_scope('discriminator'):
        logit = discriminator(xx, real_label)

    fake_logit = logit[:FLAGS.batch_size]
    true_logit = logit[FLAGS.batch_size:]
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        fake_logit, tf.zeros_like(fake_logit))
    d_loss_true = tf.nn.sigmoid_cross_entropy_with_logits(
        true_logit, tf.ones_like(true_logit))
    d_loss = tf.reduce_mean(d_loss_fake + d_loss_true)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        fake_logit, tf.ones_like(fake_logit)))
    g_loss_sum = tf.scalar_summary("g_loss", g_loss)
    d_loss_sum = tf.scalar_summary("d_loss", d_loss)
    img_sum = tf.image_summary("img", train, max_images=10)
    d_loss = tf.Print(d_loss, [d_loss, g_loss], first_n=1000)
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    theta_d = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    opt_g = tf.train.AdamOptimizer(
        FLAGS.learning_rate_ger).minimize(g_loss, var_list=theta_g)
    opt_d = tf.train.AdamOptimizer(
        FLAGS.learning_rate_dis).minimize(d_loss, var_list=theta_d)
    return g_loss_sum, d_loss_sum, img_sum, opt_g, opt_d, z, real_data


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.device('/gpu:1'):    
        g_loss_sum, d_loss_sum, img_sum, opt_g, opt_d, z, real_data = build_graph()
    summary_g = tf.merge_summary([g_loss_sum, img_sum])
    summary_d = tf.merge_summary([d_loss_sum, img_sum])
    saver = tf.train.Saver()
    npad = ((0, 0), (2, 2), (2, 2))
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        for i in xrange(FLAGS.max_iter_step):
            train_data = mnist.train.next_batch(FLAGS.batch_size)
            train_img = np.reshape(train_data[0], (-1, 28, 28))
            train_img = np.pad(train_img, pad_width=npad,
                               mode='constant', constant_values=0)
            train_img = np.expand_dims(train_img, -1)
            batch_z = np.random.uniform(-1, 1, [FLAGS.batch_size, FLAGS.z_dim]) \
                .astype(np.float32)
            feed_dict = {real_data[0]: train_img, z: batch_z, real_data[1]:train_data[1]}
            if i % 100 == 99:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged = sess.run([opt_g, summary_g], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata {}'.format(i), i)
                _, merged = sess.run([opt_g, summary_g], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'second_generator_metadata {}'.format(i), i)
                _, merged = sess.run([opt_d, summary_d], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'discriminator_metadata {}'.format(i), i)
            else:
                sess.run(opt_g, feed_dict=feed_dict)
                sess.run(opt_g, feed_dict=feed_dict)
                sess.run(opt_d, feed_dict=feed_dict)
            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    FLAGS.ckpt_dir, "model.ckpt"), global_step=i)

if __name__ == '__main__':
    main()
