from __future__ import print_function
from six.moves import xrange
from utils import *
import tensorflow.contrib.slim as slim

import tensorflow as tf
import numpy as np

import tensorflow.contrib.layers as ly

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "64", "batch size for training")
tf.flags.DEFINE_integer("z_dim", "256", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate", "1e-3",
                      "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer('max_iter_step', "1e5", "the name says itself")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("unrolled_rate", "0.01", "unrolled rate, the \eta in the formula")
tf.flags.DEFINE_bool("pt", "False", "Include pull away loss term")
tf.flags.DEFINE_integer("unrolled_step", "1",
                        "unrolled step during optimization")
tf.flags.DEFINE_string("mode", "train", "train - visualize")
tf.flags.DEFINE_string("log_dir", "./log", "dir to store summary")
tf.flags.DEFINE_string("ckpt_dir", "./ckpt", "dir to store checkpoint")


def build_graph():
    z = tf.random_normal((FLAGS.batch_size, FLAGS.z_dim))
    with tf.variable_scope('generator'):
        train = ly.fully_connected(z, 4 * 4 * 512)
        train = tf.reshape(train, (-1, 4, 4, 512))
        train = ly.conv2d_transpose(train, 256, 3, stride=2,
                        activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        train = ly.conv2d_transpose(train, 128, 3, stride=2,
                        activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        train = ly.conv2d_transpose(train, 64, 3, stride=2,
                        activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        train = ly.conv2d_transpose(train, 1, 3, stride=1,
                        activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    real_data = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, 28, 28))
    tf.image_summary('image_fake', train)
    tf.image_summary('image_real', real_data)    
    tf.histogram_summary("z", z)
    xx = tf.concat(0, [train, real_data])
    xx_raw = xx
    batch_size = FLAGS.batch_size
    unrolled_step = FLAGS.unrolled_step
    unrolled_rate = FLAGS.unrolled_rate
    learning_rate = FLAGS.learning_rate
    with tf.variable_scope('discriminator'):
        xx = xx_raw
        size_li = [1, 64, 128, 256]
        with tf.variable_scope('discriminator0'):
            for j in xrange(3):
                with tf.variable_scope('Conv' if j==0 else 'Conv_{}'.format(j)):
                    weights = slim.model_variable('weights', shape=[3,3, size_li[j], size_li[j+1]],
                                        initializer=ly.xavier_initializer())
                    xx = conv_lrelu_bn(xx, weights, stride=2)
            xx = tf.reshape(xx, (2*batch_size, -1))
            with tf.variable_scope('fully_connected'):
                weights = slim.model_variable('weights', shape=[xx.get_shape().as_list()[-1], 1],
                                            initializer=ly.xavier_initializer())
                bias = slim.model_variable('bias', shape=(1,), initializer=tf.zeros_initializer)
            xx = tf.squeeze(fully_connected(xx, weights, bias))
            f = tf.identity(tf.reduce_mean(tf.log(1-xx[:batch_size])) + tf.reduce_mean(tf.log(xx[:batch_size])), name='f0')
        va_li = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator0')
        f0 = f
        theta_d = va_li
        print([va.name for va in va_li])
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        for i in xrange(unrolled_step):
            xx = xx_raw
            with tf.variable_scope('discriminator{}'.format(i+1)):
                grad_var = list(zip(tf.gradients(f, va_li), va_li))
                new_var = [var+unrolled_rate*grad if grad is not None else 1 for grad, var in grad_var]
                for j in xrange(3):
                    with tf.variable_scope('Conv' if j==0 else 'Conv_{}'.format(j)):
                        va_li[2*j] = add_model_variable_wrapper(tf.identity(new_var[2*j], name='weights'))

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
    tf.scalar_summary('fk for generator', f)
    tf.scalar_summary('f for discriminator', f0)
    opt_g = tf.train.AdamOptimizer(learning_rate).minimize(-f, var_list=theta_g)
    opt_d = tf.train.AdamOptimizer(learning_rate).minimize(f0, var_list=theta_d)
    return real_data, opt_g, opt_d
    # print(f.name)


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    real_data, opt_g, opt_d = build_graph()
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)
    with tf.Session() as sess:
        for i in xrange(FLAGS.max_iter_step):
            feed_dict = {real_data:mnist.train.next_batch(FLAGS.batch_size)}
            if i%100==99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged = sess.run([opt_g, summary_op], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'generator_metadata', i)
                _, merged = sess.run([opt_d, summary_op], feed_dict=feed_dict,
                        options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'discriminator_metadata', i)
            else:
                sess.run(opt_g, feed_dict=feed_dict)
                sess.run(opt_d, feed_dict=feed_dict)                
            if i%1000==999:
                saver.save(sess, FLAGS.ckpt_dir + "model.ckpt", global_step=i)




if __name__ == "__main__":
    tf.app.run()
