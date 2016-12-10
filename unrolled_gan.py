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
tf.flags.DEFINE_integer("z_dim", "256", "size of input vector to generator")
tf.flags.DEFINE_float("learning_rate_dis", "2e-4",
                      "Learning rate for Adam Optimizer for discriminator")
tf.flags.DEFINE_float("learning_rate_ger", "2e-4",
                      "Learning rate for Adam Optimizer for generator")
tf.flags.DEFINE_integer('max_iter_step', "100000", "the name says itself")
tf.flags.DEFINE_float("beta1", "0.5", "beta1 for Adam optimizer")
tf.flags.DEFINE_float("unrolled_rate", "1e-4",
                      "unrolled rate, the \eta in the formula")
# tf.flags.DEFINE_bool("pt", "False", "Include pull away loss term")
# tf.flags.DEFINE_string("mode", "train", "train - visualize")
tf.flags.DEFINE_integer("unrolled_step", "1",
                        "unrolled step during optimization")
tf.flags.DEFINE_string("log_dir", "./log", "dir to store summary")
tf.flags.DEFINE_string("ckpt_dir", "./ckpt", "dir to store checkpoint")

batch_size = FLAGS.batch_size
unrolled_step = FLAGS.unrolled_step
unrolled_rate = FLAGS.unrolled_rate


def generator(z):
    # because up to now we can not derive bias_add's higher order derivative in tensorflow, 
    # so I use vanilla implementation of FC instead of a FC layer in tensorflow.contrib.layers
    # the following conv case is out of the same reason
    weights = slim.model_variable(
        'fn_weights', shape=(FLAGS.z_dim, 4 * 4 * 512), initializer=ly.xavier_initializer())
    bias = slim.model_variable(
        'fn_bias', shape=(4 * 4 * 512, ), initializer=tf.zeros_initializer)
    train = tf.nn.relu(ly.batch_norm(fully_connected(z, weights, bias)))
    train = tf.reshape(train, (-1, 4, 4, 512))
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
    train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
    train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME')
    train = ly.conv2d_transpose(train, 1, 3, stride=1,
                                activation_fn=None, padding='SAME', biases_initializer=None)
    bias = slim.model_variable('bias', shape=(
        1, ), initializer=tf.zeros_initializer)
    train += bias
    train = tf.nn.tanh(train)
    return train



def discriminator(img, name, target):
    size = 64
    with tf.variable_scope(name):
        # img = ly.conv2d(img, num_outputs=size, kernel_size=3,
        #                 stride=2, activation_fn=None, biases_initializer=None)
        # bias = slim.model_variable('conv_bias', shape=(
        #     size, ), initializer=tf.zeros_initializer)
        # img += bias
        # img = lrelu(img)
        img = ly.conv2d(img, num_outputs=size, kernel_size=3,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
                        stride=2, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
        img = tf.reshape(img, (2 * batch_size, -1))
        weights = slim.model_variable('weights', shape=[img.get_shape().as_list()[-1], 1],
                                      initializer=ly.xavier_initializer())
        bias = slim.model_variable('bias', shape=(
            1,), initializer=tf.zeros_initializer)
        logit = fully_connected(img, weights, bias)
        fake_logit = logit[:FLAGS.batch_size]
        true_logit = logit[FLAGS.batch_size:]
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            fake_logit, tf.zeros_like(fake_logit)))
        d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            true_logit, tf.ones_like(true_logit)))
        f = tf.reduce_mean(d_loss_fake + d_loss_true)

    return f, logit, d_loss_true, d_loss_fake


def build_graph():
    z = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.z_dim])
    with tf.variable_scope('generator'):
        train = generator(z)
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    real_data = tf.placeholder(
        dtype=tf.float32, shape=(FLAGS.batch_size, 32, 32, 1))

    xx = tf.concat(0, [train, real_data])
    xx_raw = tf.identity(xx)
    target = tf.concat(0, [np.zeros((batch_size, 1), dtype='float32'),
                           np.ones((batch_size, 1), dtype='float32')])
    with tf.variable_scope('discriminator'):
        xx = xx_raw
        f, logits, f_real, f_fake = discriminator(xx, 'discriminator0', target)
        va_li = list(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
        f0 = f
        theta_d = list(va_li)
        # unroll it
        for i in xrange(unrolled_step):
            xx = tf.identity(xx_raw)
            with tf.variable_scope('discriminator{}'.format(i + 1)):
                grad_var = list(zip(tf.gradients(f, va_li), va_li))
                # derive new \theta_D^K, the '1' is simply a placeholder for batch_norm variables, I choose to not update them.
                new_var = [var + unrolled_rate *
                           grad if grad is not None else 1 for grad, var in grad_var]
                # print(va_li)
                for j in xrange(3):
                    with tf.variable_scope('Conv' if j == 0 else 'Conv_{}'.format(j)):
                        va_li[
                            2 * j] = add_model_variable_wrapper(tf.identity(new_var[2 * j], name='weights'))
                        va_li[2 * j + 1] = add_model_variable_wrapper(
                            tf.identity(new_var[2 * j + 1], name='weights'))
                print([va.name for va in va_li])
                with tf.variable_scope('fully_connected'):
                    va_li[-2] = add_model_variable_wrapper(
                        tf.identity(new_var[-2], name='weights'))
                    va_li[-1] = add_model_variable_wrapper(
                        tf.identity(new_var[-1], name='biases'))
                xx = conv_lrelu_bn(xx, va_li[0], 2, name='conv_relu_bn')
                xx = conv_lrelu_bn(xx, va_li[2], 2, name='conv_relu_bn_1')
                xx = conv_lrelu_bn(xx, va_li[4], 2, name='conv_relu_bn_2')
                xx = tf.reshape(xx, (2 * batch_size, -1))
                xx = fully_connected(xx, va_li[-2], va_li[-1])
                sig = tf.nn.sigmoid_cross_entropy_with_logits(xx, target)
                f0 = tf.identity(tf.reduce_mean(sig),
                                 name='f{}'.format(i + 1))
                f_real = tf.reduce_mean(sig[FLAGS.batch_size:])
                f_fake = tf.reduce_mean(sig[:FLAGS.batch_size])
        if unrolled_step == 0:
            xx = logits
        fg = tf.reduce_mean(tf.log(tf.nn.sigmoid_cross_entropy_with_logits(
            xx[:FLAGS.batch_size], tf.ones_like(xx[:FLAGS.batch_size]))))
        fg0 = tf.reduce_mean(tf.log(tf.nn.sigmoid_cross_entropy_with_logits(
            logits[:FLAGS.batch_size], target[FLAGS.batch_size:])))
    with tf.device('/cpu:0'):
        tf.image_summary('image_fake', train)
        # tf.image_summary('image_real', real_data)
        tf.histogram_summary("z", z)
        tf.scalar_summary('fg for generator', fg)
        tf.scalar_summary('fg0 for generator', fg0)
        tf.scalar_summary('f for discriminator', f0)
        tf.scalar_summary('f_real for discriminator', f_real)
        tf.scalar_summary('f_fake for discriminator', f_fake)
        # f0 = tf.Print(f0, [fg, f0], first_n=1000)
    opt_g = tf.train.AdamOptimizer(
        FLAGS.learning_rate_ger, beta1=FLAGS.beta1).minimize(fg, var_list=theta_g)
    opt_d = tf.train.AdamOptimizer(
        FLAGS.learning_rate_dis, beta1=FLAGS.beta1).minimize(f0, var_list=theta_d)
    return real_data, z, opt_g, opt_d


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    with tf.device('/gpu:2'):
        real_data, z, opt_g, opt_d = build_graph()
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    npad = ((0, 0), (2, 2), (2, 2))
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
        for i in xrange(FLAGS.max_iter_step):
            train_img = mnist.train.next_batch(FLAGS.batch_size)[0]
            train_img = np.reshape(train_img, (-1, 28, 28))
            train_img = np.pad(train_img, pad_width=npad,
                               mode='constant', constant_values=0)
            train_img = np.expand_dims(train_img, -1)
            batch_z = np.random.normal(0, 1.0, [FLAGS.batch_size, FLAGS.z_dim]) \
                .astype(np.float32)
            feed_dict = {real_data: train_img, z: batch_z}
            if i % 100 == 99:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, merged = sess.run([opt_g, summary_op], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                _, merged = sess.run([opt_g, summary_op], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'generator_metadata{}'.format(i), i)
                _, merged = sess.run([opt_d, summary_op], feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(
                    run_metadata, 'discriminator_metadata{}'.format(i), i)
            else:
                sess.run(opt_g, feed_dict=feed_dict)
                sess.run(opt_d, feed_dict=feed_dict)
            if i % 1000 == 999:
                saver.save(sess, os.path.join(
                    FLAGS.ckpt_dir, "model.ckpt"), global_step=i)


if __name__ == "__main__":
    tf.app.run()
