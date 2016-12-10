import tensorflow.contrib.slim as slim
import tensorflow as tf
import tensorflow.contrib.layers as ly


def conv_lrelu_bn(xx, kernel, stride=2, padding = 'SAME', name='conv_relu_bn'):
    with tf.variable_scope(name):
        output = tf.nn.conv2d(xx, kernel, [1, stride, stride, 1], padding)
        # output = tf.nn.bias_add(output, bias)
        output = ly.batch_norm(output, activation_fn=lrelu)
    return output

def conv_lrelu(xx, kernel, bias, stride=2, padding = 'SAME', name='conv_relu'):
    with tf.variable_scope(name):
        output = tf.nn.conv2d(xx, kernel, [1, stride, stride, 1], padding)
        output = lrelu(output + bias)
    return output


def fully_connected(xx, weights, bias):
    output = tf.matmul(xx, weights) + bias
    # output = tf.nn.sigmoid(output)
    return output

def add_model_variable_wrapper(x):
    slim.add_model_variable(x)
    return x

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)