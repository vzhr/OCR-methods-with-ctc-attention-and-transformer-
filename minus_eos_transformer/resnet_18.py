#_*_coding:utf-8_*_
import utils

FLAGS = utuils.FLAGS

in_channel = FLAGS.image_channel
out_channel = FLAGS.out_channels

filters = [in_channel, 64, 64, 128, 256, out_channel]
strides = [1, 2]
def resnet_backbone(x, is_training=True):
    x = conv2d(x, "begin", 3, filters[0], filters[1], strides[1])
    for idx, filter_num in enumerate(filters):
        if idx < 2:
            continue
        with tf.variable_scope("unit-%d" % (idx)):
            x = block_1(x, filters[i-1], filters[i], idx)
            x = bolck_2(x, filters[i-1], filters[i], idx)
    return x

def block_1(x, filters_num1, filters_num2, idx):
    short_cut = x

    name = str(idx) + "block" + "_1_1"
    with tf.variable_scope(name):
        name_cnn = name + "_cnn"
        x = conv2d(x, name_cnn, 3, filters_num1, filters_num2, strides[1])
        name_norm = name + "_batchnorm"
        x = batch_norm(name_norm, x, is_training)
        x = leaky_relu(x, FLAGS.leakiness)

    name_ = str(idx) + "block" + "_1_2"
    with tf.variable_scope(name_):
        name_cnn = name + "_cnn"
        x = conv2d(x, name_cnn, 3, filters_num1, filters_num2, strides[0])
        name_norm = name + "_batchnorm"
        x = batch_norm(name_norm, x, is_training)

    with tf.variable_scope("shortcut_block_1"):
        short_cut = conv2d(short_cut, "short_cut", 3, filters_num1, filters_num2, strides[1] )
    return leaky_relu(short_cut + x, FLAGS.leakiness)

def block_2(x, filters_num1, filters2_num2, idx):
    short_cut = x
    name = str(idx) + "block" + "_2_1"
    with tf.variable_scope(name):
        name_cnn = name + "_cnn"
        x = conv2d(x, name_cnn, 3, filters_num1, filter_num2, strides[0])
        name_norm = name + "_batchnorm"
        x = batch_norm(name_norm, x, is_training)
        x = leaky_relu(x, FLAGS.leakiness)

    name_ = str(idx) + "block" + "_2_2"
    with tf.variable_scope(name):
        name_cnn = name + "_cnn"
        x = conv2d(x, name_cnn, 3, filters_num1, filter_num2, strides[0])
        name_norm = name + "_batchnorm"
        x = batch_norm(name_norm, x, is_training)
    return leaky_relu(short_cut+x, FLAGS.leakiness)

def conv2d(x, name, filter_size, filter_in, filter_out, strides):
    with tf.variable_scope(name):
        kernel = tf.get_variable(name="W", shape=[filter_size, filter_size, filters_in, filters_out], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=1))
        b = tf.get_variable(name="B", shape = [filter_out], dtype=tf.float32, initializer=tf.constrant_initializer())
        con2_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding="SAME")
    return tf.nn.bias_add(con2_op, b)
def batch_norm(name, x, is_training):
    with tf.variable_scope(name):
        x_bn = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, center=True, scale=True, epsilon=1e-5, update_collections=None, is_training=is_training, fused=True, data_format='NHWC', zero_debias_moving_mean=True, scope='BatchNorm')
    return x_bn

def _leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")
def block_2(x, filters_num1, filters2_num2, idx):
    name = str(idx) + "block" + "_2"
    name_cnn = name + "_cnn"
    x = conv2d(x, name_cnn, 3, filters_num1, filter_num2, strides[0])
    name_norm = name + "_batchnorm"
    x = batch_norm(name_norm, x, is_training)
    x = leaky_relu(x, FLAGS.leakiness)

def conv2d(x, name, filter_size, filter_in, filter_out, strides):
    with tf.variable_scope(name):
        kernel = tf.get_variable(name="W", shape=[filter_size, filter_size, filters_in, filters_out], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=1))
        b = tf.get_variable(name="B", shape = [filter_out], dtype=tf.float32, initializer=tf.constrant_initializer())
        con2_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding="SAME")
    return tf.nn.bias_add(con2_op, b)

def batch_norm(name, x, is_training):
    with tf.variable_scope(name):
        x_bn = tf.contrib.layers.batch_norm(inputs=x, decay=0.9, center=True, scale=True, epsilon=1e-5, update_collections=None, is_training=is_training, fused=True, data_format='NHWC', zero_debias_moving_mean=True, scope='BatchNorm')
    return x_bn

def _leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")
