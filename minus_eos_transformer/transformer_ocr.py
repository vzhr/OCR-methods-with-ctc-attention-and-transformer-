# -*- coding: utf-8 -*-
import tensorflow as tf
import utils
import resnet
from tensorflow.python.layers.core import Dense
import numpy as np
from transformer_official.model.transformer import Transformer

FLAGS = utils.FLAGS
params = utils.params

num_classes = utils.num_classes
MOVING_AVERAGE_DECAY = 0.99

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class LSTMOCR(object):
    def __init__(self, mode, gpus):
        self.mode = mode
        self.gpus = gpus

    def build_graph(self, X, Y_out, length, length_word):
        self.global_step = tf.train.get_or_create_global_step()
        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                          beta1=FLAGS.beta1,
                                          beta2=FLAGS.beta2)
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lrn_rate)
        #batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([X, Y_out, length], capacity=2 * len(self.gpus))
        
        decodes , tower_grads = [], [] 
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu in self.gpus:
                with tf.device('/gpu:%s' % gpu), tf.name_scope("GPU_%s" % gpu) as scope:
                    #image_batch,  label_out_batch, length_batch = batch_queue.dequeue()
                    #grads, prediction = self._build_model(image_batch, label_out_batch, length_batch)
                    grads, prediction = self._build_model(X, Y_out, length, length_word)
                    tf.get_variable_scope().reuse_variables()
                    tower_grads.append(grads), decodes.append(prediction)
        if FLAGS.mode == "train":
            print("training")
            grads_avg = average_gradients(tower_grads)
            apply_gradient_op = self.opt.apply_gradients(grads_avg, global_step=self.global_step)
        else:
            print("inference")
            apply_gradient_op = tf.constant(1)
        self.variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, self.global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variable_averages_op = self.variable_averages.apply(variables_to_average)
        
        update_op = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
        with tf.control_dependencies(update_op):
            train_op = tf.group(apply_gradient_op, variable_averages_op)
        return train_op, decodes

    def _build_model(self, X, label_out_batch, length_batch, length_word_batch):
        #length_batch = tf.reshape(length_batch, [FLAGS.batch_size])
        filters = [FLAGS.image_channel, 64, 128, 256, FLAGS.out_channels]
        print(filters)
        strides = [1, 2]
        x = X
        '''for i in range(FLAGS.cnn_count):
            with tf.variable_scope('unit-%d' % (i + 1)):
                x = self._conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[0])
                x = self._batch_norm('bn%d' % (i + 1), x)
                x = self._leaky_relu(x, FLAGS.leakiness)
                x = self._max_pool(x, 2, strides[1], strides[1])'''
        x = resnet.resnet_backbone(x, is_training=(self.mode=='train'))
        #x = self._max_pool(x, 2, 2, 1)
        _, feature_h, feature_w, feature_channel = x.get_shape().as_list()
        print("the shape of x is:", x.get_shape().as_list())
        x = tf.transpose(x, [0, 2, 1, 3])
        #x = tf.reshape(x, [FLAGS.batch_size, feature_w, -1])
        x = tf.reshape(x, [FLAGS.batch_size, -1, feature_channel])
        print("after reshape, the shape of x is:", x.get_shape().as_list())

        #transformer
        is_training = (self.mode == "train")
        transformer = Transformer(params, is_training)
        if is_training: 
            self.logits = transformer(x, length_batch, label_out_batch)
            #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=label_out_batch)
            self.labels = self.label_smoothing(tf.one_hot(label_out_batch, depth=utils.num_classes))
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
            
            length_word_batch = tf.reshape(length_word_batch, [FLAGS.batch_size])
            length_word_batch_1 = tf.sequence_mask(length_word_batch, utils.max_len, dtype=tf.float32)
            length_word_batch_2 = tf.one_hot(length_word_batch-1, utils.max_len) * 0.9
            length_word_batch_1_1 = length_word_batch_1 - length_word_batch_2
            #self.loss *= length_word_batch_1_1 
            #self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(length_word_batch_1)
            self.loss = tf.reduce_mean(self.loss)
            tf.summary.scalar('loss', self.loss)
            grads = self.opt.compute_gradients(self.loss)
            prediction = tf.constant(1)
            return grads, prediction
        else:
            grads = tf.constant(1)
            self.logits = transformer(x, length_batch, None)
            prediction = self.logits["outputs"]
            return grads, prediction        

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(mean=0, stddev=1))

            b = tf.get_variable(name='b',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            x_bn = tf.contrib.layers.batch_norm(inputs=x,
                                                decay=0.9,
                                                center=True,
                                                scale=True,
                                                epsilon=1e-5,
                                                updates_collections=None,
                                                is_training=(self.mode == 'train'),
                                                fused=True,
                                                data_format='NHWC',
                                                zero_debias_moving_mean=True,
                                                scope='BatchNorm' )
        return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides1, strides2):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides1, strides2, 1],
                              padding='SAME',
                              name='max_pool')
    
    def label_smoothing(self, inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1]
        return ((1 - epsilon) * inputs) + (epsilon / K)
