# -*- coding: utf-8 -*-
import tensorflow as tf
import utils
import resnet


FLAGS = utils.FLAGS
num_classes = utils.num_classes
MOVING_AVERAGE_DECAY = 0.99
def average_gradients(tower_grads):
    average_grads = []

    #print('tower_grads:', tower_grads)

    # 枚举所有的变量和变量在不同GPU上计算得出的梯度
    for grad_and_vars in zip(*tower_grads):
        # 计算所有GPU上的梯度平均值

        #print('grad_and_vars:', grad_and_vars)

        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # 将变量和它的平均梯度对应起来
        average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这将被用于变量更新
    return average_grads

class LSTMOCR(object):
    def __init__(self, mode, gpus):
        self.mode = mode
        self.gpus = gpus

    def build_graph_for_export(self, X):
        with tf.variable_scope(tf.get_variable_scope()):
            gpu = self.gpus
            print(gpu)
            with tf.device('/gpu:%s' % gpu), tf.name_scope("GPU_%s" % gpu) as scope:
                logits = self._build_model(X)
                decoded, log_prob = \
                        tf.nn.ctc_beam_search_decoder(logits,
                                        self.seq_len,
                                        beam_width=5,
                                        merge_repeated=False)
                dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        return dense_decoded, log_prob


    def build_graph(self, X, Y):
        self.global_step = tf.train.get_or_create_global_step()
        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    self.global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.decay_rate,
                                    staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                                 beta1=FLAGS.beta1,
                                                 beta2=FLAGS.beta2)
        #self.opt = tf.train.GradientDescentOptimizer(self.lrn_rate)
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([X, Y], capacity=2 * len(self.gpus))
        decodes , tower_grads = [], [] 
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu in self.gpus:
                with tf.device('/gpu:%s' % gpu), tf.name_scope("GPU_%s" % gpu) as scope:
                    image_batch, label_batch = batch_queue.dequeue()
                    logits = self._build_model(image_batch)
                    dense_decoded, grads = self._build_train_op(logits, label_batch)
                    tf.get_variable_scope().reuse_variables()
                    tower_grads.append(grads), decodes.append(dense_decoded)
        grads_avg = average_gradients(tower_grads)
        apply_gradient_op = self.opt.apply_gradients(grads_avg, global_step=self.global_step)
        self.variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                              self.global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variable_averages_op = self.variable_averages.apply(variables_to_average)
        #update_op = tf.get_collection(resnet.UPDATE_OPS_COLLECTION)
        #with tf.control_dependencies(update_op):
        train_op = tf.group(apply_gradient_op, variable_averages_op)
        return train_op, decodes

    def _build_model(self, X):
        filters = [FLAGS.image_channel, 64, 128, 256, FLAGS.out_channels]
        strides = [1, 2]
        x = X
        for i in range(FLAGS.cnn_count):
            with tf.variable_scope('unit-%d' % (i + 1)):
                x = self._conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[0])
                x = self._batch_norm('bn%d' % (i + 1), x)
                x = self._leaky_relu(x, FLAGS.leakiness)
                x = self._max_pool(x, 2, strides[1])
        #x = resnet.resnet_backbone(x, is_training=self.mode == 'train')
        _, feature_h, feature_w, _ = x.get_shape().as_list()
        # LSTM part
        with tf.variable_scope('lstm'):
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            # treat `feature_w` as max_timestep in lstm.
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            # x = tf.reshape(x, [FLAGS.batch_size, feature_w, feature_h * filters[-1]])
            x = tf.reshape(x, [FLAGS.batch_size, feature_w, -1])
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
            # print('self.seq_len.shape: {}'.format(self.seq_len.shape.as_list()))

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)

            cell1 = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=FLAGS.output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(FLAGS.batch_size, dtype=tf.float32)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=x,
                sequence_length=self.seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )  # [batch_size, max_stepsize, FLAGS.num_hidden]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])  # [batch_size * max_stepsize, FLAGS.num_hidden]

            W = tf.get_variable(name='W_out',
                                shape=[FLAGS.num_hidden, num_classes],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(mean=0, stddev=1))  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            shape = tf.shape(x)
            logits = tf.reshape(logits, [shape[0], -1, num_classes])
            # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            return logits

    def _build_train_op(self, logits, Y):
        # self.global_step = tf.Variable(0, trainable=False)
        Y = utils.dense_to_sparse(Y)
        loss = tf.nn.ctc_loss(labels=Y,
                                   inputs=logits,
                                   sequence_length=self.seq_len,
                                   # preprocess_collapse_repeated = True,
                                   # ctc_merge_repeated=False,
                                  )
        #all_vars   = tf.trainable_variables()
        #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in all_vars ]) * FLAGS.decay_weight
        #self.loss = tf.reduce_mean(loss) + lossL2
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('learning_rate', self.lrn_rate)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)
        grads = self.opt.compute_gradients(self.loss)
        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = \
             tf.nn.ctc_beam_search_decoder(logits,
                                           self.seq_len,
                              			beam_width=5,             
                                           merge_repeated=False)
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)
        return dense_decoded, grads

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
            x_bn = \
                tf.contrib.layers.batch_norm(
                    inputs=x,
                    decay=0.9,
                    center=True,
                    scale=True,
                    epsilon=1e-5,
                    updates_collections=None,
                    is_training=self.mode == 'train',
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=True,
                    scope='BatchNorm'
                )

        return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
