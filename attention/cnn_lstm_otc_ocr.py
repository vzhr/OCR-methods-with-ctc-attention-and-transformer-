# -*- coding: utf-8 -*-
import tensorflow as tf
import utils
import resnet
from tensorflow.python.layers.core import Dense
import numpy as np

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


    def build_graph(self, X, Y_in, Y_out, length):
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
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([X, Y_in, Y_out, length], capacity=2 * len(self.gpus))
        decodes , tower_grads = [], [] 
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu in self.gpus:
                with tf.device('/gpu:%s' % gpu), tf.name_scope("GPU_%s" % gpu) as scope:
                    image_batch, label_in_batch, label_out_batch, length_batch = batch_queue.dequeue()
                    grads, train_output_result, pred_output_result = self._build_model(image_batch, label_in_batch, label_out_batch, length)
                    tf.get_variable_scope().reuse_variables()
                    tower_grads.append(grads), decodes.append(pred_output_result)
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

    def _build_model(self, X, label_in_batch, label_out_batch, length_batch):
        length_batch = tf.reshape(length_batch, [FLAGS.batch_size])

        filters = [FLAGS.image_channel, 64, 128, 256, FLAGS.out_channels]
        strides = [1, 2]
        feature_h = FLAGS.image_height
        feature_w = FLAGS.image_width
        x = X
        for i in range(FLAGS.cnn_count):
            with tf.variable_scope('unit-%d' % (i + 1)):
                x = self._conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[0])
                x = self._batch_norm('bn%d' % (i + 1), x)
                x = self._leaky_relu(x, FLAGS.leakiness)
                x = self._max_pool(x, 2, strides[1])
        _, feature_h, feature_w, _ = x.get_shape().as_list()
        # encoder part
        with tf.variable_scope('encoder'):
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            # treat `feature_w` as max_timestep in lstm.
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            x = tf.reshape(x, [FLAGS.batch_size, feature_w, -1])
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
            
            #cell =tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            #if self.mode == 'train':
            #    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)
            #cell1 = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            #if self.mode == 'train':
            #    cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=FLAGS.output_keep_prob)
            #stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            #initial_state = cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

            #encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=cell, inputs=x, sequence_length=self.seq_len,
            #                                                   initial_state=initial_state, dtype=tf.float32,
            #                                                   time_major=False)
            '''cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.num_hidden)
            if self.mode == "train":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)

            cell1 = tf.contrib.rnn.GRUCell(num_units=FLAGS.num_hidden)
            if self.mode == "train":
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=FLAGS.output_keep_prob)
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)'''
            
            stack1 = self._create_rnn_cell()
            stack2 = self._create_rnn_cell()
            enc_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack1, cell_bw=stack2, inputs=x, dtype=tf.float32)
            encoder_outputs = tf.concat(enc_outputs, -1)
            #encoder_state = tf.concat(enc_state, -1)
            #encoder_state = tf.reduce_mean(encoder_state, 0)
            encoder_state = encoder_state[0]

        with tf.variable_scope("decoder"):
            with tf.variable_scope("decoder_embedding"):
                decoder_embedding = tf.Variable(tf.truncated_normal(shape=[utils.num_classes, utils.embedding_dim], stddev=0.1), name="decoder_embedding")
            with tf.device('/cpu:0'):
                label_in_embedding = tf.nn.embedding_lookup(decoder_embedding, tf.cast(label_in_batch, tf.int32))
                label_in_embedding = tf.nn.dropout(label_in_embedding, FLAGS.output_keep_prob)
                
            start_token = tf.fill([FLAGS.batch_size], utils.TOKEN['<GO>'])   
            train_helper = tf.contrib.seq2seq.TrainingHelper(label_in_embedding, utils.train_length)
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embedding, start_tokens=tf.to_int32(start_token), end_token=utils.TOKEN['<EOS>'])
          
            train_outputs = self._decode(train_helper, encoder_outputs, 'decode', encoder_state, None)
            pred_outputs = self._decode(pred_helper, encoder_outputs, 'decode', encoder_state, True)
            
            train_decode_result = train_outputs[0].rnn_output[:, :-1, :]
            pred_decode_result = pred_outputs[0].sample_id

            #mask = tf.cast(tf.sequence_mask(FLAGS.batch_size*[utils.train_length[0]-1], utils.train_length[0]), tf.float32)
            #target_max_length = tf.reduce_max(length_batch)
            mask = tf.cast(tf.sequence_mask(length_batch , utils.max_len), tf.float32)
            att_loss = tf.contrib.seq2seq.sequence_loss(train_outputs[0].rnn_output, label_out_batch, weights=mask)
            #label_out_batch = tf.one_hot(label_out_batch, utils.num_classes)
            #att_loss = tf.losses.softmax_cross_entropy(onehot_labels=label_out_batch, logits=train_outputs[0].rnn_output, weights=mask, label_smoothing=0.05)
           
        self.loss = tf.reduce_mean(att_loss)
        grads = self.opt.compute_gradients(self.loss)
        return grads, train_decode_result, pred_decode_result

    def _create_rnn_cell(self):
        def single_rnn_cell():
            cell = tf.contrib.rnn.GRUCell(FLAGS.num_hidden)
            if self.mode == "train":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.output_keep_prob)
            return cell
        cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(2)], state_is_tuple=True)
        return cell

    def _decode(self, helper, memory, scope, encoder_state, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=FLAGS.num_hidden, memory=memory)
            #cell = tf.contrib.rnn.GRUCell(num_units=FLAGS.num_hidden)
            #cell = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden)
            #if self.mode == "train":
            #    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.output_keep_prob)
            cell = self._create_rnn_cell()
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=FLAGS.num_hidden, output_attention=True)
            output_layer = Dense(units=utils.num_classes)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell, helper=helper, 
                                                      initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=FLAGS.batch_size).clone(cell_state=encoder_state),
                                                      output_layer=output_layer)
            outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=utils.max_len)
        return outputs

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
