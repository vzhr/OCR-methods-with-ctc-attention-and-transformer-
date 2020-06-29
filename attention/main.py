# -*- coding: utf-8 -*-
import datetime
import logging
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import cnn_lstm_otc_ocr
import utils
import helper
import sys

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)
def train(train_dir=None, mode='train'):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        gpus = list(filter(lambda x: x,  FLAGS.gpus.split(',')))
        model = cnn_lstm_otc_ocr.LSTMOCR(mode, gpus)
        train_feeder = utils.DataIterator()
        X, Y_in, Y_out, length = train_feeder.distored_inputs()
        train_op, _ = model.build_graph(X, Y_in, Y_out, length)
        print('len(labels):%d, batch_size:%d'%(len(train_feeder.labels_out), FLAGS.batch_size))
        num_batches_per_epoch = int(len(train_feeder.labels_out) / FLAGS.batch_size / len(gpus))
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement = False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print('restore from checkpoint{0}'.format(ckpt))
                    print('global_step:', model.global_step.eval())
                    print('assign value %d' % (FLAGS.num_epochs*num_batches_per_epoch/3))
                    #sess.run(tf.assign(model.global_step, FLAGS.num_epochs*num_batches_per_epoch/3))
            print('=============================begin training=============================')
            for cur_epoch in range(FLAGS.num_epochs):
                start_time = time.time()
                batch_time = time.time()
                # the training part
                for cur_batch in range(num_batches_per_epoch):
                    res, step = sess.run([train_op, model.global_step])
                    #print("step ", step)
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
                    if (step + 1) % 100 == 1:
                        print('step: %d, batch: %d time: %d, learning rate: %.8f, loss:%.4f' %( step, cur_batch,time.time() - batch_time, model.lrn_rate.eval(), model.loss.eval()))
            coord.request_stop()
            coord.join(threads)


def infer(mode='infer'):
    FLAGS.num_threads = 1
    gpus = list(filter(lambda x: x,  FLAGS.gpus.split(',')))
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        train_feeder = utils.DataIterator(is_val = True, random_shuff = False)
        X, Y_in, Y_out, length = train_feeder.distored_inputs()
        model = cnn_lstm_otc_ocr.LSTMOCR(mode, gpus)
        train_op, decodes = model.build_graph(X, Y_in, Y_out, length)
        total_steps = int((len(train_feeder.image) + FLAGS.batch_size - 1) / FLAGS.batch_size)
        config = tf.ConfigProto(allow_soft_placement=True)
        result_dir = os.path.dirname(FLAGS.infer_file)
        with tf.Session(config=config) as sess, open(os.path.join(result_dir,  'result_digit_v1.txt'), 'w') as f:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            #saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            #saver.restore(sess, './checkpoint_zhuyiwei/ocr-model-55001')
            variables_to_restore = model.variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)  

            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            print("search from ", FLAGS.checkpoint_dir)
            print(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print('restore from ckpt{}'.format(ckpt))
            else:
                print('cannot restore')
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            count = 0
            for curr_step in range(total_steps):
                decoded_expression = []
                
                dense_decoded_code = sess.run(decodes)
                #print('dense_decode', dense_decoded_code)
                for batch in dense_decoded_code:
                    for sequence in batch:
                        expression = ''
                        for code in sequence:
                            if code == utils.TOKEN["<EOS>"]:
                                break
                            if code not in utils.decode_maps:
                                expression += ''
                            else:
                                expression += utils.decode_maps[code]
                        decoded_expression.append(expression)
                for expression in decoded_expression:
                    if count >= len(train_feeder.image): break
                #    f.write("%s,%s,%s\n"%(train_feeder.image[count], train_feeder.anno[count].encode('utf-8'), code.encode('utf-8')))
                    print(train_feeder.image[count])
                    #print(train_feeder.anno[count].encode('utf-8'))
                    #print(expression.encode('utf-8'))
                    print(train_feeder.anno[count])
                    print(expression)
                    print('')
                    filename = os.path.splitext(os.path.basename(train_feeder.image[count]))[0] + ".txt"
                    output_file = os.path.join(FLAGS.output_dir, filename)
                    cur = open(output_file, "w")
                    #cur.write(expression.encode('utf-8'))
                    cur.write(expression)
                    cur.close()
                    count+=1

            coord.request_stop()
            coord.join(threads)

def main(_):
    if FLAGS.mode == 'train':
        train(FLAGS.train_file, FLAGS.mode)
    elif FLAGS.mode == 'infer':
        infer(FLAGS.mode)
    else :
        raise Exception('input the mode')
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
