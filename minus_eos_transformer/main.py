# -*- coding: utf-8 -*-
import datetime
import logging
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import transformer_ocr
import utils
import sys

FLAGS = utils.FLAGS

logger = logging.getLogger('Training for OCR using Transformer')
logger.setLevel(logging.INFO)
def train(train_dir=None, mode='train'):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        gpus = list(filter(lambda x: x,  FLAGS.gpus.split(',')))
        model = transformer_ocr.LSTMOCR(mode, gpus)
        train_feeder = utils.DataIterator()
        filenames, X, Y_out, length, length_word = train_feeder.distored_inputs()
        train_op, _ = model.build_graph(X, Y_out, length, length_word)
        print('len(labels):%d, batch_size:%d'%(len(train_feeder.labels_out), FLAGS.batch_size))
        num_batches_per_epoch = int(len(train_feeder.labels_out) / FLAGS.batch_size / len(gpus))
        summary_merge = tf.summary.merge_all()
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement = False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print('restore from checkpoint{0}'.format(ckpt))
                    #sess.run(tf.assign(model.global_step, FLAGS.num_epochs*num_batches_per_epoch/3))
            print('=============================begin training=============================')
            start_time = time.time()
            for cur_epoch in range(FLAGS.num_epochs):
                #start_time = time.time()
                # the training part
                for cur_batch in range(num_batches_per_epoch):
                    summary, res, step = sess.run([summary_merge, train_op, model.global_step])
                    #print("step ", step)
                    train_writer.add_summary(summary, step)
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
                    if step % 50 == 1:
                        loss, lrn_rate = sess.run([model.loss, model.lrn_rate])
                        print('step: %d, batch: %d time: %d, learning rate: %.8f, loss:%.4f' %( step, cur_batch,time.time() - start_time, lrn_rate, loss))
            coord.request_stop()
            coord.join(threads)


def infer(mode='infer'):
    gpus = list(filter(lambda x: x,  FLAGS.gpus.split(',')))
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        infer_feeder = utils.DataIterator(is_val = True, random_shuff = False)
        filenames, X, Y_out, length, length_word = infer_feeder.distored_inputs()
        model = transformer_ocr.LSTMOCR(mode, gpus)
        train_op, decodes = model.build_graph(X, Y_out, length, length_word)
        total_steps = int((len(infer_feeder.images) + FLAGS.batch_size - 1) / FLAGS.batch_size) + 5
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            variables_to_restore = model.variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)  

            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            print("search from ", FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print('restore from ckpt{}'.format(ckpt))
            else:
                print('cannot restore')
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            count = 0
            start_time = time.time() 
            print(total_steps)
            for curr_step in range(total_steps):
                decoded_expression = [] 
                filenames_, dense_decoded_code, targets = sess.run([filenames, decodes, Y_out])
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
                
                target_expression = []
                for sentence in targets:
                    target_se = ''
                    for code in sentence:
                        if code == utils.TOKEN['<EOS>']:
                            break
                        if code not in utils.decode_maps:
                            target_se += ''
                        else:
                            target_se += utils.decode_maps[code]
                    target_expression.append(target_se)
                for filename, target_, pred_ in zip(filenames_, target_expression, decoded_expression):
                    print('name      ', filename)
                    count += 1
                    print(count)
                    print('target    ', target_)
                    print('pred      ', pred_)
                    #print('anno      ', anno_.decode('utf-8'))
                    print()
                    filename = os.path.splitext(os.path.basename(filename.decode()))[0] + '.txt'
                    output_filename = os.path.join(FLAGS.output_dir, filename)
                    truth_filename = os.path.join(FLAGS.output_dir_truth, filename)
                    with open(output_filename, 'w') as f:
                        f.write(pred_)
                    with open(truth_filename, 'w') as f:
                        f.write(target_)

            print("time:", time.time()- start_time)
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
