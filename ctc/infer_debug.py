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
import cv2
ft_path = '/data/notebooks/yuandaxing1/ft_lib/'
sys.path.append(ft_path)
import ft2
FLAGS = utils.FLAGS
logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)
class Infer(object):
    def __init__(self):
        self.ft = ft2.put_chinese_text(os.path.join(ft_path, 'msyh.ttf'))
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
    def draw_debug(self, image, txt, suss = True):
        h, w, c = image.shape
        image2 = np.zeros((h+20, w+100, c))
        image_list = [image, image2]
        t_size = len(txt.decode('utf-8')) or 1
        size = min(w / t_size, h)
        color = (255, 0, 0) if suss else (0, 255, 0)
        image_list[1] = self.ft.draw_text(image_list[1], (10, 10), txt, size, color)
        image_list[0] = cv2.resize(image_list[0], (w+20, h+20))
        combine = np.concatenate(image_list, axis = 1)
        return combine
    def infer(self):
        FLAGS.num_threads = 1
        gpus = list(filter(lambda x: x,  FLAGS.gpus.split(',')))
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            train_feeder = utils.DataIterator(is_val = True, random_shuff = False)
            X, Y = train_feeder.distored_inputs()
            model = cnn_lstm_otc_ocr.LSTMOCR('infer', gpus)
            train_op, decodes = model.build_graph(X, Y)
            total_steps = (len(train_feeder.image) + FLAGS.batch_size - 1) / FLAGS.batch_size
            config = tf.ConfigProto(allow_soft_placement=True)
            result_dir = os.path.dirname(FLAGS.infer_file)
            with tf.Session(config=config) as sess, open(os.path.join(FLAGS.output_dir,  'result'), 'w') as f:
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                print(FLAGS.checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)
                    print('restore from ckpt{}'.format(ckpt))
                else:
                    print('cannot restore')
                count = 0
                for curr_step in range(total_steps):
                    decoded_expression = []
                    start = time.time()
                    dense_decoded_code = sess.run(decodes)
                    print('time cost:', (time.time() - start))
		    print("dense_decoded_code:", dense_decoded_code)
                    for d in dense_decoded_code:
                        for item in d:
                            expression = ''
                            for i in item:
                                if i not in utils.decode_maps:
                                    expression += ''
                                else:
                                    expression += utils.decode_maps[i]
                            decoded_expression.append(expression)
                    for code in decoded_expression:
                        if count >= len(train_feeder.image): break
                        f.write("%s,%s,%s\n"%(train_feeder.image[count], train_feeder.anno[count].encode('utf-8'), code.encode('utf-8')))
                        filename = os.path.splitext(os.path.basename(train_feeder.image[count]))[0] + ".txt"
                        output_file = os.path.join(FLAGS.output_dir, filename)
                        cur = open(output_file, "w")
                        cur.write(code.encode('utf-8'))
                        cur.close()
			print(code.encode('utf-8'))
                        try:
                            image_debug = cv2.imread(train_feeder.image[count])
                            image_debug = self.draw_debug(image_debug, code.encode('utf-8'), code == train_feeder.anno[count])
                            image_path = os.path.join(FLAGS.output_dir, os.path.basename(train_feeder.image[count]))
                            cv2.imwrite(image_path, image_debug)
                        except Exception as e:
                            print(e)
                        count+=1
                coord.request_stop()
                coord.join(threads)

def main(_):
    Infer().infer()
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
