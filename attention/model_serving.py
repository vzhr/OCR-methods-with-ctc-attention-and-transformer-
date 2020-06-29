# -*- coding: utf-8 -*-
from flask import Flask
from flask import request
import datetime, logging, os, time, cv2, sys, base64, json
import numpy as np
import tensorflow as tf
import cnn_lstm_otc_ocr
import utils
import helper
from tensorflow.python.platform import flags
from utils import *
from cnn_lstm_otc_ocr import *
tf_version = tf.__version__
model_dir = '/data/notebooks/yuandaxing1/OCR/CNN_LSTM_CTC_Tensorflow/checkpoint'
class Infer(object):
    def __init__(self, model_dir = model_dir):
        self.X = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel], name='input')
        model = cnn_lstm_otc_ocr.LSTMOCR('infer', '0')
        self.decodes, self.prob = model.build_graph_for_export(self.X)
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(model_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        saver.restore(self.sess, ckpt)
    def infer(self, images):
        decode, prob = self.sess.run([self.decodes, self.prob], feed_dict={self.X : images})
        decoded_expression = []
        for d, p in zip(decode, prob):
            codes = ['' if item not in utils.decode_maps else utils.decode_maps[item] for item in d ]
            expression = ''.join(codes)
            decoded_expression.append([expression, float(p[0])])
        return decoded_expression
infer = Infer()
app = Flask(__name__)
@app.route('/', methods=['POST'])
def index():
    ret = {'result': []}
    if not request.json or 'images' not in request.json:
        return json.dumps(ret)
    result = []
    for idx, image in enumerate(request.json['images']):
        name = image['name']
        content = base64.b64decode(image['content'])
        image = np.asarray(bytearray(content), dtype="uint8")
        channel = cv2.IMREAD_GRAYSCALE if FLAGS.image_channel == 1 else cv2.IMREAD_COLOR
        image_cv = cv2.imdecode(image, channel)
        image_cv = cv2.resize(image_cv, (FLAGS.image_width, FLAGS.image_height))
        if FLAGS.image_channel == 1 :
            image_cv = np.expand_dims(image_cv, axis = 2)
        result.append(image_cv)
    while len(result) % FLAGS.batch_size : result.append(result[-1])
    decode_text = []
    for i in range(0, len(result), FLAGS.batch_size):
        decode_text += infer.infer(result[i:i+FLAGS.batch_size])
    for idx, image in enumerate(request.json['images']):
        ret['result'].append({'name': image['name'],
                                'text' : decode_text[idx]})
    return json.dumps(ret)
if __name__ == '__main__':
    if tf_version <= '1.4.':
        flags.FLAGS._parse_flags()
    else :
        flags.FLAGS(sys.argv)
    app.run(host='0.0.0.0', debug=False, port=18888, threaded=True)
