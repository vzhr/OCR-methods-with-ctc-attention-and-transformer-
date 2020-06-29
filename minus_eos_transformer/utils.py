# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import cv2
import codecs
from data_aug import DataAug

#train   batch_size mode 
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_boolean('allow_ffn_pad', True, '')
tf.app.flags.DEFINE_integer('image_height', 64, 'image height')
tf.app.flags.DEFINE_integer('image_width', 256, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')
tf.app.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 512, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('hidden_size', 512, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 300000, 'maximum epochs')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('decay_steps', 2000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')
tf.app.flags.DEFINE_integer('max_stepsize', 2, 'num of max step')
tf.app.flags.DEFINE_integer('vocab_size', 40, 'number of different words')
tf.app.flags.DEFINE_integer('extra_decode_length', 50, '')
tf.app.flags.DEFINE_integer('num_hidden_layers', 4, '')
tf.app.flags.DEFINE_integer('beam_size', 1, '')
tf.app.flags.DEFINE_integer('num_heads', 8, '')
tf.app.flags.DEFINE_integer('filter_size', 1024, '')
tf.app.flags.DEFINE_float('relu_dropout', 0.1, '')
tf.app.flags.DEFINE_float('alpha', 0.6, '')
tf.app.flags.DEFINE_float('attention_dropout', 0.1, '')
tf.app.flags.DEFINE_float('layer_postprocess_dropout', 0.1, "")
tf.app.flags.DEFINE_float('initializer_gain', 1.0, '')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_float('decay_weight', 0.0000005, 'L2 regularization')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, 'inital lr')
tf.app.flags.DEFINE_float('output_keep_prob', 0.75, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')
tf.app.flags.DEFINE_string('train_file', '/data/users/yiweizhu/ocr/IIIT5K/org/train_img.txt', 'the train data dir')
#tf.app.flags.DEFINE_string('train_file', '/data/users/yiweizhu/ocr/syn_90/img_list/train_img_to_4000000.txt', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', '.', 'the val data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('batch_size', 50, 'the batch_size')
tf.app.flags.DEFINE_integer('num_threads', 20, 'the step to validation')
tf.app.flags.DEFINE_string('output_dir', '/data/users/yiweizhu/ocr/IIIT5K/org/predict', 'output dir')
tf.app.flags.DEFINE_string('output_dir_truth', '/data/users/yiweizhu/ocr/IIIT5K/org/truth', '')
tf.app.flags.DEFINE_string('infer_file', '/data/users/yiweizhu/ocr/IIIT5K/org/test_img.txt', 'the infer data dir')

tf.app.flags.DEFINE_string('gpus', '0', 'train, val or infer')
tf.app.flags.DEFINE_string('map_file', '/data/users/yiweizhu/ocr/map_37.txt', "")
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'the checkpoint dir')
FLAGS = tf.app.flags.FLAGS

#print(FLAGS.checkpoint_dir)
params = {}
encode_maps = {}
decode_maps = {}

params = FLAGS.flag_values_dict()

with open(FLAGS.map_file, "r") as f:
    for line in f.readlines():
        if not line:
            continue
        char, i = line[0], int(line[2:])
        encode_maps[char] = i
        decode_maps[i] = char

index = len(decode_maps) + 3
SPACE_INDEX = index
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN
print('max_value', max(encode_maps.values()))

TOKEN={"<GO>":0, "<EOS>":1, "<PAD>":0}
num_classes = FLAGS.vocab_size
max_len = 27
image_length = 64
image_height_length = 4
image_width_length = 16

class DataIterator:
    def __init__(self, random_shuff = True, is_val = False):
        self.images = []
        self.labels_out = []
        self.lengths = []
        #self.annos = []
        self.data_file = FLAGS.infer_file if is_val else FLAGS.train_file
        self.random_shuff = random_shuff
        print("random_shuffle: ", self.random_shuff)
        self.is_val = is_val
        self.data_aug = DataAug()
        with open(self.data_file, "r") as f:   
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            anno_file = os.path.splitext(line)[0] + ".txt"
            with open(anno_file, 'r') as f1:
                annotation = f1.read().strip().lower()
                #print(annotation)
            #self.annos.append(annotation)

            '''dir_path = "/home/yiweizhu/yiweizhu/img/gray_iii/train"
            name = os.path.basename(line)
            line = os.path.join(dir_path, name)
            print(line)'''

            self.images.append(line)
            if annotation == SPACE_TOKEN:
                code = [SPACE_INDEX]
            else :
                code = [encode_maps.get(c, TOKEN["<PAD>"]) for c in list(annotation)]
            if len(code) >= 23: 
                print(anno_file)
                print(len(code))
            self.labels_out.append(code)
            
        #max_len = max(map(len, self.labels_out)) + 3
        #print(max_len)

        for i in range(len(self.labels_out)):
            self.lengths.append([len(self.labels_out[i])+1])

        for label in self.labels_out:
            label.append(int(TOKEN['<EOS>']))
            while len(label) < max_len:
                label.append(int(TOKEN['<PAD>']))
        
        #self.images_copy = self.images.copy()
        #self.length = []
        #for i in range(len(self.images)):
        #    self.length.append(image_length)

    def distored_inputs(self):
        filename, label_out, length_word = tf.train.slice_input_producer([self.images, self.labels_out, self.lengths], shuffle=self.random_shuff)
        num_preprocess_threads = FLAGS.num_threads
        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            image_buffer = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.image_channel)
            if not self.is_val:
                image = self.data_aug.run(image)
                
            initial_height = tf.shape(image)[0]
            initial_width = tf.shape(image)[1]
            ratio = tf.to_float(initial_height) / tf.constant(FLAGS.image_height, dtype=tf.float32)
            new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
            padding_width = tf.to_int32(FLAGS.image_width) - new_width
            
            def reshape_undirect():
                length = tf.to_int32(tf.to_float(new_width) / tf.constant(FLAGS.image_width, dtype=tf.float32) * image_width_length) + 1
                #length = tf.constant(image_length)
                length *= image_height_length
                image_ = tf.image.resize_images(image, (FLAGS.image_height, new_width))
                image_ = tf.concat([image_, tf.fill([FLAGS.image_height, padding_width, FLAGS.image_channel], 245.0)], 1)
                image_.set_shape([FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                return image_, length
            
            def reshape_direct():
                image_ = tf.image.resize_images(image, (FLAGS.image_height, FLAGS.image_width))
                length = tf.constant(image_length)
                return image_, length
            
            image_, length = tf.cond(tf.math.less(new_width, tf.constant(FLAGS.image_width)), reshape_undirect, reshape_direct)
            images_and_labels.append([filename, image_, label_out, length, length_word])

        filenames, images, label_out_batch, length_batch, length_word_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=FLAGS.batch_size,
                capacity=5 * num_preprocess_threads * FLAGS.batch_size)
         
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([filenames, images, label_out_batch, length_batch, length_word_batch], capacity = 5)
        filenames, images, label_out_batch, length_batch, length_word_batch = batch_queue.dequeue()

        return filenames, images, tf.reshape(label_out_batch, [FLAGS.batch_size, -1]), tf.reshape(length_batch, [FLAGS.batch_size, -1]), tf.reshape(length_word_batch, [FLAGS.batch_size, -1])
