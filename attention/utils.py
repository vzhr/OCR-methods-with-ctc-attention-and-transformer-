# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import cv2
import codecs
from data_aug import DataAug

# +-* + () + 10 digit + blank + space
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 5e-5, 'inital lr')
tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
tf.app.flags.DEFINE_integer('image_width', 180, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')
tf.app.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_float('output_keep_prob', 0.75, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 3000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 40, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 5000, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')
tf.app.flags.DEFINE_integer('num_threads', 20, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_float('decay_weight', 0.0000005, 'L2 regularization')

tf.app.flags.DEFINE_integer('decay_steps', 5000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_file', './imgs/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_file', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_string('gpus', 'train', 'train, val or infer')
tf.app.flags.DEFINE_string('output_dir', '', 'output dir')
tf.app.flags.DEFINE_integer('max_stepsize', 2, 'num of max step')
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('map_file', '', "")
encode_maps = {}
decode_maps = {}


with open(FLAGS.map_file, "r") as f:
    for line in f.readlines():
        if not line:
            continue
        #line = line.decode('utf-8')
        char, i = line[0], int(line[2:])
        #print(char, " ", i)
        encode_maps[char] = i
        decode_maps[i] = char

size = len(decode_maps) + 1
SPACE_INDEX = size
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN
TOKEN={"<GO>":0, "<EOS>":1, "<PAD>":2}
num_classes = 4000
max_len = 38
embedding_dim = 20
train_length=np.array([max_len]*FLAGS.batch_size, dtype=np.int32)
print('max_value', max(encode_maps.values()))
print('max_code', max(decode_maps.keys()), 'num_classes', num_classes)
def dense_to_sparse(dense_tensor, out_type = tf.int32):
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(2, dense_tensor.dtype)))
    values=tf.gather_nd(dense_tensor, indices)
    shape=tf.cast(tf.shape(dense_tensor), tf.int64)
    return tf.SparseTensor(indices, values, shape)

class DataIterator:
    def __init__(self, random_shuff = True, is_val = False):
        self.image = []
        self.labels_out = []
        self.labels_in = []
        self.length = []
        self.anno = []
        self.data_file = FLAGS.infer_file if is_val else FLAGS.train_file
        self.random_shuff = random_shuff
        self.is_val = is_val
        self.data_aug = DataAug()
        with open(self.data_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            anno_file = os.path.splitext(line)[0] + ".txt"
            f = open(anno_file)
            #print(anno_file)
            #annotation = f.read().strip().decode('utf-8')
            annotation = f.read().strip()
            #print(annotation)
            f.close()
            self.anno.append(annotation)
            self.image.append(line)
            if is_val or annotation == SPACE_TOKEN:
                code = [SPACE_INDEX]
            else :
                code = [encode_maps[c] for c in list(annotation)]
            if len(code) >= 25 : 
                print(anno_file)
                print(len(code))
            self.labels_out.append(code)
        
        '''for label in self.labels_out:
            label.append(int(TOKEN['<EOS>']))

        for i in range(len(self.labels_out)):
            label_in = []
            label_in.append(int(TOKEN['<GO>']))
            for j in range(len(self.labels_out[i]) - 1):
                label_in.append(self.labels_out[i][j])
            self.labels_in.append(label_in)'''

        for i in range(len(self.labels_out)):
            self.length.append([len(self.labels_out[i])+1])
       
        self.labels_in = np.ones((len(self.labels_out), max_len), dtype=np.float32) + 1
        for i in range(len(self.labels_out)):
            self.labels_in[i][0] = int(TOKEN['<GO>'])
            for j in range(len(self.labels_out[i])):
                self.labels_in[i][j+1] = self.labels_out[i][j]
            while len(self.labels_in[i]) < max_len:
                self.labels_in[i].append(int(TOKEN['<PAD>']))

        for label in self.labels_out:
            label.append(int(TOKEN['<EOS>']))
            while len(label) < max_len:
                label.append(int(TOKEN['<PAD>']))

    def distored_inputs(self):
        '''max_len = max(map(len, self.labels))+1
        for e in self.labels:
            while len(e) < max_len:
                e.append(0)'''
        filename, label_in, label_out, length = tf.train.slice_input_producer([self.image, self.labels_in, self.labels_out, self.length], shuffle=self.random_shuff)
        num_preprocess_threads = FLAGS.num_threads
        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            image_buffer = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_buffer, channels=FLAGS.image_channel)
            #if not self.is_val: image = self.data_aug.run(image)
            
            initial_height = tf.shape(image)[0]
            initial_width = tf.shape(image)[1]
            ratio = tf.to_float(initial_height) / tf.constant(FLAGS.image_height, dtype=tf.float32)
            new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
            padding_width = tf.to_int32(FLAGS.image_width) - new_width

            image = tf.image.resize_images(image, (FLAGS.image_height, new_width))
            image = tf.concat([image, tf.fill([FLAGS.image_height, padding_width, FLAGS.image_channel], 255.0)], 1)
            image = tf.image.resize_images(image, (FLAGS.image_height, FLAGS.image_width))
            # if not self.is_val:
            #     image = tf.py_func(self.data_aug.add_salt_pepper_noise, [image], tf.float32)
            #     image = tf.reshape(image, ([FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel]))
            images_and_labels.append([image, label_in, label_out, length])
        images, label_in_batch, label_out_batch, length_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=FLAGS.batch_size,
                capacity=2 * num_preprocess_threads * FLAGS.batch_size)
        return images, tf.reshape(label_in_batch, [FLAGS.batch_size, -1]), tf.reshape(label_out_batch, [FLAGS.batch_size, -1]), length_batch

def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')
        if origin_label == decoded_label:
            count += 1
    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue
    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')
    return eval_rs
