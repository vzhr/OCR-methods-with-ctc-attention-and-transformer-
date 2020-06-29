# -*- coding: utf-8 -*-
from __future__ import print_function
import os.path
# This is a placeholder for a Google-internal import.
import tensorflow as tf
import cnn_lstm_otc_ocr
import utils
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
FLAGS = tf.app.flags.FLAGS

def export():
  with tf.device('/cpu:0'):
    with tf.Graph().as_default():
      serialized_tf_recognition = tf.placeholder(tf.string, name='tf_recognition')
      feature_configs = {
          'image/encoded': tf.FixedLenFeature(
              shape=[], dtype=tf.string),
      }
      tf_recognition = tf.parse_example(serialized_tf_recognition, feature_configs)
      jpegs = tf_recognition['image/encoded']
      images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
      model = cnn_lstm_otc_ocr.LSTMOCR('infer', '0')
      decodes = model.build_graph_for_export(images)
      with tf.device('/cpu:0'), tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        saver.restore(sess, ckpt)
        output_path = os.path.join(
          tf.compat.as_bytes(FLAGS.output_dir),
          tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        # Build the signature_def_map.
        classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
            serialized_tf_recognition)
        classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
          decodes)

        classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                      classify_inputs_tensor_info
              },
              outputs={
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                      classes_output_tensor_info
              },
              method_name=tf.saved_model.signature_constants.
              CLASSIFY_METHOD_NAME))

        predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
        prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': predict_inputs_tensor_info},
              outputs={
                  'classes': classes_output_tensor_info,
              },
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          ))
        builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'predict_images':
                  prediction_signature,
              tf.saved_model.signature_constants.
              DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                  classification_signature,
          }, 
          clear_devices=True)


        builder.save()
        print('Successfully exported model to %s' % FLAGS.output_dir)


def preprocess_image(image_buffer):
  image = tf.image.decode_jpeg(image_buffer, channels=1)
  image = tf.image.resize_images(image, (FLAGS.image_height,
                                             FLAGS.image_width))

  return image
def main(unused_argv=None):
  export()
if __name__ == '__main__':
  tf.app.run()
