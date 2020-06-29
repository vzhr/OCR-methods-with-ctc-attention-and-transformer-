# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
'''
可以结合numpy，或者cv， 里面的函数，然后image_t = tf.py_func(preprocess, [image_t], [tf.float32])
'''
class DataAug(object):
    def __init__(self, color_order = True):
        self.color_order = color_order
    def distort_color(self, image):
        if self.color_order == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)#亮度
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)#饱和度
            image = tf.image.random_hue(image, max_delta=0.2)#色相
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)#对比度
        if self.color_order == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        if self.color_order == 2:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        if self.color_order == 3:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        #return tf.clip_by_value(image, 0.0, 1.0)
        return image
    def random_rotate(self, image):
        r = tf.random_normal([1], stddev=0.087, seed = 1) #(NHWC)
        return tf.contrib.image.rotate(image, r)
    def run(self, image):
        #return self.distort_color(self.random_rotate(image))
        image = self.random_rotate(image)
 #       image = tf.py_func(self.add_py_noise, [image], tf.float32)
        return image
    def add_py_noise(self, X_img):
        return self.add_salt_pepper_noise(self.add_gaussian_noise(X_img))
    def add_salt_pepper_noise(self, X_img):
        # Need to produce a copy as to not modify the original image
        row, col, _ = X_img.shape
        salt_vs_pepper = 0.2
        amount = 0.002
        num_salt = np.ceil(amount * X_img.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_img.size * (1.0 - salt_vs_pepper))
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape[0:2]]
        X_img[coords[0], coords[1], :] = 1
        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape[0:2]]
        X_img[coords[0], coords[1], :] = 0
        # Gaussian distribution parameters
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        #gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 1.0, gaussian, 0.25, 0)
        return gaussian_img

    def augment(images, labels,
                resize=None, # (width, height) tuple or None
                horizontal_flip=False,
                vertical_flip=False,
                rotate=0, # Maximum rotation angle in degrees
                crop_probability=0, # How often we do crops
                crop_min_percent=0.6, # Minimum linear dimension of a crop
                crop_max_percent=1.,  # Maximum linear dimension of a crop
                mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
      if resize is not None:
        images = tf.image.resize_bilinear(images, resize)

      # My experiments showed that casting on GPU improves training performance
      if images.dtype != tf.float32:
        images = tf.image.convert_image_dtype(images, dtype=tf.float32)
        images = tf.subtract(images, 0.5)
        images = tf.multiply(images, 2.0)
      labels = tf.to_float(labels)

      with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
          coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
          flip_transform = tf.convert_to_tensor(
              [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
          transforms.append(
              tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
          coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
          flip_transform = tf.convert_to_tensor(
              [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
          transforms.append(
              tf.where(coin,
                       tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
          angle_rad = rotate / 180 * math.pi
          angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
          transforms.append(
              tf.contrib.image.angles_to_projective_transforms(
                  angles, height, width))

        if crop_probability > 0:
          crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                       crop_max_percent)
          left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
          top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
          crop_transform = tf.stack([
              crop_pct,
              tf.zeros([batch_size]), top,
              tf.zeros([batch_size]), crop_pct, left,
              tf.zeros([batch_size]),
              tf.zeros([batch_size])
          ], 1)

          coin = tf.less(
              tf.random_uniform([batch_size], 0, 1.0), crop_probability)
          transforms.append(
              tf.where(coin, crop_transform,
                       tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))
        if transforms:
          images = tf.contrib.image.transform(
              images,
              tf.contrib.image.compose_transforms(*transforms), interpolation='BILINEAR') # or 'NEAREST'
