# -*- coding: utf-8 -*-
"""
SSD demo
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import random
from ssd_300_vgg import SSD
from utils import preprocess_image, process_bboxes
from visualization import bboxes_draw_on_img

tf.reset_default_graph() #重设图
ssd_net = SSD()
classes, scores, bboxes = ssd_net.detections()
images = ssd_net.images()

sess = tf.Session()
# Restore SSD model.
ckpt_filename = './ssd_checkpoints/ssd_vgg_300_weights.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

img_old = cv2.imread('./demo/dog.jpg')
img = cv2.cvtColor(img_old , cv2.COLOR_BGR2RGB)
img_prepocessed = preprocess_image(img)
#去均值归一化
rclasses, rscores, rbboxes = sess.run([classes, scores, bboxes],
                                      feed_dict={images: img_prepocessed})
rclasses, rscores, rbboxes = process_bboxes(rclasses, rscores, rbboxes)
colors = []
for i in range(20):
    colors.append((random.random(),random.random(),random.random()))
#plt_bboxes(img, rclasses, rscores, rbboxes)
bboxes_draw_on_img(img_old , rclasses, rscores, rbboxes, colors, thickness=2)