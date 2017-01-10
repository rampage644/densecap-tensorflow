'''Train the model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import os
import json

import tensorflow as tf
import numpy as np
import scipy.misc

import densecap.model as model
import densecap.util as util


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('ckpt_dir', os.path.join(os.getcwd(), 'ckpt'), 'Directory for model checkpoints')
tf.app.flags.DEFINE_string('image', '', 'Image for proposals generation')
tf.app.flags.DEFINE_string('output', 'output.png', 'Output image with bounding boxes')
tf.app.flags.DEFINE_integer('proposals', 50, 'Number of proposals to generate')


def main(_):
    '''entry point'''

    image_input = tf.placeholder(tf.float32, shape=[1, None, None, 3])

    vgg16 = model.VGG16(image_input)
    rpn = model.RegionProposalNetwork(vgg16.layers['conv5_3'])

    saver = tf.train.Saver()
    saved_model = tf.train.latest_checkpoint(FLAGS.ckpt_dir)

    with tf.Session() as sess:
        if saved_model:
            saver.restore(sess, saved_model)
        else:
            print('Can\'t find saved checkpoint, exiting.')
            return 1

        k = FLAGS.proposals
        image = scipy.misc.imread(FLAGS.image, mode='RGB')
        height, width, _ = image.shape
        fraction = 720.0 / max(height, width)
        image = scipy.misc.imresize(image, fraction)
        height, width, _ = image.shape

        boxes, scores = sess.run(
            [rpn.proposals, tf.nn.softmax(rpn.scores)], {
                rpn.image_height: height,
                rpn.image_width: width,
                vgg16.input: [image]
            })
        proposals = np.squeeze(boxes[np.argsort(scores[:, 1])][-k:])

        # [y_min, x_min, y_max, x_max]
        # floats 0.0 - 1.0
        ymin, xmin, h, w = np.split(proposals, 4, axis=1)

        xmax, ymax = xmin + w, ymin + h
        xmin, xmax, ymin, ymax = xmin / w, xmax / w, ymin / h, ymax / h

        images = tf.placeholder(tf.float32, [1, height, width, 3])
        boxes = tf.placeholder(tf.float32, [1, k, 4])
        bbox_image = tf.image.draw_bounding_boxes(images, boxes)
        output_images = sess.run(bbox_image, {
            images: [image],
            boxes: np.expand_dims(np.hstack([ymin, xmin, ymax, xmax]), axis=0)
        })

        scipy.misc.imsave(FLAGS.output, output_images[0])


if __name__ == '__main__':
    tf.app.run()
