'''Model tests'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import tensorflow as tf
import numpy as np
import scipy.misc

import densecap.model as model
import densecap.util as util

sess = tf.InteractiveSession()

def equal(a, b):
    return math.isclose(a, b, abs_tol=0.001)


def test_huber_loss():
    '''Test huber loss'''

    input_x = np.array([-2, -1, -0.9, -0.5, 0, 0.2, 0.1, 0.5, 1, 2, 30])
    target_loss = (input_x[np.abs(input_x) <= 1] ** 2 * 0.5).sum() +\
                  (np.abs(input_x[np.abs(input_x) > 1]) - 0.5).sum()
    loss = model.huber_loss(
        tf.constant(input_x, tf.float32)
    )

    loss = sess.run(loss)
    assert equal(loss, target_loss)

def test_iou():
    # 3 x 5 x 2
    grid = np.dstack(np.meshgrid(10 * np.arange(5), 10 * np.arange(3)))
    boxes = np.tile(
        np.expand_dims(np.expand_dims(np.array([10, 10]), 0), 0),
        [3, 5, 1]
    )
    proposals = np.reshape(np.concatenate([grid, boxes], axis=2), (-1, 4))

    proposals = tf.constant(proposals, tf.float32)
    ground_truth = tf.constant(np.array([
        [4, 4, 10, 10],
        [10, 10, 10, 10]
    ]), tf.float32)
    iou_metric = sess.run(model.iou(ground_truth, 2, proposals, 15))
    assert equal(iou_metric[0, 0], 0.2195)
    assert equal(iou_metric[1, 0], 0.1363)
    assert equal(iou_metric[5, 0], 0.1363)
    assert equal(iou_metric[6, 0], 0.0869)

    assert equal(iou_metric[6, 1], 1.0)

    for (boxes, count) in [(proposals, 15), (ground_truth, 2)]:
        iou_metric = sess.run(model.iou(boxes, count, boxes, count))
        assert np.all(np.diag(iou_metric) == 1)


def test_generate_anchors():
    boxes = tf.constant(
        np.array([[10, 10], [50, 50], [15, 45]]),
        tf.float32
    )
    height = tf.placeholder(tf.float32)
    width = tf.placeholder(tf.float32)
    conv_height = tf.placeholder(tf.int32)
    conv_width = tf.placeholder(tf.int32)

    anchors = model.generate_anchors(
        boxes,
        height, width, conv_height, conv_width
    )

    anchors = sess.run(anchors, {
        height: 300,
        width: 200,
        conv_height: 300 // 16,
        conv_width: 200 // 16,
    })

    assert anchors.shape == (300 // 16, 200 // 16, 3, 4)

    # bottom-right position
    assert anchors[300 // 16 - 1, 200 // 16 - 1, 0, 0] == 299.5
    assert anchors[300 // 16 - 1, 200 // 16 - 1, 0, 1] == 199.5

    # top-right position
    assert anchors[0, 200 // 16 - 1, 0, 0] == -0.5
    assert anchors[0, 200 // 16 - 1, 0, 1] == 199.5

    # bottom-left position
    assert anchors[300 // 16 - 1, 0, 0, 0] == 299.5
    assert anchors[300 // 16 - 1, 0, 0, 1] == -0.5

