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


def test_generate_proposals():
    N = 2
    anchors = tf.constant(
        np.array([[10, 10, 100, 100], [30, 40, 50, 150]]),
        tf.float32
    )

    coef = tf.constant(np.ones((N, 4)), tf.float32)
    proposals = sess.run(
        model.generate_proposals(coef, anchors))
    assert proposals.shape == (N, 4)
    assert np.allclose(proposals, np.array([
        [110, 110, 271.82818604, 271.82818604],
        [80, 190, 135.91409302, 407.74224854]
    ]))


def test_generate_batches():
    # 10 x 10 proposal locations, 100 x 100 pixels
    grid = np.dstack(np.meshgrid(10 * np.arange(10), 10 * np.arange(10)))
    boxes = np.tile(
        np.expand_dims(np.expand_dims(np.array([10, 10]), 0), 0),
        [10, 10, 1]
    )
    proposals = np.reshape(np.concatenate([grid, boxes], axis=2), (-1, 4))
    proposals_num = len(proposals)
    scores = np.tile([-0.5, 0], [proposals_num, 1])
    # okay, now we have 100 proposals. let's select some of them as ground truth
    # take 10 of them
    ground_truth = proposals[::11]
    scores[::11] *= -1

    scores = tf.constant(scores, tf.float32)
    proposals_num = tf.constant(proposals_num, tf.int32)
    gt_num = tf.constant(len(ground_truth), tf.int32)
    proposals = tf.constant(proposals, tf.float32)
    ground_truth = tf.constant(ground_truth, tf.float32)

    result = sess.run(
        model.generate_batches(proposals, proposals_num, ground_truth, gt_num, scores, 10))
    (pos_boxes, pos_scores, pos_labels), (neg_boxes, neg_scores, neg_labels) = result

    assert np.all(pos_boxes == np.array([
        [0, 0, 10, 10],
        [10, 10, 10, 10],
        [20, 20, 10, 10],
        [30, 30, 10, 10],
        [40, 40, 10, 10]]
    ))
    assert np.all(pos_scores == np.array([[0.5, 0]] * 5))
    assert np.all(pos_labels == np.array([1] * 5))

    assert np.all(neg_boxes == np.array([
        [10, 0, 10, 10],
        [20, 0, 10, 10],
        [30, 0, 10, 10],
        [40, 0, 10, 10],
        [50, 0, 10, 10]
    ]))
    assert np.all(neg_scores == np.array([[-0.5, 0]] * 5))
    assert np.all(neg_labels == np.array([0] * 5))

    # now let's try to simulate negative padding, i.e. size of batch // 2 exceeds number
    # of positive samples
    result = sess.run(
        model.generate_batches(proposals, proposals_num, ground_truth, gt_num, scores, 24))
    (pos_boxes, pos_scores, pos_labels), (neg_boxes, neg_scores, neg_labels) = result

    assert np.all(pos_boxes == np.array([
        [0, 0, 10, 10],
        [10, 10, 10, 10],
        [20, 20, 10, 10],
        [30, 30, 10, 10],
        [40, 40, 10, 10],
        [50, 50, 10, 10],
        [60, 60, 10, 10],
        [70, 70, 10, 10],
        [80, 80, 10, 10],
        [90, 90, 10, 10],
        [10, 0, 10, 10],
        [20, 0, 10, 10],
    ]))
    assert np.all(pos_scores == np.concatenate([np.array([[0.5, 0]] * 10), np.array([[-0.5, 0]] * 2)]))
    assert np.all(pos_labels == np.concatenate([np.array([1] * 10), np.array([0] * 2)]))


def test_cross_border_filter():
    # 10 x 10 proposal locations, 100 x 100 pixels
    grid = np.dstack(np.meshgrid(10 * np.arange(-2, 12), 10 * np.arange(-2, 12)))
    boxes = np.tile(
        np.expand_dims(np.expand_dims(np.array([10, 10]), 0), 0),
        [14, 14, 1]
    )
    np_proposals = np.reshape(np.concatenate([grid, boxes], axis=2), (-1, 4))
    proposals_num = len(np_proposals)
    scores = np.tile([1, 0], [proposals_num, 1])

    proposals = tf.constant(np_proposals, tf.float32)
    scores = tf.constant(scores, tf.float32)
    height = tf.constant(100, tf.float32)
    width = tf.constant(100, tf.float32)
    fproposals, scores = sess.run(model.cross_border_filter(proposals, scores, height, width))

    assert fproposals.shape == (100, 4)
    assert scores.shape == (100, 2)

    mask = (np_proposals[:, 0] >= 0) & (np_proposals[:, 1] >= 0) & \
           (np_proposals[:, 0] + np_proposals[:, 2] <= 100) & \
           (np_proposals[:, 1] + np_proposals[:, 3] <= 100)
    assert np.all(np_proposals[mask] == fproposals)
