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


def test_split_proposals():
    # 10 x 10 proposal locations, 100 x 100 pixels
    grid = np.dstack(np.meshgrid(10 * np.arange(10), 10 * np.arange(10)))
    boxes = np.tile(
        np.expand_dims(np.expand_dims(np.array([10, 10]), 0), 0),
        [10, 10, 1]
    )
    np_proposals = np.reshape(np.concatenate([grid, boxes], axis=2), (-1, 4))
    np_proposals_num = len(np_proposals)
    np_scores = np.tile([-0.5, 0], [np_proposals_num, 1])
    # okay, now we have 100 proposals. let's select some of them as ground truth
    # take 10 of them
    np_ground_truth = np_proposals[::11]
    np_ground_truth_num = len(np_ground_truth)
    np_scores[::11] *= -1

    scores = tf.constant(np_scores, tf.float32)
    proposals_num = tf.constant(np_proposals_num, tf.int32)
    gt_num = tf.constant(len(np_ground_truth), tf.int32)
    proposals = tf.constant(np_proposals, tf.float32)
    ground_truth = tf.constant(np_ground_truth, tf.float32)

    iou = model.iou(ground_truth, gt_num, proposals, proposals_num)
    mask = tf.cast(tf.ones([proposals_num]), tf.bool)
    result = sess.run(
        model.split_proposals(proposals, proposals_num, ground_truth, gt_num, iou, scores, mask))
    (pos_boxes, pos_scores, pos_labels), (neg_boxes, neg_scores, neg_labels) = result

    assert np.all(pos_boxes == np_ground_truth)
    assert np.all(pos_scores == np.array([[0.5, 0]] * np_ground_truth_num))
    assert np.all(pos_labels == np.array([1] * np_ground_truth_num))

    assert len(neg_boxes) == (np_proposals_num - np_ground_truth_num)
    assert np.all(neg_scores == np.array([[-0.5, 0]] * (np_proposals_num - np_ground_truth_num)))
    assert np.all(neg_labels == np.array([0] * (np_proposals_num - np_ground_truth_num)))


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
    mask = model.cross_border_filter(proposals, height, width)
    fproposals, scores = tf.boolean_mask(proposals, mask), tf.boolean_mask(scores, mask)
    fproposals, scores = sess.run([fproposals, scores])

    assert mask.get_shape().as_list() == [196]
    assert fproposals.shape == (100, 4)
    assert scores.shape == (100, 2)

    mask = (np_proposals[:, 0] >= 0) & (np_proposals[:, 1] >= 0) & \
           (np_proposals[:, 0] + np_proposals[:, 2] <= 100) & \
           (np_proposals[:, 1] + np_proposals[:, 3] <= 100)
    assert np.all(np_proposals[mask] == fproposals)


def test_generate_batches():
    proposal_count = 10
    batch_size = 5

    pos_bbox_source = np.random.uniform(0, 1, size=(proposal_count, 4))
    pos_score_source = np.random.uniform(0, 1, size=(proposal_count, 2))
    pos_label_source = np.ones((proposal_count, 1))

    neg_bbox_source = np.random.uniform(0, -1, size=(2 * proposal_count, 4))
    neg_score_source = np.random.uniform(0, -1, size=(2 * proposal_count, 2))
    neg_label_source = np.zeros((2 * proposal_count, 1))

    result = model.generate_batches(
        (pos_bbox_source, pos_score_source, pos_label_source),
        (neg_bbox_source, neg_score_source, neg_label_source),
        batch_size * 2
    )
    (pos_bbox, pos_score, pos_label), (neg_bbox, neg_score, neg_label) = result

    assert pos_bbox.shape == (batch_size, 4)
    assert pos_score.shape == (batch_size, 2)
    assert pos_label.shape == (batch_size, 1)

    assert neg_bbox.shape == (batch_size, 4)
    assert neg_score.shape == (batch_size, 2)
    assert neg_label.shape == (batch_size, 1)

    assert np.all(pos_bbox >= 0.0)
    assert np.all(pos_score >= 0.0)
    assert np.all(pos_label == 1.0)

    assert np.all(neg_bbox < 0.0)
    assert np.all(neg_score < 0.0)
    assert np.all(neg_label == 0.0)

    batch_size = 12
    result = model.generate_batches(
        (pos_bbox_source, pos_score_source, pos_label_source),
        (neg_bbox_source, neg_score_source, neg_label_source),
        batch_size * 2
    )
    (pos_bbox, pos_score, pos_label), (neg_bbox, neg_score, neg_label) = result

    assert pos_bbox.shape == (batch_size, 4)
    assert neg_bbox.shape == (batch_size, 4)

    print(pos_bbox)
    assert np.any(pos_bbox > 0)
    assert np.any(pos_bbox < 0)


def test_centerize_ground_truth():
    ground_truth = tf.constant(np.array([
        [4, 4, 10, 10],
        [10, 10, 10, 10]
    ]), tf.float32)

    centered = sess.run(model.centerize_ground_truth(ground_truth))
    assert np.all(centered == np.array([
        [9, 9, 10, 10],
        [15, 15, 10, 10],
    ]))


def test_recall():
    grid = np.dstack(np.meshgrid(10 * np.arange(10), 10 * np.arange(10)))
    boxes = np.tile(
        np.expand_dims(np.expand_dims(np.array([10, 10]), 0), 0),
        [10, 10, 1]
    )
    np_proposals = np.reshape(np.concatenate([grid, boxes], axis=2), (-1, 4))
    np_proposals_num = len(np_proposals)
    # okay, now we have 100 proposals. let's select some of them as ground truth
    # take 10 of them
    np_ground_truth = np_proposals[::11]
    np_ground_truth_num = len(np_ground_truth)


    proposals = tf.placeholder(tf.float32, [None, 4])
    ground_truth = tf.placeholder(tf.float32, [None, 4])
    proposals_num = tf.placeholder(tf.int32)
    ground_truth_num = tf.placeholder(tf.int32)

    recall = model.recall(proposals, proposals_num, ground_truth, ground_truth_num, 0.7)
    assert equal(sess.run(recall, {
        proposals: np_proposals,
        proposals_num: np_proposals_num,
        ground_truth: np_proposals,
        ground_truth_num: np_proposals_num,
    }), 1.0)
    assert equal(sess.run(recall, {
        proposals: np_ground_truth,
        proposals_num: np_ground_truth_num,
        ground_truth: np_ground_truth,
        ground_truth_num: np_ground_truth_num,
    }), 1.0)
    # here we have 100 proposals and only 10 ground_truth
    # we capture all relevant boxes hence recall is 1
    assert equal(sess.run(recall, {
        proposals: np_proposals,
        proposals_num: np_proposals_num,
        ground_truth: np_ground_truth,
        ground_truth_num: np_ground_truth_num,
    }), 1.0)
    # here we have 10 proposals while 100 total samples
    # we capture only 1/10th of all
    assert equal(sess.run(recall, {
        proposals: np_ground_truth,
        proposals_num: np_ground_truth_num,
        ground_truth: np_proposals,
        ground_truth_num: np_proposals_num,
    }), 0.1)
