'''Models'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import tensorflow as tf

import densecap.util as util


def huber_loss(x, delta=1):
    coef = 0.5
    l2_mask = tf.less_equal(tf.abs(x), delta)
    l1_mask = tf.greater(tf.abs(x), delta)

    term_1 = tf.reduce_sum(coef * tf.square(tf.boolean_mask(x, l2_mask)))
    term_2 = tf.reduce_sum(delta * (tf.abs(tf.boolean_mask(x, l1_mask)) - coef * delta))

    return term_1 + term_2

def iou(ground_truth, ground_truth_count, proposals, proposals_count):
    '''Caclulate IoU for given ground truth and proposal boxes

    ground_truth: M x 4 ground truth boxes tensor
    proposals: N x 4 ground truth boxes tensor

    returns:
    N x M IoU tensor
    '''
    proposals = tf.expand_dims(proposals, axis=1)
    proposals = tf.tile(proposals, [1, ground_truth_count, 1])

    ground_truth = tf.expand_dims(ground_truth, axis=0)
    ground_truth = tf.tile(ground_truth, [proposals_count, 1, 1])

    x11, y11, width1, height1 = tf.unstack(proposals, axis=2)
    x21, y21, width2, height2 = tf.unstack(ground_truth, axis=2)
    x12, y12 = x11 + width1, y11 + height1
    x22, y22 = x21 + width2, y21 + height2

    intersection = (
        tf.maximum(0.0, tf.minimum(x12, x22) - tf.maximum(x11, x21)) *
        tf.maximum(0.0, tf.minimum(y12, y22) - tf.maximum(y11, y21))
    )

    iou_metric = intersection / (
        width1 * height1 + width2 * height2 - intersection
    )
    return iou_metric


def generate_anchors(boxes, height, width, conv_height, conv_width):
    '''Generate anchors for given geometry

    boxes: K x 2 tensor for anchor geometries, K different sizes
    height: source image height
    width: source image width
    conv_height: convolution layer height
    conv_width: convolution layer width

    returns:
    conv_height x conv_width x K x 4 tensor with boxes for all
    positions. Last dimension 4 numbers are (y, x, h, w)
    '''
    k, _ = boxes.get_shape().as_list()

    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    grid = tf.transpose(tf.stack(tf.meshgrid(
        tf.linspace(-0.5, height - 0.5, conv_height),
        tf.linspace(-0.5, width - 0.5, conv_width)), axis=2), [1, 0, 2])

    # convert boxes from K x 2 to 1 x 1 x K x 2
    boxes = tf.expand_dims(tf.expand_dims(boxes, 0), 0)
    # convert grid from H' x W' x 2 to H' x W' x 1 x 2
    grid = tf.expand_dims(grid, 2)

    # combine them into single H' x W' x K x 4 tensor
    return tf.concat(
        3,
        [tf.tile(grid, [1, 1, k, 1]),
         tf.tile(boxes, [conv_height, conv_width, 1, 1])]
    )


class VGG16(object):
    pools = [
        (2, 64),
        (2, 128),
        (3, 256),
        (3, 512),
        (3, 512),
    ]
    mean_pixel = [103.939, 116.779, 123.68]

    def __init__(self, input_images):
        self.layers = {}
        self.input = input_images

        value = self.input
        value -= self.mean_pixel

        for idx, (layers, filters) in enumerate(self.pools):
            for layer in range(layers):
                name = 'conv{}_{}'.format(idx+1, layer+1)
                value = tf.contrib.layers.conv2d(
                    value,
                    filters,
                    [3, 3],
                    trainable=False,
                    scope=name
                )
                self.layers[name] = value
            value = tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.layers['pool{}'.format(idx+1)] = value


class RegionProposalNetwork(object):
    def __init__(self, vgg_conv_layer):
        self._create_variables()

        self.input = vgg_conv_layer
        self.filters_num = 256
        self.ksize = [3, 3]
        self.learning_rate = 0.001
        self.batch_size = 256
        self.l1_coef = 10.0
        self.k, _ = self.boxes.get_shape().as_list()

        self.layers = {}
        self._build()
        self._create_loss()
        self._create_train()

    def _create_variables(self):
        self.image_height, self.image_width = tf.placeholder(tf.int32), tf.placeholder(tf.int32)
        self.ground_truth_num = tf.placeholder(tf.int32)
        self.ground_truth = tf.placeholder(tf.float32, [None, 4])

        self.boxes = tf.Variable([
            (45, 90), (90, 45), (64, 64),
            (90, 180), (180, 90), (128, 128),
            (181, 362), (362, 181), (256, 256),
            (362, 724), (724, 362), (512, 512),
        ], dtype=tf.float32)
        self.global_step = tf.Variable(0, name='global_step')

    def _create_loss(self):
        predicted_scores = tf.concat(0, [self.pos_scores, self.neg_scores])
        true_labels = tf.to_int32(tf.concat(0, [self.true_pos_scores, self.true_neg_scores]))

        score_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            predicted_scores, true_labels
        ))

        box_reg_loss = self._box_params_loss(
            self.ground_truth,
            tf.reshape(self.anchors, [-1, 4]),
            self.pos_sample_mask, self.offsets
        )
        self.loss = tf.add(score_loss, tf.mul(self.l1_coef, box_reg_loss, name='box_loss_lambda'), name='total_loss')

        tf.summary.scalar('score_loss', score_loss)
        tf.summary.scalar('box_regression_loss', box_reg_loss)
        tf.summary.scalar('loss', self.loss)

    def _create_train(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build(self):
        self._create_conv6()

        # VGG architecture - conv5 layer has 4 maxpools, hence 16 = 2 ** 4
        conv_height, conv_width = self.image_height // 16, self.image_width // 16
        proposals_num = conv_height * conv_width * self.k

        self.anchors = generate_anchors(self.boxes,
            self.image_height, self.image_width, conv_height, conv_width)

        self.offsets = tf.reshape(self.layers['offsets'], [proposals_num, 4])
        self.scores = tf.reshape(self.layers['scores'], [proposals_num, 2])

        proposals = self._generate_proposals(self.offsets, conv_height, conv_width)

        # TODO: implement cross-boundary filetering
        proposals, scores = self._cross_border_filter(proposals, self.scores)
        self.proposals = proposals

        pos_batch, neg_batch = self._generate_batches(
            proposals, proposals_num, self.ground_truth, self.ground_truth_num, scores)

        self.pos_boxes, self.pos_scores, self.true_pos_scores = pos_batch
        self.neg_boxes, self.neg_scores, self.true_neg_scores = neg_batch

    def _generate_batches(self, proposals, proposals_num, gt, gt_num, scores):
        iou_metric = iou(gt, gt_num,
                               proposals, proposals_num)

        # now let's get rid of non-positive and non-negative samples
        # here we take either iou value if it greater than threshold
        # or zero. We sum over all options. Sample is considered
        # positive if it has IoU with _any_ ground truth boxes
        # we will need that for calculating ground truch box params and loss
        mask = tf.greater(iou_metric, 0.7)
        self.pos_sample_mask = mask
        positive_mask = tf.reduce_any(mask, axis=1)

        # here we compare iou metric with another threshold. Sample
        # would be considered negative if _all_ ground truch boxes
        # have iou less than threshold
        neg_mask = tf.less(iou_metric, 0.3)
        negative_mask = tf.reduce_all(neg_mask, axis=1)

        positive_boxes = tf.boolean_mask(proposals, positive_mask)
        negative_boxes = tf.boolean_mask(proposals, negative_mask)

        positive_scores = tf.boolean_mask(scores, positive_mask)
        true_positive_scores = tf.reduce_mean(tf.ones_like(positive_scores), axis=1)
        negative_scores = tf.boolean_mask(scores, negative_mask)
        true_negative_scores = tf.reduce_mean(tf.zeros_like(negative_scores), axis=1)

        B = self.batch_size // 2
        # pad positive samples with negative if there are not enough
        # TODO: shuffle? random sampling?
        positive_boxes = tf.slice(
            tf.concat(0, [positive_boxes, negative_boxes]), [0, 0], [B, -1],
            name='pos_box_slice'
        )
        positive_scores = tf.slice(
            tf.concat(0, [positive_scores, negative_scores]), [0, 0], [B, -1],
            name='pos_score_slice'
        )
        true_scores = tf.slice(
            tf.concat(0, [true_positive_scores, true_negative_scores]), [0], [B],
            name='true_score_slice'
        )

        negative_boxes = tf.slice(
            negative_boxes, [0, 0], [B, -1], name='neg_box_slice'
        )
        negative_scores = tf.slice(
            negative_scores, [0, 0], [B, -1], name='neg_score_slice'
        )

        return (
            (positive_boxes, positive_scores, true_scores),
            (negative_boxes, negative_scores, tf.reduce_sum(tf.zeros_like(negative_scores), axis=1))
        )

    def _generate_proposals(self, offsets, Hp, Wp):
        # each shape is Hp x Wp x k
        tx, ty, tw, th = tf.unstack(offsets, axis=3)
        # each shape is Hp x Wp x k
        xa, ya, wa, ha = tf.unstack(self.anchors, axis=3)

        x = tf.add(xa, tf.mul(tx, wa, name='tx_times_wa'), name='xa_plus_tx_times_wa')
        y = tf.add(ya, tf.mul(ty, ha, name='ty_times_ha'), name='ya_plus_ty_times_ha')
        w = tf.mul(wa, tf.exp(tw), name='wa_times_exp_tw_')
        h = tf.mul(ha, tf.exp(th), name='ha_times_exp_th_')

        # shape is Hp*Wp*k x 4
        proposals = tf.stack([x, y, w, h], axis=3)
        # XXX: replace explicit shape with `-1`
        proposals = tf.reshape(proposals, [Hp * Wp * self.k, 4], name='6')
        return proposals

    def _box_params_loss(self, ground_truth, anchor_centers, pos_sample_mask, offsets):
        N = self.proposals_num
        M = self.ground_truth_num
        # ground_truth shape is M x 4, where M is count and 4 are x,y,w,h
        gt = tf.expand_dims(ground_truth, axis=0)
        gt = tf.tile(gt, [N, 1, 1])
        # anchor_centers shape is N x 4 where N is count and 4 are xa,ya,wa,ha
        anchor_centers = tf.expand_dims(anchor_centers, axis=1)
        anchor_centers = tf.tile(anchor_centers, [1, M, 1])
        # pos_sample_mask shape is N x M, True are for positive proposals
        mask = tf.expand_dims(tf.cast(pos_sample_mask, tf.float32), axis=2)

        xa, ya, wa, ha = tf.unstack(anchor_centers, axis=2)
        x, y, w, h = tf.unstack(gt, axis=2)

        # idea is to calculate N x M tx, ty, tw, th for ground truth boxes
        # for every proposal. Then we caclulate loss, multiply it with mask
        # to filter out non-positive samples and sum to one

        # each shape is N x M
        tx = tf.div(tf.sub(x, xa, name='x_-_xa'), wa, name='x_minus_xa_div_wa')
        ty = tf.div(tf.sub(y, ya, name='y_minus_ya'), ha, name='y_minus_ya_div_ha')
        tw = tf.log(tf.div(w, wa, name='w_div_wa'))
        th = tf.log(tf.div(h, ha, name='h_div_ha'))

        gt_params = tf.stack([tx, ty, tw, th], axis=2)

        offsets = tf.expand_dims(tf.reshape(offsets, [N, 4], name='7'), axis=1)
        offsets = tf.tile(offsets, [1, M, 1])

        return huber_loss((offsets - gt_params) * mask)

    def _cross_border_filter(self, proposals, scores):
        return proposals, scores

    def _create_conv6(self):
        # throw away first dimention - don't allow multiple images,
        # batches are generated internally from one image
        # slice all inputs to take first item

        conv = tf.contrib.layers.conv2d(
            self.input,
            self.filters_num,
            self.ksize,
            scope='conv6_1'
        )
        self.layers['conv6_1'] = conv

        offsets = tf.contrib.layers.conv2d(
            conv,
            4 * self.k,
            [1] * 2,
            scope='offsets'
        )  # H' x W' x 4k
        self.layers['offsets'] = offsets[0]

        scores = tf.contrib.layers.conv2d(
            conv,
            2 * self.k,
            [1] * 2,
            scope='scores'
        )  # H' x W' x 2k
        self.layers['scores'] = scores[0]
