'''Models'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np
import tensorflow as tf

import densecap.util as util


def iou(x):
    '''Helper function to calculate IoU

    x: 2 x 4 Tensor
    '''
    return util.tf_iou(x[0], x[1])


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
                    scope=name
                )
                self.layers[name] = value
            value = tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.layers['pool{}'.format(idx+1)] = value


class RegionProposalNetwork(object):
    # TODO: make H and W dynamic
    def __init__(self, conv_layer):
        # conv_layer should be (N, H', W', C) tensor
        self.input = conv_layer
        self.filters_num = 256
        self.ksize = [3, 3]
        self.H, self.W = tf.placeholder(tf.int32), tf.placeholder(tf.int32)
        # VGG architecture - conv5 layer has 4 maxpools, hence 16 = 2 ** 4
        self.Hp, self.Wp = self.H // 16, self.W // 16
        self.gt_box_count = tf.placeholder(tf.int32)
        self.boxes = tf.Variable([
            (45, 90), (90, 45), (64, 64),
            (90, 180), (180, 90), (128, 128),
            (181, 362), (362, 181), (256, 256),
            (362, 724), (724, 362), (512, 512),
        ], dtype=tf.float32)
        self.k, _ = self.boxes.get_shape().as_list()
        # TODO: parametrize
        self.learning_rate = 0.001
        self.batch_size = 256
        self.l1_coef = 10.0

        self.layers = {}
        self._build()
        self._create_loss()
        self._create_train()

    def _create_loss(self):
        scores = tf.concat(0, [self.pos_scores, self.neg_scores])
        # XXX: check log-loss correctness (is minus required)
        score_loss = tf.reduce_sum(
            scores * tf.log(scores) + (1 - scores) * tf.log(1 - scores)
        )

        box_reg_loss = self._box_params_loss(
            self.gt,
            tf.reshape(self.anchor_centers, [-1, 4]),
            self.pos_sample_mask, self.offsets
        )
        self.loss = tf.add(score_loss, tf.mul(self.l1_coef, box_reg_loss, name='box_loss_lambda'), name='total_loss')

    def _create_train(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build(self):
        self._create_conv6()

        Hp, Wp = self.Hp, self.Wp
        self._generate_anchor_centers(self.H, self.W, Hp, Wp)

        self.offsets = tf.reshape(self.layers['offsets'], [Hp, Wp, self.k, 4], name='1')
        proposals = self._generate_proposals(self.offsets, Hp, Wp)
        # XXX: replace sigmoid with logits once scores are 2 numbers instead of 1
        scores = tf.reshape(tf.sigmoid(self.layers['scores']), [Hp * Wp * self.k, 1], name='2')

        # TODO: implement cross-boundary filetering
        proposals, scores = self._cross_border_filter(proposals, scores)

        # XXX: consider not specifying input size.
        self.gt = tf.placeholder(tf.float32, [None, 4])  # M ground truth boxes
        pos_batch, neg_batch = self._generate_batches(proposals, self.gt, scores)

        self.pos_boxes, self.pos_scores = pos_batch
        self.neg_boxes, self.neg_scores = neg_batch

    def _generate_batches(self, proposals, gt, scores):
        iou_metric = self._iou(gt, proposals)

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
        negative_scores = tf.boolean_mask(scores, negative_mask)

        B = self.batch_size // 2
        # pad positive samples with negative if there are not enough
        # TODO: shuffle? random sampling?
        postitve_boxes = tf.slice(
            tf.concat(0, [positive_boxes, negative_boxes]), [0, 0], [B, -1],
            name='pos_box_slice'
        )
        postitve_scores = tf.slice(
            tf.concat(0, [positive_scores, negative_scores]), [0, 0], [B, -1],
            name='pos_score_slice'
        )

        negative_boxes = tf.slice(
            negative_boxes, [0, 0], [B, -1], name='neg_box_slice'
        )
        negative_scores = tf.slice(
            negative_scores, [0, 0], [B, -1], name='neg_score_slice'
        )

        return (
            (positive_boxes, positive_scores),
            (negative_boxes, negative_scores)
        )

    def _iou(self, gt, proposals):
        N = self.Hp * self.Wp * self.k
        M = self.gt_box_count

        proposals = tf.expand_dims(proposals, axis=1)
        proposals = tf.tile(proposals, [1, M, 1])

        gt = tf.expand_dims(gt, axis=0)
        gt = tf.tile(gt, [N, 1, 1])

        x11, y11, w1, h1 = tf.unstack(proposals, axis=2)
        x12, y12 = x11 + w1, y11 + h1
        x21, y21, w2, h2 = tf.unstack(gt, axis=2)
        x22, y22 = x21 + w2, y21 + h2

        intersection = (
            tf.maximum(0.0, tf.minimum(x12, x22) - tf.maximum(x11, x21)) *
            tf.maximum(0.0, tf.minimum(y12, y22) - tf.maximum(y11, y21))
        )

        iou_metric = intersection / (
            w1 * h1 + w2 * h2 - intersection
        )
        return iou_metric


    def _generate_proposals(self, offsets, Hp, Wp):
        # each shape is Hp x Wp x k
        tx, ty, tw, th = tf.unstack(offsets, axis=3)
        # each shape is Hp x Wp x k
        xa, ya, wa, ha = tf.unstack(self.anchor_centers, axis=3)

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
        N = self.Wp * self.Hp * self.k
        M = self.gt_box_count
        print('M = ', M, 'N = ', N)
        # ground_truth shape is M x 4, where M is count and 4 are x,y,w,h
        gt = tf.expand_dims(ground_truth, axis=0)
        gt = tf.tile(gt, [N, 1, 1])
        print('gt.shape', gt.get_shape())
        # anchor_centers shape is N x 4 where N is count and 4 are xa,ya,wa,ha
        anchor_centers = tf.expand_dims(anchor_centers, axis=1)
        anchor_centers = tf.tile(anchor_centers, [1, M, 1])
        print('anchor_centers.shape', anchor_centers.get_shape())
        # pos_sample_mask shape is N x M, True are for positive proposals
        mask = tf.expand_dims(tf.cast(pos_sample_mask, tf.float32), axis=2)
        print('mask.shape', mask.get_shape())

        xa, ya, wa, ha = tf.unstack(anchor_centers, axis=2)
        print('anchor_centers xa.shape', xa.get_shape())
        x, y, w, h = tf.unstack(gt, axis=2)
        print('gt x.shape', x.get_shape())

        # idea is to calculate N x M tx, ty, tw, th for ground truth boxes
        # for every proposal. Then we caclulate loss, multiply it with mask
        # to filter out non-positive samples and sum to one

        # each shape is N x M
        tx = tf.div(tf.sub(x, xa, name='x_-_xa'), wa, name='x_minus_xa_div_wa')
        ty = tf.div(tf.sub(y, ya, name='y_minus_ya'), ha, name='y_minus_ya_div_ha')
        tw = tf.log(tf.div(w, wa, name='w_div_wa'))
        th = tf.log(tf.div(h, ha, name='h_div_ha'))

        gt_params = tf.stack([tx, ty, tw, th], axis=2)
        print('gt_params.shape', gt_params.get_shape())

        offsets = tf.expand_dims(tf.reshape(offsets, [N, 4], name='7'), axis=1)
        offsets = tf.tile(offsets, [1, M, 1])
        print('offsets.shape', offsets.get_shape())

        # TODO: replace l2 loss with huber loss (L1 smooth)
        return tf.nn.l2_loss((offsets - gt_params) * mask)

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
            1 * self.k,  # XXX: check if 1 is enough (switch to 2?)
            [1] * 2,
            scope='scores'
        )  # H' x W' x k
        self.layers['scores'] = scores[0]


    def _generate_anchor_centers(self, H, W, Hp, Wp):
        # those are strides in terms of original image
        # i.e. what x and y base image strides corresponds to 1,1 conv layer stride
        # XXX: check for Hp Wp everywhere!!
        sh, sw = 16, 16
        H, W = tf.cast(H, tf.float32), tf.cast(W, tf.float32)

        # TODO: probably replace `numpy` ops with tf ones
        grid = tf.stack(tf.meshgrid(
                tf.linspace(-0.5, H - 0.5, Wp),
                tf.linspace(-0.5, W - 0.5, Hp)), axis=2)

        # convert boxes from K x 2 to 1 x 1 x K x 2
        boxes = tf.expand_dims(tf.expand_dims(self.boxes, 0), 0)
        # convert grid from Hp x Wp x 2 to Hp x Wp x 1 x 2
        grid = tf.expand_dims(grid, 2)

        # combine them into single Hp x Wp x K x 4 tensor
        self.anchor_centers = tf.concat(
            3,
            [tf.tile(grid, [1, 1, self.k, 1]), tf.tile(boxes, [Hp, Wp, 1, 1])]
        )
