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
    flatten_dim = 7 * 7 * 512
    fully_c = [
        (4096),
        (4096),
        (1000)
    ]

    mean_pixel = [103.939, 116.779, 123.68]

    def __init__(self, height, width):
        self.layers = {}
        self.input = tf.placeholder(tf.float32, [None, height, width, 3])

        value = self.input

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
            value = tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.layers['pool{}'.format(idx+1)] = value

        value = tf.reshape(value, [-1, self.flatten_dim])
        for idx, num_outputs in enumerate(self.fully_c, start=len(self.pools)):
            name = 'fc{}'.format(idx+1)
            value = tf.contrib.layers.fully_connected(
                value,
                num_outputs,
                scope=name
            )
            self.layers[name] = value
        self.predicted = tf.nn.softmax(value)


class RegionProposalNetwork(object):
    # TODO: make H and W dynamic
    def __init__(self, conv_layer, H, W):
        # conv_layer should be (N, H', W', C) tensor
        self.input = conv_layer
        self.filters_num = 256
        self.ksize = [3, 3]
        self.H, self.W = H, W
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

        self.layers = {}
        self._build()
        self._create_loss()
        self._create_train()

    def _create_loss(self):
        logit_loss = tf.reduce_sum(
            self.pos_scores * tf.log(self.pos_scores) +
            self.pos_scores * tf.log(self.pos_scores)
        ) / self.batch_size
        # TODO: implement
        box_reg_loss = 0.0
        self.loss = logit_loss + box_reg_loss

    def _create_train(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build(self):
        self._create_conv6()

        _, Hp, Wp, _ = self.layers['conv6_1'].get_shape().as_list()
        self._generate_anchor_centers(self.H, self.W, Hp, Wp)

        offsets = tf.reshape(self.layers['offsets'], [Hp, Wp, self.k, 4])
        proposals = self._generate_proposals(offsets, Hp, Wp)
        scores = tf.reshape(self.layers['scores'], [Hp * Wp * self.k, 1])

        # TODO: implement cross-boundary filetering
        proposals, scores = self._cross_border_filter(proposals, scores)

        # XXX: consider specifying input size.
        self.gt = tf.placeholder(tf.float32, [self.batch_size // 4, 4])  # M ground truth boxes
        pos_batch, neg_batch = self._generate_batches(proposals, self.gt, scores)

        self.pos_scores, self.pos_boxes = pos_batch
        self.neg_scores, self.neg_boxes = neg_batch

    def _generate_batches(self, proposals, gt, scores):
        N, d1 = proposals.get_shape().as_list()
        M, d2 = gt.get_shape().as_list()
        assert d1 == d2, 'Wrong proposal/ground truth boxes shape.'
        d = d1

        orig_proposals = proposals
        proposals = tf.expand_dims(proposals, axis=1)
        proposals = tf.tile(proposals, [1, M, 1])

        gt = tf.expand_dims(gt, axis=0)
        gt = tf.tile(gt, [N, 1, 1])

        proposals = tf.reshape(proposals, (N*M, d))
        gt = tf.reshape(gt, (N*M, d))

        # shape is N*M x 1
        # TODO: speed up (!)
        iou_metric = tf.map_fn(iou, tf.stack([proposals, gt], axis=1))
        iou_metric = tf.reshape(iou_metric, [N, M])

        # now let's get rid of non-positive and non-negative samples
        zeros = tf.zeros(iou_metric.get_shape().as_list())

        # here we take either iou value if it greater than threshold
        # or zero. We sum over all options. Sample is considered
        # positive if it has IoU with _any_ ground truth boxes, so
        # we check if sum is greater than threshold
        options_1 = tf.reduce_sum(tf.select(tf.greater(iou_metric, 0.7), iou_metric, zeros), axis=1)
        positive_mask = tf.greater(options_1, 0)

        # here we compare iou metric with another threshold. Sample
        # would be considered negative if _all_ ground truch boxes
        # have iou less than threshold. We mark those with zeros.
        # We need to find those where sum is equal to zero.
        options_2 = tf.reduce_sum(tf.select(tf.less(iou_metric, 0.3), zeros, iou_metric), axis=1)
        negative_mask = tf.equal(options_2, 0)

        positive_boxes = tf.boolean_mask(orig_proposals, positive_mask)
        negative_boxes = tf.boolean_mask(orig_proposals, negative_mask)

        positive_scores = tf.boolean_mask(scores, positive_mask)
        negative_scores = tf.boolean_mask(scores, negative_mask)

        B = self.batch_size // 2
        # pad positive samples with negative if there are not enough
        # TODO: shuffle? random sampling?
        print(positive_boxes.get_shape(), negative_boxes.get_shape())
        postitve_boxes = tf.slice(tf.concat(0, [positive_boxes, negative_boxes]), [0, 0], [B, -1])
        postitve_scores = tf.slice(tf.concat(0, [positive_scores, negative_scores]), [0, 0], [B, -1])

        negative_boxes = tf.slice(negative_boxes, [0, 0], [B, -1])
        negative_scores = tf.slice(negative_scores, [0, 0], [B, -1])

        return (
            (positive_boxes, positive_scores),
            (negative_boxes, negative_scores)
        )


    def _generate_proposals(self, offsets, Hp, Wp):
        # XXX: consider using tf.split instead
        # each shape is Hp x Wp x k
        tx = offsets[:, :, :, 0]  # tx
        ty = offsets[:, :, :, 1]  # ty
        th = offsets[:, :, :, 2]  # th
        tw = offsets[:, :, :, 3]  # tw

        # XXX: consider using tf.split instead
        # each shape is Hp x Wp x k
        xa = self.anchor_centers[:, :, :, 0]
        ya = self.anchor_centers[:, :, :, 1]
        ha = self.anchor_centers[:, :, :, 2]
        wa = self.anchor_centers[:, :, :, 3]

        x = xa + tx * wa
        y = ya + ty * ha
        w = wa * tf.exp(tw)
        h = ha * tf.exp(th)

        # shape is Hp*Wp*k x 4
        proposals = tf.stack([x, y, w, h], axis=3)
        proposals = tf.reshape(proposals, [Hp * Wp * self.k, 4])
        return proposals


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
        sh, sw = H // Hp, W // Wp

        # TODO: probably replace `numpy` ops with tf ones
        grid = tf.constant(
            np.dstack(np.meshgrid(np.arange(-0.5, H - 0.5, sh), np.arange(-0.5, W - 0.5, sw))),
            dtype=tf.float32
        )

        # convert boxes from K x 2 to 1 x 1 x K x 2
        boxes = tf.expand_dims(tf.expand_dims(self.boxes, 0), 0)
        # convert grid from Hp x Wp x 2 to Hp x Wp x 1 x 2
        grid = tf.expand_dims(grid, 2)

        # combine them into single Hp x Wp x K x 4 tensor
        self.anchor_centers = tf.concat(
            3,
            [tf.tile(grid, [1, 1, self.k, 1]), tf.tile(boxes, [Hp, Wp, 1, 1])]
        )
