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

    yc11, xc11, height1, width1 = tf.unstack(proposals, axis=2)
    yc21, xc21, height2, width2 = tf.unstack(ground_truth, axis=2)

    x11, y11 = xc11 - width1 // 2, yc11 - height1 // 2
    x21, y21 = xc21 - width2 // 2, yc21 - height2 // 2
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


def generate_proposals(coefficients, anchors):
    '''Generate proposals from static anchors and normalizing coefficients

    coefficients: N x 4 tensor: N x (ty, tx, th, tw)
    anchors: N x 4 tensor with boxes N x (y, x, h, w)

    anchors contains x,y of box _center_ while returned tensor x,y coordinates
    are top-left corner.

    returns:
    N x 4 tensor with bounding box proposals
    '''

    y_coef, x_coef, h_coef, w_coef = tf.unstack(coefficients, axis=1)
    y_anchor, x_anchor, h_anchor, w_anchor = tf.unstack(anchors, axis=1)

    w = w_anchor * tf.exp(w_coef)
    h = h_anchor * tf.exp(h_coef)
    x = x_anchor + x_coef * w_anchor
    y = y_anchor + y_coef * h_anchor

    proposals = tf.stack([y, x, h, w], axis=1)
    return proposals


def generate_batches(proposals, proposals_num, gt, gt_num, iou, scores, cross_boundary_mask, batch_size):
    '''Generate batches from proposals and ground truth boxes

    Idea is to drastically reduce number of proposals to evaluate. So, we find those
    proposals that have IoU > 0.7 with _any_ ground truth and mark them as positive samples.
    Proposals with IoU < 0.3 with _all_ ground truth boxes are considered negative. All
    other proposals are discarded.

    We generate batch with at most half of examples being positive. We also pad them with negative
    have we not enough positive proposals.

    proposals: N x 4 tensor
    proposal_num: N
    gt: M x 4 tensor
    gt_num: M
    iou: N x M tensor of IoU between every proposal and ground truth
    scores: N x 2 tensor with scores object/not-object
    cross_boundary_mask: N x 1 Tensor masking out-of-image proposals
    batch_size: Size of a batch to generate
    '''
    # now let's get rid of non-positive and non-negative samples
    # Sample is considered positive if it has IoU > 0.7 with _any_ ground truth box
    # XXX: maximal IoU ground truth proposal should be treated as positive
    positive_mask = tf.reduce_any(tf.greater(iou, 0.7), axis=1) & cross_boundary_mask

    # Sample would be considered negative if _all_ ground truch box
    # have iou less than 0.3
    negative_mask = tf.reduce_all(tf.less(iou, 0.3), axis=1) & cross_boundary_mask

    # Select only positive boxes and their corresponding predicted scores
    positive_boxes = tf.boolean_mask(proposals, positive_mask)
    positive_scores = tf.boolean_mask(scores, positive_mask)

    # Same for negative
    negative_boxes = tf.boolean_mask(proposals, negative_mask)
    negative_scores = tf.boolean_mask(scores, negative_mask)

    true_positive_scores = tf.reduce_mean(tf.ones_like(positive_scores), axis=1)
    true_negative_scores = tf.reduce_mean(tf.zeros_like(negative_scores), axis=1)

    B = batch_size // 2
    # pad positive samples with negative if there are not enough
    # TODO: shuffle? random sampling?
    # XXX: look at random_crop, random_shuffle
    positive_boxes = tf.slice(
        tf.concat(0, [positive_boxes, negative_boxes]), [0, 0], [B, -1],
        name='pos_box_slice'
    )
    positive_predicted_scores = tf.slice(
        tf.concat(0, [positive_scores, negative_scores]), [0, 0], [B, -1],
        name='pos_score_slice'
    )
    positive_labels = tf.slice(
        tf.concat(0, [true_positive_scores, true_negative_scores]), [0], [B],
        name='true_score_slice'
    )

    negative_boxes = tf.slice(
        negative_boxes, [0, 0], [B, -1], name='neg_box_slice'
    )
    negative_predicted_scores = tf.slice(
        negative_scores, [0, 0], [B, -1], name='neg_score_slice'
    )
    negative_labels = tf.slice(
        true_negative_scores, [0], [B]
    )

    return (
        (positive_boxes, positive_predicted_scores, positive_labels),
        (negative_boxes, negative_predicted_scores, negative_labels)
    )


def cross_border_filter(proposals, image_height, image_width):
    '''Calculate mask to filter out proposals that are partally out of image'''
    im_height = tf.cast(image_height, tf.float32)
    im_width = tf.cast(image_width, tf.float32)
    mask = (proposals[:, 0] >= 0) & (proposals[:, 1] >= 0) & \
           (proposals[:, 0] + proposals[:, 2] <= im_height) & \
           (proposals[:, 1] + proposals[:, 3] <= im_width)

    # TODO: check if it's adequate
    mask.set_shape([None])
    return mask


def centerize_ground_truth(ground_truth_pre):
    y, x, height, width = tf.unstack(ground_truth_pre, axis=1)
    yc, xc = y + height // 2, x + width // 2
    return tf.stack([yc, xc, height, width], axis=1)


# XXX: consider replacing with `tf.contrib.metrics.streaming_recall_at_thresholds`
def recall(proposals, proposals_num, ground_truth, ground_truth_num, iou_threshold):
    '''Calculate recall with given IoU threshold

    proposals: N x 4 tensor (N x (y, x, h, w))
    proposals_num: proposals count
    ground_truth: M x 4 tensor (M x (y, x, h, w))
    ground_truth_num: ground truth boxes count
    iou_threshold: float in range [0; 1]

    returns recall
    '''
    # shape is N x M
    iou_metric = iou(ground_truth, ground_truth_num, proposals, proposals_num)
    # shape is M x 1
    true_positives = tf.reduce_sum(
        tf.cast(tf.reduce_any(iou_metric >= iou_threshold, axis=0), tf.float32))
    return true_positives / tf.cast(ground_truth_num, tf.float32)


def precision(proposals, proposals_num, ground_truth, ground_truth_num, iou_threshold):
    '''Calculate precision with given IoU threshold

    proposals: N x 4 tensor (N x (y, x, h, w))
    proposals_num: proposals count
    ground_truth: M x 4 tensor (M x (y, x, h, w))
    ground_truth_num: ground truth boxes count
    iou_threshold: float in range [0; 1]

    returns precision
    '''
    # shape is N x M
    iou_metric = iou(ground_truth, ground_truth_num, proposals, proposals_num)
    # shape is M x 1
    true_positives = tf.reduce_sum(
        tf.cast(tf.reduce_any(iou_metric >= iou_threshold, axis=1), tf.float32))
    return true_positives / tf.cast(proposals_num, tf.float32)


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
        self.filters_num = 512
        self.ksize = [3, 3]
        self.learning_rate = 1e-6
        self.batch_size = 256
        self.l1_coef = 10.0
        self.k, _ = self.boxes.get_shape().as_list()
        self.l2_loss = 0.1

        self.layers = {}
        self._build()
        self._create_loss()
        self._create_train()
        self._create_summaries()

    def _create_summaries(self):
        tf.summary.scalar('loss', self.loss)

        tf.contrib.layers.summaries.summarize_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tf.contrib.layers.summaries.summarize_activations()

        tf.contrib.layers.summaries.summarize_tensor(self.iou_metric, 'iou_metric')
        tf.contrib.layers.summaries.summarize_tensor(
            tf.reduce_mean(tf.cast(self.iou_metric > 0.7, tf.float32)), 'iou_positive_rate')
        tf.contrib.layers.summaries.summarize_tensor(
            tf.reduce_mean(tf.cast(self.iou_metric < 0.3, tf.float32)), 'iou_negative_rate')
        tf.contrib.layers.summaries.summarize_tensor(
            tf.reduce_mean(tf.cast(self.cross_boundary_mask, tf.float32)), 'cross_rate'
        )

    def _create_variables(self):
        self.image_height, self.image_width = tf.placeholder(tf.int32), tf.placeholder(tf.int32)
        self.ground_truth_num = tf.placeholder(tf.int32)
        self.ground_truth_pre = tf.placeholder(tf.float32, [None, 4])

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
            self.ground_truth_num,
            self.anchors,
            self.offsets,
            (self.image_height // 16) * (self.image_width // 16) * self.k
        )

        reg_loss = sum(map(
            tf.reduce_sum,
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        ))

        reg_num = tf.cast((self.image_height // 16) * (self.image_width // 16), tf.float32)
        cls_num = tf.cast(self.batch_size, tf.float32)
        self.loss = (
            score_loss / cls_num +
            self.l1_coef * box_reg_loss / reg_num +
            reg_loss
        )

        # XXX: move to dedicated method
        tf.summary.scalar('score_loss', score_loss)
        tf.summary.scalar('l2_loss', reg_loss)
        tf.summary.scalar('box_regression_loss', box_reg_loss)


    def _create_train(self):
        # XXX: change to vanilla SGD?
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build(self):
        self._create_conv6()

        # VGG architecture - conv5 layer has 4 maxpools, hence 16 = 2 ** 4
        conv_height, conv_width = self.image_height // 16, self.image_width // 16
        proposals_num = conv_height * conv_width * self.k

        self.anchors = generate_anchors(
            self.boxes, self.image_height, self.image_width, conv_height, conv_width)

        self.offsets = tf.reshape(self.layers['offsets'], [proposals_num, 4])
        self.scores = tf.reshape(self.layers['scores'], [proposals_num, 2])
        self.anchors = tf.reshape(self.anchors, [proposals_num, 4])

        self.proposals = generate_proposals(self.offsets, self.anchors)
        self.cross_boundary_mask = cross_border_filter(
            self.proposals, self.image_height, self.image_width)
        self.ground_truth = centerize_ground_truth(self.ground_truth_pre)
        self.iou_metric = iou(self.ground_truth, self.ground_truth_num,
                              self.proposals, proposals_num)

        pos_batch, neg_batch = generate_batches(
            self.proposals, proposals_num,
            self.ground_truth, self.ground_truth_num,
            self.iou_metric, self.scores, self.cross_boundary_mask, self.batch_size)

        self.pos_boxes, self.pos_scores, self.true_pos_scores = pos_batch
        self.neg_boxes, self.neg_scores, self.true_neg_scores = neg_batch


    def _box_params_loss(self, ground_truth, ground_truth_num,
                         anchor_centers, offsets, proposals_num):
        # ground_truth shape is M x 4, where M is count and 4 are y,x,h,w
        ground_truth = tf.expand_dims(ground_truth, axis=0)
        ground_truth = tf.tile(ground_truth, [proposals_num, 1, 1])
        # anchor_centers shape is N x 4 where N is count and 4 are ya,xa,ha,wa
        anchor_centers = tf.expand_dims(anchor_centers, axis=1)
        anchor_centers = tf.tile(anchor_centers, [1, ground_truth_num, 1])
        # pos_sample_mask shape is N x M, True are for positive proposals and, hence,
        # for anchor centers
        pos_sample_mask = tf.greater(self.iou_metric, 0.7)
        # convert mask shape from N to N x 1 to make it broadcastable with pos_sample_mask
        mask = tf.expand_dims(self.cross_boundary_mask, axis=1)
        # convert resulting shape to align it with offsets
        mask = tf.expand_dims(tf.cast(pos_sample_mask & mask, tf.float32), axis=2)

        y_anchor, x_anchor, height_anchor, width_anchor = tf.unstack(anchor_centers, axis=2)
        y_ground_truth, x_ground_truth, height_ground_truth, width_ground_truth = tf.unstack(
            ground_truth, axis=2)

        # idea is to calculate N x M tx, ty, tw, th for ground truth boxes
        # for every proposal. Then we caclulate loss, multiply it with mask
        # to filter out non-positive samples and sum to one

        # each shape is N x M
        tx_ground_truth = (x_ground_truth - x_anchor) / width_anchor
        ty_ground_truth = (y_ground_truth - y_anchor) / height_anchor
        tw_ground_truth = tf.log(width_ground_truth / width_anchor)
        th_ground_truth = tf.log(height_ground_truth / height_anchor)

        gt_params = tf.stack(
            [ty_ground_truth, tx_ground_truth, th_ground_truth, tw_ground_truth], axis=2)

        offsets = tf.expand_dims(offsets, axis=1)
        offsets = tf.tile(offsets, [1, ground_truth_num, 1])

        return huber_loss((offsets - gt_params) * mask)

    def _create_conv6(self):
        # throw away first dimention - don't allow multiple images,
        # batches are generated internally from one image
        # slice all inputs to take first item

        conv = tf.contrib.layers.conv2d(
            self.input,
            self.filters_num,
            self.ksize,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            scope='conv6_1'
        )
        self.layers['conv6_1'] = conv

        offsets = tf.contrib.layers.conv2d(
            conv,
            4 * self.k,
            [1] * 2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_loss),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            activation_fn=None,
            scope='offsets'
        )  # H' x W' x 4k
        self.layers['offsets'] = tf.minimum(offsets[0], 10.0)

        scores = tf.contrib.layers.conv2d(
            conv,
            2 * self.k,
            [1] * 2,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_loss),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            activation_fn=None,
            scope='scores'
        )  # H' x W' x 2k
        self.layers['scores'] = scores[0]
