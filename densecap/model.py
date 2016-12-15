'''Models'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import tensorflow as tf


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
    def __init__(self, conv_layer, anchor_centers):
        self.boxes = [
            (100, 100)
        ]  # one box only, k=1
        ha, wa = tf.constant(100, dtype=tf.float32), tf.constant(100, dtype=tf.float32)
        self.anchor_centers = tf.constant(anchor_centers, tf.float32)  # H' x W' x 2, 2 = (Xa, Ya)

        self.input = conv_layer  # should be (N, H', W', C) tensor
        self.boxes_num = len(self.boxes)  # k in paper
        self.filters_num = 256
        self.ksize = [3, 3]

        self.layers = {}
        self._build()

    def _build(self):
        conv = tf.contrib.layers.conv2d(
            self.input,
            self.filters_num,
            self.ksize,
            scope='conv6_1'
        )
        self.layers['conv6_1'] = conv

        offsets = tf.contrib.layers.conv2d(
            conv,
            4 * self.boxes_num,
            [1] * 2,
            scope='offsets'
        )  # N x H' x W' x 4k
        self.layers['offsets'] = offsets

        scores = tf.contrib.layers.conv2d(
            conv,
            1 * self.boxes_num,
            [1] * 2,
            scope='scores'
        )  # N x H' x W' x k
        self.layers['scores'] = scores

        tx = offsets[:, :, :, 0]  # tx
        ty = offsets[:, :, :, 1]  # ty
        th = offsets[:, :, :, 2]  # th
        tw = offsets[:, :, :, 3]  # tw

        x = self.anchor_centers[:, :, :, 0] + tx * wa
        y = self.anchor_centers[:, :, :, 1] + ty * ha
        w = wa * tf.exp(tw)
        h = ha * tf.exp(th)
