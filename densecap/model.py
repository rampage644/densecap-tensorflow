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

