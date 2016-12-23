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



