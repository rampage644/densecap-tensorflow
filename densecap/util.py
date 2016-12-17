import math
import tensorflow as tf


def iou(box1, box2):
    '''Caclulate IoU of two boxes'''
    x11, y11, w1, h1 = box1
    x12, y12 = x11 + w1, y11 + h1
    x21, y21, w2, h2 = box2
    x22, y22 = x21 + w2, y21 + h2

    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))

    intersection = x_overlap * y_overlap

    return 1. * intersection / (w1 * h1 + w2 * h2 - intersection)


def tf_iou(box1, box2):
    x11, y11, w1, h1 = tf.split(0, 4, box1)
    x12, y12 = x11 + w1, y11 + h1
    x21, y21, w2, h2 = tf.split(0, 4, box2)
    x22, y22 = x21 + w2, y21 + h2

    x_overlap = tf.maximum(0.0, tf.minimum(x12, x22) - tf.maximum(x11, x21))
    y_overlap = tf.maximum(0.0, tf.minimum(y12, y22) - tf.maximum(y11, y21))

    intersection = x_overlap * y_overlap

    return 1. * intersection / (w1 * h1 + w2 * h2 - intersection)
