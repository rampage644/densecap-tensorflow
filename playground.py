# pylint: disable=c0103
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import densecap.model
import importlib
import scipy.misc
import collections
import json


import imagenet_classes
import densecap.util as util
import densecap.model as model
#%%
tf.reset_default_graph()
importlib.reload(densecap.model)



#%%
for v in tf.global_variables():
    print(v.name, v.get_shape())

with np.load('data/vgg16_weights.npz') as ifile:
    for name, value in ifile.items():
        print(name, value.shape)

#%%
image = scipy.misc.imread('data/laska.png', mode='RGB')
image = image[:224, :224, :].astype(np.float32)
image -= [123.68, 116.779, 103.939]

#%%
tf.reset_default_graph()
importlib.reload(densecap.model)

H, W, _ = image.shape
vgg16 = densecap.model.VGG16(H, W)

_, Hp, Wp, _ = vgg16.layers['conv5_3'].get_shape().as_list()
sx, sy = H // Hp, W // Wp
anchor_centers = np.dstack(np.meshgrid(np.arange(-0.5, H - 0.5, sx), np.arange(-0.5, W - 0.5, sy)))

rpn = densecap.model.RegionProposalNetwork(vgg16.layers['conv5_3'], anchor_centers)
# rpn = densecap.model.RegionProposalNetwork(cfeatures, cfg)

#%%
def run():
    with tf.Session() as sess, np.load('data/vgg16_weights.npz') as ifile:
        sess.run(tf.global_variables_initializer())
        for v in tf.global_variables():
            name = v.name.replace('weights', 'W').replace('biases', 'b').replace('/', '_')[:-2]

            if name in ifile:
                print('Assigning {} [{}] from loaded {} [{}]'.format(v.name, v.get_shape(), name, ifile[name].shape))
                sess.run(tf.assign(v, ifile[name]))
        res = sess.run([rpn.layers['offsets'], rpn.layers['scores']], {vgg16.input: [image]})
    return res

offsets, scores = run()
scores.shape


#%%
tf.reset_default_graph()
# filename = '101310-of-108064'
filename = '00000-of-00010'



with tf.Session() as sess:
    coord = tf.train.Coordinator()
    f_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    k, v = reader.read(f_queue)
    feats = tf.parse_single_example(v, features={
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/bbox/x': tf.VarLenFeature(tf.float32),
        'image/bbox/y': tf.VarLenFeature(tf.float32),
        'image/bbox/width': tf.VarLenFeature(tf.float32),
        'image/bbox/height': tf.VarLenFeature(tf.float32),
        'image/encoded': tf.FixedLenFeature([], tf.string)
    })
    H, W = feats['image/height'], feats['image/width']
    H, W = tf.cast(H, tf.float32), tf.cast(W, tf.float32)

    image = feats['image/encoded']
    image = tf.image.decode_jpeg(image, channels=3)

    filename = feats['image/filename']
    filename = tf.decode_raw(filename, tf.uint8)

    bbox_xmin, bbox_ymin = feats['image/bbox/x'].values, feats['image/bbox/y'].values
    bbox_w, bbox_h = feats['image/bbox/width'].values, feats['image/bbox/height'].values

    bbox_xmax = bbox_xmin + bbox_w
    bbox_ymax = bbox_ymin + bbox_h

    bbox_xmin, bbox_xmax = bbox_xmin / W, bbox_xmax / W
    bbox_ymin, bbox_ymax = bbox_ymin / H, bbox_ymax / H

    boxes = tf.stack([bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax], axis=1)

    images = tf.image.draw_bounding_boxes(
        tf.expand_dims(tf.cast(image, tf.float32), axis=0), tf.expand_dims(boxes, axis=0))
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    img, imgs, H, W, filename = sess.run([
        image, images, H, W, filename])

    coord.request_stop()
    coord.join(threads)


print(filename.tobytes().decode('utf-8'))
print(H, W)
print(img.shape)
plt.figure(figsize=(12, 12))
plt.subplot(211); plt.imshow(img)
plt.subplot(212); plt.imshow(imgs.astype(np.uint8)[0])
plt.show()


#%%
importlib.reload(util)
d = 4

#%%
H, W = 100, 100
sh, sw = 10, 10

N = (H // sh) * (W // sw)
M = 2


regions = np.dstack(
    np.meshgrid(np.arange(0, H, sh), np.arange(0, W, sw).reshape(-1, 2))
)
sizes = np.tile(np.expand_dims(np.array([10, 10]), 0), [H, 1])
regions = regions.reshape(-1, 2)

np_proposals = np.hstack((regions, sizes)).astype(np.float)

np_gt = np.array([[50, 50, 10, 10], [0, 0, 20, 20]]).astype(np.float)


#%%
proposals = tf.Variable(np_proposals, dtype=tf.float32)
gt = tf.Variable(np_gt, dtype=tf.float32)

print(proposals.get_shape(), gt.get_shape())

proposals = tf.expand_dims(proposals, axis=1)
proposals = tf.tile(proposals, [1, M, 1])

gt = tf.expand_dims(gt, axis=0)
gt = tf.tile(gt, [N, 1, 1])

proposals = tf.reshape(proposals, (N*M, d))
gt = tf.reshape(gt, (N*M, d))

a = tf.stack([proposals, gt], axis=1)
print(a.get_shape())

def iou(x):
    # x is 2x4 tensor
    return util.tf_iou(x[0], x[1])


res = tf.map_fn(iou, a)

#%%
options = tf.cast(tf.Variable(np.random.rand(10, 3)), tf.float32)
scores = tf.cast(tf.Variable(np.radom.rand(10)), tf.float32)
zeros = tf.zeros(options.get_shape().as_list())
ones = tf.ones(options.get_shape().as_list())
options_1 = tf.reduce_sum(tf.select(tf.greater(options, 0.7), options, zeros), axis=1)
positive_mask = tf.greater(options_1, 0)

# change threshold to 0.3 as in paper
options_2 = tf.reduce_sum(tf.select(tf.less(options, 0.7), zeros, options), axis=1)
negative_mask = tf.equal(options_2, 0)


positive_boxes = tf.boolean_mask(options, positive_mask)
negative_boxes = tf.boolean_mask(options, negative_mask)

positive_scores = tf.boolean_mask(scores, positive_mask)
negative_scores = tf.boolean_mask(scores, negative_mask)



#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    base, res1, res2 = sess.run([options, indices_1, indices_2])

print(base)
print(res1)
print(res2)


#%%
# generate anchor_centers from boxes and given prev conv_layer
H, W = 224, 224
N, Hp, Wp, C = 10, 14, 14, 64  # batch, height, width, filters
k = 12  # boxes num
boxes = tf.Variable([
    (45, 90), (90, 45), (64, 64),
    (90, 180), (180, 90), (128, 128),
    (181, 362), (362, 181), (256, 256),
    (362, 724), (724, 362), (512, 512),
], dtype=tf.float32)
conv_layer = tf.constant(np.random.rand(N, Hp, Wp, C), dtype=tf.float32)

# those are strides in terms of original image
# i.e. what x and y base image strides corresponds to 1,1 conv layer stride
sh, sw = H // Hp, W // Wp

grid = tf.Variable(
    np.dstack(np.meshgrid(np.arange(-0.5, H - 0.5, sh), np.arange(-0.5, W - 0.5, sw))),
    dtype=tf.float32
)

# grid = tf.reshape(grid, [Hp * Wp, 2])
boxes = tf.expand_dims(tf.expand_dims(boxes, 0), 0)
grid = tf.expand_dims(grid, 2)

boxes.get_shape(), grid.get_shape()
tf.tile(boxes, [Hp * Wp, 1])
tf.tile(grid, [k, 1])

anchor_centers = tf.concat(3, [tf.tile(grid, [1, 1, k, 1]), tf.tile(boxes, [Hp, Wp, 1, 1])])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(anchor_centers)

result[0][0]

#%%
import json
bbox_filepath = 'output.json'
image_filepath = '1.jpg'

with open(bbox_filepath) as ifile:
    data = json.load(ifile)
image = scipy.misc.imread(image_filepath, mode='RGB')
gt = np.array([[r['x'], r['y'], r['width'], r['height']] for r in data])


importlib.reload(model)


tf.reset_default_graph()

H, W, _ = image.shape


inputs = tf.placeholder(tf.float32, shape=[1, None, None, 3])
H_input = tf.placeholder(tf.int32)
W_input = tf.placeholder(tf.int32)

vgg16 = model.VGG16(inputs)
rpn = model.RegionProposalNetwork(vgg16.layers['conv5_3'], H_input, W_input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss, _ = sess.run(
        [rpn.loss, rpn.train_op], {
            vgg16.input: [image],
            rpn.H: H,
            rpn.W: W,
            rpn.gt: gt[:64]
        }
    )

#%%
sh, sw = 16, 16
H_input = tf.placeholder(tf.int32)
W_input = tf.placeholder(tf.int32)
H, W = tf.cast(H_input, tf.float32), tf.cast(W_input, tf.float32)

# TODO: probably replace `numpy` ops with tf ones
grid = tf.stack(tf.meshgrid(
        tf.linspace(-0.5, H - 0.5, sh + 1),
        tf.linspace(-0.5, W - 0.5, sw + 1)), axis=2)


with tf.Session() as sess:
    # TODO: continue! check if grid is creatable with that code!
    # make H and W tf constants!
    res = sess.run(grid, {
        H_input: 224,
        W_input: 224
    })


#%%
# Going to check the speed of map_fn ops and try to improve things

N = 20000
M, d = 200, 4
gt = tf.random_uniform([M, d])
proposals = tf.random_uniform([N, d])

def run(proposals, gt, device='/cpu:0'):

    with tf.device(device):
        proposals = tf.expand_dims(proposals, axis=1)
        proposals = tf.tile(proposals, [1, M, 1])

        gt = tf.expand_dims(gt, axis=0)
        gt = tf.tile(gt, [N, 1, 1])

        proposals = tf.reshape(proposals, (N*M, d))
        gt = tf.reshape(gt, (N*M, d))

        # shape is N*M x 1
        iou_metric = tf.map_fn(model.iou, tf.stack([proposals, gt], axis=1))
        iou_metric = tf.reshape(iou_metric, [N, M])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iou_metric)

# result is 2min48s
%timeit -n 1 -r 1 run(proposals, gt)

def run2(proposals, gt, device):
    with tf.device(device):
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

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iou_metric)

# result is 427ms!
%timeit run2(proposals, gt, '/cpu:0')


#%%
importlib.reload(model)
tf.reset_default_graph()

image_input = tf.placeholder(tf.float32, shape=[1, None, None, 3])
vgg16 = model.VGG16(image_input)
rpn = model.RegionProposalNetwork(vgg16.layers['conv5_3'])
saver = tf.train.Saver()
saved_model = tf.train.latest_checkpoint('ckpt/')

k = 50
image = scipy.misc.imread('1.jpg', mode='RGB')
data = json.load(open('output.json'))
gt_boxes = np.array([[r['x'], r['y'], r['width'], r['height']]
                             for r in data]).astype(np.float32)
height, width, _ = image.shape

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, saved_model)

    boxes, scores = sess.run(
        [rpn.proposals, tf.nn.softmax(rpn.scores)], {
            rpn.H: height,
            rpn.W: width,
            vgg16.input: [image]
        })
    proposals = np.squeeze(boxes[np.argsort(scores[:, 1])][-k:])


    xmin, ymin, w, h = np.split(proposals, 4, axis=1)

    xmax, ymax = xmin + w, ymin + h
    xmin, xmax, ymin, ymax = xmin / w, xmax / w, ymin / h, ymax / h

    images = tf.placeholder(tf.float32, [1, height, width, 3])
    boxes_p = tf.placeholder(tf.float32, [1, k, 4])
    bbox_image = tf.image.draw_bounding_boxes(images, boxes_p)
    output_images = sess.run(bbox_image, {
        images: [image],
        boxes_p: np.expand_dims(np.hstack([ymin, xmin, ymax, xmax]), axis=0)
    })

    scipy.misc.imsave('output.png', output_images[0])

#%%
boxes.shape
scores.shape
np.argsort(scores[:, 1])
proposals
