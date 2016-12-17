# pylint: disable=c0103
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import densecap.model
import importlib
import scipy.misc
import collections


import imagenet_classes
import densecap.util as util
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

N = (H / sh) * (W / sw)
M = 2


regions = np.dstack(
    np.meshgrid(np.arange(0, H, sh), np.arange(0, W, sw).reshape(-1, 2))
)
sizes = np.tile(np.expand_dims(np.array([10, 10]), 0), [H, 1])
regions = regions.reshape(-1, 2)

np_proposals = np.hstack((regions, sizes))

np_gt = np.array([[50, 50, 10, 10], [0, 0, 20, 20]])


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

a = tf.stack([gt, gt], axis=1)
print(a.get_shape())


def iou(x):
    # x is 2x4 tensor
    return util.tf_iou(x[0], x[1])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = tf.map_fn(iou, a)

    result = sess.run(res)

result.reshape((N, M))

