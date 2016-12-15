#%%
import tensorflow as tf
import numpy as np
import densecap.model
import importlib
import scipy.misc
import collections


import imagenet_classes
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
filename = '101310-of-108064'



with tf.Session() as sess:
    coord = tf.train.Coordinator()
    f_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    k, v = reader.read(f_queue)
    feats = tf.parse_single_example(v, features={
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/bbox/x': tf.VarLenFeature(tf.float32),
        'image/bbox/y': tf.VarLenFeature(tf.float32),
        'image/bbox/width': tf.VarLenFeature(tf.float32),
        'image/bbox/height': tf.VarLenFeature(tf.float32),
        'image/encoded': tf.FixedLenFeature([], tf.string)
    })
    H, W = feats['image/height'], feats['image/width']
    H, W = tf.cast(H, tf.int64), tf.cast(W, tf.int64)

    image = tf.decode_raw(feats['image/encoded'], tf.uint8)
    image = tf.reshape(image, tf.pack([W, H, 3]))

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print(H.eval())
    print(W.eval())
    img = image.eval()

    coord.request_stop()
    coord.join(threads)

    # image = tf.decode_raw(feats['image/encoded'], tf.uint8)
    # image.set_shape([W.eval(), H.eval()])

