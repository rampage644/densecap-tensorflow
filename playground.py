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

#%%



#%%
def f(output, ksize, stride):
    return (output - 1) * stride + ksize

#%%
o = 1

o = f(o, 3, 1)  # conv5_3 -> conv5_2, 3
o = f(o, 3, 1)  # conv5_3 -> conv5_1, 5
o = f(o, 2, 2)  # conv5_3 -> pool4, 10
o = f(o, 3, 1)  # conv5_3 -> conv4_3, 12
o = f(o, 3, 1)  # conv5_3 -> conv4_2, 14
o = f(o, 3, 1)  # conv5_3 -> conv4_1, 16
o = f(o, 2, 2)  # conv5_3 -> pool3, 32
o = f(o, 3, 1)  # conv5_3 -> conv3_3, 34
o = f(o, 3, 1)  # conv5_3 -> conv3_2, 36
o = f(o, 3, 1)  # conv5_3 -> conv3_1, 38
o = f(o, 2, 2)  # conv5_3 -> pool2, 76
o = f(o, 3, 1)  # conv5_3 -> conv2_2, 78
o = f(o, 3, 1)  # conv5_3 -> conv2_1, 80
o = f(o, 2, 2)  # conv5_3 -> pool1, 160
o = f(o, 3, 1)  # conv5_3 -> conv1_2, 162
o = f(o, 3, 1)  # conv5_3 -> conv1_1, 164
o = f(o, 3, 1)  # conv5_3 -> source, 166

print(o)