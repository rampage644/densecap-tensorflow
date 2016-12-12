#%%
import tensorflow as tf
import numpy as np
import densecap.model
import importlib
import scipy.misc


import imagenet_classes
#%%
tf.reset_default_graph()
importlib.reload(densecap.model)
vgg16 = densecap.model.VGG16(224, 224)
vgg16.layers


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
with tf.Session() as sess, np.load('data/vgg16_weights.npz') as ifile:
    sess.run(tf.global_variables_initializer())
    for v in tf.global_variables():
        name = v.name.replace('weights', 'W').replace('biases', 'b').replace('/', '_')[:-2]
        print('Assigning {} [{}] from loaded {} [{}]'.format(v.name, v.get_shape(), name, ifile[name].shape))
        sess.run(tf.assign(v, ifile[name]))
    scores = sess.run(vgg16.predicted, {vgg16.input: [image]})


for arg in np.argsort(scores[0])[:5]:
    print(scores[0][arg], imagenet_classes.class_names[arg])