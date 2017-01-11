# Description
Replicate study (that is, implementation) of _'DenseCap: Fully Convolutional Localization Networks for Dense Captioning'_ paper  with `tensorflow`

# Current state

**Project is frozen**. Only RPN part is implemented, but it's now working as intended. It was started as experiment but now it takes too much effort with no or little progress. Better to switch and (maybe) return to it later with more experience and knowledge.
Current _known_ problems are:

 * `Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory` message somewhere inside `_box_params_loss` fucntion.
 * Memory leak
 * No Localization Layer
 * No Language Layer
 * Model not actually working ;)

# Links

 * [Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Johnson_DenseCap_Fully_Convolutional_CVPR_2016_paper.pdf)
 * [Original code](https://github.com/jcjohnson/densecap)

# Helpful links

 * [VGG16](https://www.cs.toronto.edu/~frossard/post/vgg16/)
