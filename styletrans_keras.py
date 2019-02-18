import tensorflow as tf
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19

target_image_path = 'img/portrait.jpg'
style_reference_image_path = 'img/transfer_style_reference.jpg'

width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939  # B
    x[:, :, 1] += 116.779  # G
    x[:, :, 2] += 123.68  # R
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

from keras import backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(
    preprocess_image(style_reference_image_path))
combination_image = K.placeholder(shape=(1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image],axis=0)

model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('model loaded')

# define loss

# 内容损失
# 此处建议使用感知损失以提高性能
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 格拉姆矩阵
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# 风格损失
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels**2)*(size**2))

# 总变差损失
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :]
    )
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :]
    )
    return K.sum(K.pow(a + b, 1.25))


# Dictionalize layers
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
content_layer = 'block5_conv2'
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# initialize weights
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Add content loss
loss = K.variable(0.)
layer_features=outputs_dict[content_layer]
target_image_features=layer_features[0,:,:,:]
combination_features=layer_features[2,:,:,:]
loss += content_weight * content_loss(target_image_features, combination_features)

# Add style loss
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl

# Add total variation loss
loss += total_variation_weight * total_variation_loss(combination_image)



# define process
grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grads_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grads_values
        return self.loss_value
    def grads(self, x):
        assert self.loss_value is not None
        grads_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grads_values
evaluator = Evaluator()

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import time

result_prefix = 'my_result'
iterations = 20

x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration ', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=iterations)
    print('Current loss value: ',min_val)
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    print('Image saved as ', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
