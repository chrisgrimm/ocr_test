import tensorflow as tf
import numpy as np

def component(function):
    def wrapper(*args, **kwargs):
        reuse = kwargs.get('reuse', None)
        name = kwargs['name']
        if 'reuse' in kwargs:
            del kwargs['reuse']
        del kwargs['name']
        with tf.variable_scope(name, reuse=reuse):
            out = function(*args, **kwargs)
            variables = tf.get_variable_scope().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            return out, variables
    return wrapper

def horz_stack_images(*images, spacing=5, background_color=(0,0,0)):
    # assert that all shapes have the same siz
    if len(set([tuple(image.shape) for image in images])) != 1:
        raise Exception('All images must have same shape')
    if images[0].shape[2] != len(background_color):
        raise Exception('Depth of background color must be the same as depth of image.')
    height = images[0].shape[0]
    width = images[0].shape[1]
    depth = images[0].shape[2]
    canvas = np.ones([height, width*len(images) + spacing*(len(images) - 1), depth])
    bg_color = np.reshape(background_color, [1, 1, depth])
    canvas *= bg_color
    width_pos = 0
    for image in images:
        canvas[:, width_pos:width_pos+width, :] = image
        width_pos += (width + spacing)
    return canvas