import tensorflow as tf
import numpy as np

class TextFinder(object):

    def __init__(self):
        self.filter_size = 5
        self.learning_rate = 0.001
        image_size=128

        self.inp_text_image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])
        self.inp_text_out = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3])

        self.processed = self.processor(self.inp_text_image, self.filter_size, 'processor')

        self.loss = tf.reduce_mean(tf.square(self.processed - self.inp_text_out))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def processor(self, inp, filter_size, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            c1 = tf.layers.conv2d(inp, 16, filter_size, padding='SAME', activation=tf.nn.relu, name='c1')
            out = tf.tile(tf.layers.conv2d(c1, 1, filter_size, padding='SAME', activation=tf.nn.sigmoid, name='out'), [1, 1, 1, 3])
        return out


    def restore(self, path):
        self.saver.restore(self.sess, path)

    def save(self, path):
        self.saver.save(self.sess, path)

    def train(self, text_images, mask_outs):
        [_, loss] = self.sess.run([self.train_op, self.loss], feed_dict={self.inp_text_image: text_images,
                                                                         self.inp_text_out: mask_outs})
        return loss

    def produce_mask(self, text_images):
        [processed] = self.sess.run([self.processed], feed_dict={self.inp_text_image: text_images})
        return processed




