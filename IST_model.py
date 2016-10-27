from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import sys

from six.moves import urllib
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self,model_file, height,width):
        if model_file is None:
            logging.error('please inp ut model file')
        if not os.path.isfile(model_file):
            logging.error(('model file is not exist:'), model_file)
# load VGG19 model parameter
        self.param_dict = np.load(model_file).item()
        print('model file loaded')
# set layer weight of style loss
        self.STYLE_LAYERS = [
                    ('conv1_1', 1.0),
                    ('conv2_1', 1.0), 
                    ('conv3_1', 1.0), 
                    ('conv4_1', 1.0), 
                    ('conv5_1', 1.0) ]
        self.IMAGE_HEIGHT = height
        self.IMAGE_WIDTH = width
        self.graph={}

# get fiter and bias parameter, set untrainable
    def get_conv_filter(self,name):
        var = self.param_dict[name][0]
        print('conv Layer name: %s' % name)
        print('conv Layer shape: %s' % str(var.shape))
        return var

        ''' with tf.variable_scope(name) as scope:
            init = tf.constant_initializer(value=self.param_dict[name][0], dtype=tf.float32)
            shape = self.param_dict[name][0].shape
            print('conv Layer name: %s' % name)
            print('conv Layer shape: %s' % str(shape))
            var = tf.get_variable(name = 'filter', initializer=init, shape=shape, trainable=False)
        '''
    def get_conv_bias(self,name):
        var = self.param_dict[name][1]
        print('conv Layer name: %s' % name)
        print('conv Layer shape: %s' % str(var.shape))
        return var
        '''with tf.variable_scope(name) as scope:
            init = tf.constant_initializer(value=self.param_dict[name][1], dtype=tf.float32)
            shape = self.param_dict[name][1].shape
            var = tf.get_variable(name = 'bias', initializer=init, shape=shape, trainable=False)
        '''
    
# get conv_layer
    def conv_layer(self, bottom, name):
        filter = self.get_conv_filter(name)
        bias = self.get_conv_bias(name)
        conv = tf.nn.conv2d(bottom, filter, strides=[1 ,1 ,1 ,1], padding='SAME')
        relu = tf.nn.relu( tf.nn.bias_add(conv, bias) )
        return relu

# construct vgg19 conv layers model
    def constructModel(self):
        self.graph['input'] = tf.Variable(np.zeros((1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3)), dtype = 'float32')
        self.graph['conv1_1'] = self.conv_layer(self.graph['input'],'conv1_1')
        self.graph['conv1_2'] = self.conv_layer(self.graph['conv1_1'],'conv1_2')
        self.graph['pool1'] = tf.nn.avg_pool(self.graph['conv1_2'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')

        self.graph['conv2_1'] = self.conv_layer(self.graph['pool1'],'conv2_1')
        self.graph['conv2_2'] = self.conv_layer(self.graph['conv2_1'],'conv2_2')
        self.graph['pool2'] = tf.nn.avg_pool(self.graph['conv2_2'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')

        self.graph['conv3_1'] = self.conv_layer(self.graph['pool2'],'conv3_1')
        self.graph['conv3_2'] = self.conv_layer(self.graph['conv3_1'],'conv3_2')
        self.graph['conv3_3'] = self.conv_layer(self.graph['conv3_2'],'conv3_3')
        self.graph['conv3_4'] = self.conv_layer(self.graph['conv3_3'],'conv3_4')
        self.graph['pool3'] = tf.nn.avg_pool(self.graph['conv3_4'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')
        
        self.graph['conv4_1'] = self.conv_layer(self.graph['pool3'],'conv4_1')
        self.graph['conv4_2'] = self.conv_layer(self.graph['conv4_1'],'conv4_2')
        self.graph['conv4_3'] = self.conv_layer(self.graph['conv4_2'],'conv4_3')
        self.graph['conv4_4'] = self.conv_layer(self.graph['conv4_3'],'conv4_4')
        self.graph['pool4'] = tf.nn.avg_pool(self.graph['conv4_4'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')

        self.graph['conv5_1'] = self.conv_layer(self.graph['pool4'],'conv5_1')
        self.graph['conv5_2'] = self.conv_layer(self.graph['conv5_1'],'conv5_2')
        self.graph['conv5_3'] = self.conv_layer(self.graph['conv5_2'],'conv5_3')
        self.graph['conv5_4'] = self.conv_layer(self.graph['conv5_3'],'conv5_4')
        self.graph['pool5'] = tf.nn.avg_pool(self.graph['conv5_4'],ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5')

# get content loss
    def get_content_loss(self,sess):
        def _content_loss(p,x):
            M = p.shape[1] * p.shape[2]
            N = p.shape[3]
            return (1/(2*M*N)) * tf.reduce_sum(tf.pow(x-p,2))
        return _content_loss(sess.run(self.graph['conv4_2']), self.graph['conv4_2'])

# get style loss
    def get_style_loss(self,sess):

        def _gram_matrix(F, M, N):
            Ft = tf.reshape(F,(M,N))
            return tf.matmul(tf.transpose(Ft),Ft)
        def _style_loss(a,x):
            M = a.shape[1]*a.shape[2]
            N = a.shape[3]
            A = _gram_matrix(a,M,N)
            G = _gram_matrix(x,M,N)
            return (1/(4*N*N*M*M))*tf.reduce_sum(tf.pow(G-A,2))
# compute layerwised loss 
        E = [_style_loss(sess.run(self.graph[layer_name]),self.graph[layer_name]) for layer_name,_ in self.STYLE_LAYERS]
        W = [w for _,w in self.STYLE_LAYERS]
# add layerwised loss by weight
        loss = sum([ W[i]*E[i] for i in range(len(self.STYLE_LAYERS)) ])
        return loss

# optimizer
    def optimizerImage(self,total_loss, lr):
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(total_loss)
        return self.optimizer

