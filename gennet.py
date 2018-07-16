import tensorflow as tf
from PIL import Image
import numpy as np

'''
=================================================================
Generate any networks needed, add one convolution layer each time
written by xueqian Zhang.
-------------------
Example:
    from gennet import gennet
    ...
    
    net = gennet('netname')
    net.add_layer([3, 3, 3, 64], [64], 'conv1_1')
    net.add_layer([3, 3, 64, 64], [64], 'conv1_2')
    net.maxpooling()
    net.add_layer([3, 3, 64, 128], [128], 'conv2_1')
    net.add_layer([3, 3, 64, 128], [128], 'conv2_2')
    ...
    net.fc(4096, 'fc1')
    net.fc(4096, 'fc2')
    net.fc(1000, 'fc3')
    ...
    net.softmax()
    net.generate('path/to/ckpt/file')
=================================================================
'''

class gennet():
    def __init__(self, name, height, width, channel):
        self.name = name
        self.x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
        self.y = tf.placeholder(tf.int64, shape=[None, 2], name='labels')
        self.net = None
        self.net_shape = None
        self.isflattened = False
        self.fcout = None
    
    # Initialize weight for each layer
    def weight_var(self, shape, name='weights'):
        init = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(init, name=name)
        
    # Initialize bias for each layer
    def bias_var(self, shape, name='bias'):
        init = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(init, name=name)
    
    # Conv2d
    def conv2d(self, data, w, padding='SAME'):
        return tf.nn.conv2d(data, w, [1, 1, 1, 1], padding=padding)
    
    # Use tf.nn.max_pool to solve maxpooling
    def pool(self, data, name='pool', padding='SAME'):
        return tf.nn.max_pool(data, ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    # Create full-connection layer
    def fullconnection(self, data, w, b):
        return tf.matmul(data, w) + b
        #return tf.tensordot(data, w, axes = [[1], [0]]) + b
        
    # Add new layer to net, note that layer_name cannot be None or same as other layers
    def add_layer(self, kernel_shape, bias_shape, layer_name, padding='SAME', 
                  kernel_name='weights', bias_name='bias'):
        with tf.name_scope(layer_name) as scope:
            kernel = self.weight_var(kernel_shape, kernel_name)
            bias = self.bias_var(bias_shape, bias_name)
            if self.net == None:
                self.net = tf.nn.relu(self.conv2d(self.x, kernel, padding=padding) + bias, name=scope)
            else:
                self.net = tf.nn.relu(self.conv2d(self.net, kernel, padding=padding) + bias, name=scope)
            self.net_shape = self.net.get_shape()[1:]
    
    # Solve maxpooling
    def maxpooling(self, padding='SAME'):
        if self.net == None:
            raise IOError('The input of maxpooling is None, maybe there is no conv layer?')
        self.net = self.pool(self.net, padding=padding)
        self.net_shape = self.net.get_shape()[1:]
    
    # Add fully connected layer
    def fc(self, fc_channel, layer_name):
        if self.net_shape == None or self.net == None:
            raise IOError('The input of fc is None, maybe there is no conv layer?')
        if not self.isflattened:
            raise IOError('You has not flattened input layer.')
        with tf.name_scope(layer_name) as scope:
            #shape = [int(np.prod(self.net_shape))]
            #shape = self.net.get_shape().as_list()[1:]
            #shape.append(fc_channel)
            #print([-1]+shape+[fc_channel])
            #raise IOError('111')
            kernel = self.weight_var([self.fcout, fc_channel])
            #print(kernel.shape)
            bias = self.bias_var([fc_channel])
            #pool5_flat = tf.reshape(self.net, [-1]+shape)
            self.net = tf.nn.relu(self.fullconnection(self.net, kernel, bias), name=scope)
            self.net_shape = self.net.get_shape()
            self.fcout = fc_channel
            #print(self.net_shape)
    
    # Add softmax layer to net
    def softmax(self, name='softmax'):
        if self.net_shape == None or self.net == None:
            raise IOError('The input of softmax is None, maybe there is no conv layer?')
        self.net = tf.nn.softmax(self.net, name=name)
        
    def flatten(self, name='flatten'):
        if self.net_shape == None or self.net == None:
            raise IOError('The input of flatten is None, maybe there is no conv layer?')
        #shape = self.net_shape
        shape = int(np.prod(self.net_shape))
        #print(self.net.get_shape().as_list()[1])
        #raise IOError('www')
        poolx_flat = tf.reshape(self.net, [-1, shape])
        #print(poolx_flat.shape)
        self.fcout = shape
        self.net = poolx_flat
        self.net_shape = self.net.get_shape()
        self.isflattened = True
        
    # Generate *.ckpt file
    def generate(self, output_dir):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, output_dir)
    
