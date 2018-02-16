import numpy as np
import tensorflow as tf
import tf_util as U

class Model:
    def __init__(self, loss = "weighted"):
        self.loss = loss
        self.prob = tf.placeholder(shape=(), dtype=tf.float32)
        self.lr = tf.placeholder(shape=(), dtype=tf.float32)
    
    def block(self, net, filters):
        with tf.variable_scope('block_'+str(filters)):
            net = U.conv2d(net, filters, 'conv1', (3, 3))
            net = U.swish(net)
            net = U.conv2d(net, filters, 'conv2', (3, 3))
            net = U.swish(net)
            net = U.maxpool(net, 2)
        return net

    def build(self, image, mos_score):
        self.n_images  = tf.shape(mos_score)[0]
        self.n_patches = tf.shape(image)[0]
        net = self.block(image, 32)
        net = self.block(net, 64)
        net = self.block(net, 128)
        net = self.block(net, 256)
        net = self.block(net, 512)

        net1 = tf.reshape(net, (-1, 512))
        net1 = U.dense(net1, 512, 'fc1')
        net1 = U.swish(net1)
        net1 = tf.nn.dropout(net1, keep_prob = self.prob)
        net1 = U.dense(net1, 1, 'fc2')

        if self.loss == "patchwise":
            mos_score = tf.tile(mos_score, self.n_patches)
            self.loss_op = self.patchwise_loss(net1, mos_score)
            self.output = net1
        
        elif self.loss == "weighted":
            net2 = tf.reshape(net, (-1, 512))
            net2 = U.dense(net2, 512, 'fc1_weight')
            net2 = U.swish(net2)
            net2 = tf.nn.dropout(net2, keep_prob = self.prob)
            net2 = U.dense(net2, 1, 'fc2_weight')
            net2 = tf.nn.relu(net2) + 1e-6
            self.loss_op = self.weighted_loss(net1, net2, mos_score)

        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss_op)

        
    def patchwise_loss(self, h, t):
        t = U.repeat(t, self.n_patches//self.n_images)
        loss = tf.reduce_sum(tf.abs(h - t))
        return loss

    def weighted_loss(self, h, a, t):
        loss = 0
        
        h = tf.split(h, self.n_images, 0)
        a = tf.split(a, self.n_images, 0)
        t = tf.split(t, self.n_images, 0)

        for i in range(self.n_images):
            y = tf.reduce_sum(h[i] * a[i], 0) /  tf.reduce_sum(a[i], 0)
            self.output += y
            loss += tf.abs(y - t[i])
        
        loss /= self.n_images
        
        return loss
