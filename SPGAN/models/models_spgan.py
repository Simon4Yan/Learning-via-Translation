from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import Utils.ops as ops
import functools
import tensorflow as tf
import tensorflow.contrib.slim as slim


conv = functools.partial(slim.conv2d, activation_fn=None)
deconv = functools.partial(slim.conv2d_transpose, activation_fn=None)
relu = tf.nn.relu
Mpool = tf.nn.max_pool 
Apool = tf.nn.avg_pool
FC = functools.partial(slim.fully_connected, activation_fn=None)
lrelu = functools.partial(ops.leak_relu, leak=0.2)
instance_norm = ops.instance_norm


def discriminator(img, scope, df_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        h0 = lrelu(conv(img, df_dim, 4, 2, scope='h0_conv'))    # h0 is (128 x 128 x df_dim)
        h1 = lrelu(instance_norm(conv(h0, df_dim * 2, 4, 2, scope='h1_conv'), scope='h1_instance_norm'))  # h1 is (64 x 64 x df_dim*2)
        h2 = lrelu(instance_norm(conv(h1, df_dim * 4, 4, 2, scope='h2_conv'), scope='h2_instance_norm'))  # h2 is (32x 32 x df_dim*4)
        h3 = lrelu(instance_norm(conv(h2, df_dim * 8, 4, 1, scope='h3_conv'), scope='h3_instance_norm'))  # h3 is (32 x 32 x df_dim*8)
        h4 = conv(h3, 1, 4, 1, scope='h4_conv')  # h4 is (32 x 32 x 1)
        return h4
		
def metric_net(img, scope, df_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    with tf.variable_scope(scope + '_discriminator', reuse=reuse):
        h0 = lrelu(conv(img, df_dim, 4, 2, scope='h0_conv'))    # h0 is (128 x 128 x df_dim)
        pool1 = Mpool(h0, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
		
        h1 = lrelu(conv(pool1, df_dim * 2, 4, 2, scope='h1_conv'))  # h1 is (32 x 32 x df_dim*2)
        pool2 = Mpool(h1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
		
        h2 = lrelu(conv(pool2, df_dim * 4, 4, 2, scope='h2_conv'))  # h2 is (8 x 8 x df_dim*4)
        pool3 = Mpool(h2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
		
        h3 = lrelu(conv(pool3, df_dim * 8, 4, 2, scope='h3_conv'))  # h3 is (2 x 2 x df_dim*4)
        pool4 = Mpool(h3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
		
        shape = pool4.get_shape()
        flatten_shape = shape[1].value * shape[2].value * shape[3].value
        h3_reshape = tf.reshape(pool4, [-1, flatten_shape], name = 'h3_reshape')
		
        fc1 = lrelu(FC(h3_reshape, df_dim*2, scope='fc1'))
        dropout_fc1 = slim.dropout(fc1, 0.5, scope='dropout_fc1')  
        net = FC(dropout_fc1, df_dim, scope='fc2') 
        
        #print_activations(net)
        #print_activations(pool4)
        return net

def generator(img, scope, gf_dim=64, reuse=False, train=True):

    bn = functools.partial(slim.batch_norm, scale=True, is_training=train,
                           decay=0.9, epsilon=1e-5, updates_collections=None)

    def residule_block(x, dim, scope='res'):
        y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = relu(instance_norm(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv1'), scope=scope + '_instance_norm1'))
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        y = instance_norm(conv(y, dim, 3, 1, padding='VALID', scope=scope + '_conv2'), scope=scope + '_instance_norm2')
        return y + x

    with tf.variable_scope(scope + '_generator', reuse=reuse):
        c0 = tf.pad(img, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = relu(instance_norm(conv(c0, gf_dim, 7, 1, padding='VALID', scope='c1_conv'), scope='c1_instance_norm'))
        c2 = relu(instance_norm(conv(c1, gf_dim * 2, 3, 2, scope='c2_conv'), scope='c2_instance_norm'))
        c3 = relu(instance_norm(conv(c2, gf_dim * 4, 3, 2, scope='c3_conv'), scope='c3_instance_norm'))

        r1 = residule_block(c3, gf_dim * 4, scope='r1')
        r2 = residule_block(r1, gf_dim * 4, scope='r2')
        r3 = residule_block(r2, gf_dim * 4, scope='r3')
        r4 = residule_block(r3, gf_dim * 4, scope='r4')
        r5 = residule_block(r4, gf_dim * 4, scope='r5')
        r6 = residule_block(r5, gf_dim * 4, scope='r6')
        r7 = residule_block(r6, gf_dim * 4, scope='r7')
        r8 = residule_block(r7, gf_dim * 4, scope='r8')
        r9 = residule_block(r8, gf_dim * 4, scope='r9')

        d1 = relu(instance_norm(deconv(r9, gf_dim * 2, 3, 2, scope='d1_dconv'), scope='d1_instance_norm'))
        d2 = relu(instance_norm(deconv(d1, gf_dim, 3, 2, scope='d2_dconv'), scope='d2_instance_norm'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = conv(d2, 3, 7, 1, padding='VALID', scope='pred_conv')
        pred = tf.nn.tanh(pred)

        return pred
def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())