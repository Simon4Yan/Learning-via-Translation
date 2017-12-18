from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import Utils.ops as ops
import Utils.data as data
import Utils.utils as utils
import models.models_spgan as models
import argparse
import numpy as np
import tensorflow as tf
import Utils.image_utils as im
import time
import sys
from glob import glob



""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='market2duke', help='which dataset to use')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', dest='epoch', type=int, default=6, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=4, help='GPU ID')
args = parser.parse_args()

dataset = args.dataset
load_size = args.load_size
crop_size = args.crop_size
epoch = args.epoch
batch_size = args.batch_size
lr = args.lr
gpu_id = args.gpu_id


"""parameters"""
lambda1 = 10.0
lambda2 = 5.0
lambda3 = 2.0
margin = 2.0

""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' graph '''
    # nodes
    a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    a2b_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    b2a_sample = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

    a2b = models.generator(a_real, 'a2b')
    b2a = models.generator(b_real, 'b2a')
    b2a2b = models.generator(b2a, 'a2b', reuse=True)
    a2b2a = models.generator(a2b, 'b2a', reuse=True)
    
    
    b2b = models.generator(b_real, 'a2b', reuse=True)
    a2a = models.generator(a_real, 'b2a', reuse=True)
    
    a_dis = models.discriminator(a_real, 'a')
    b2a_dis = models.discriminator(b2a, 'a', reuse=True)
    b2a_sample_dis = models.discriminator(b2a_sample, 'a', reuse=True)
    
    b_dis = models.discriminator(b_real, 'b')
    a2b_dis = models.discriminator(a2b, 'b', reuse=True)
    a2b_sample_dis = models.discriminator(a2b_sample, 'b', reuse=True)
	
    # siamese network
    a_metric = tf.nn.l2_normalize(models.metric_net(a_real, 'metric'), 1)
    a2b_metric = tf.nn.l2_normalize(models.metric_net(a2b, 'metric', reuse=True), 1)
	
    b_metric = tf.nn.l2_normalize(models.metric_net(b_real, 'metric', reuse=True), 1)
    b2a_metric = tf.nn.l2_normalize(models.metric_net(b2a, 'metric', reuse=True), 1)
	
	#------------------- Postive Pair----------------#
    C = tf.constant(margin, name="C")
    S_eucd_pos = tf.pow(tf.subtract(a_metric, a2b_metric), 2)
    S_metric_POS = tf.reduce_sum(S_eucd_pos, 1)
    S_eucd_neg = tf.pow(tf.subtract(a_metric, b_metric), 2)
    S_eucd_neg = tf.reduce_sum(S_eucd_neg, 1)
    S_eucd_neg = tf.sqrt(S_eucd_neg + 1e-6)
    S_metric_NEG = tf.pow(tf.maximum(tf.subtract(C, S_eucd_neg), 0), 2)
    S_metric_LOSS = S_metric_NEG + S_metric_POS
	
    #------------------- Negative Pair----------------#   
    T_eucd_pos = tf.pow(tf.subtract(b_metric, b2a_metric), 2)
    T_metric_POS = tf.reduce_sum(T_eucd_pos, 1)
    T_eucd_neg = tf.pow(tf.subtract(b_metric,a_metric), 2)
    T_eucd_neg = tf.reduce_sum(T_eucd_neg, 1)
    T_eucd_neg = tf.sqrt(T_eucd_neg + 1e-6)
    T_metric_NEG = tf.pow(tf.maximum(tf.subtract(C, T_eucd_neg), 0), 2)	
    T_metric_LOSS = T_metric_NEG + T_metric_POS
	# contrastive loss
    m_loss = tf.identity((S_metric_LOSS + T_metric_LOSS)/4.0 , name = 'metric_losses') 
	
    # losses
    g_loss_a2b = tf.identity(ops.l2_loss(a2b_dis, tf.ones_like(a2b_dis)), name='g_loss_a2b')
    g_loss_b2a = tf.identity(ops.l2_loss(b2a_dis, tf.ones_like(b2a_dis)), name='g_loss_b2a')
    g_orig = g_loss_a2b + g_loss_b2a
    cyc_loss_a = tf.identity(ops.l1_loss(a_real, a2b2a) * lambda1, name='cyc_loss_a')
    cyc_loss_b = tf.identity(ops.l1_loss(b_real, b2a2b) * lambda1, name='cyc_loss_b')
    cyc_loss = cyc_loss_a + cyc_loss_b

    # identity loss
    idt_losss_b = tf.identity(ops.l1_loss(b2b, b_real) * lambda2, name='idt_loss_b')
    idt_losss_a = tf.identity(ops.l1_loss(a2a, a_real) * lambda2, name='idt_loss_a')
    idt_loss = idt_losss_b + idt_losss_a
    
    g_loss = g_loss_a2b + g_loss_b2a + cyc_loss + idt_loss +  lambda3 * m_loss
    
    d_loss_a_real = ops.l2_loss(a_dis, tf.ones_like(a_dis))
    d_loss_b2a_sample = ops.l2_loss(b2a_sample_dis, tf.zeros_like(b2a_sample_dis))
    d_loss_a = tf.identity((d_loss_a_real + d_loss_b2a_sample) / 2.0, name='d_loss_a')
    
    d_loss_b_real = ops.l2_loss(b_dis, tf.ones_like(b_dis))
    d_loss_a2b_sample = ops.l2_loss(a2b_sample_dis, tf.zeros_like(a2b_sample_dis))
    d_loss_b = tf.identity((d_loss_b_real + d_loss_a2b_sample) / 2.0, name='d_loss_b')

    # summaries
    g_summary = ops.summary_tensors([g_loss_a2b, g_loss_b2a, cyc_loss_a, cyc_loss_b, idt_losss_a, idt_losss_b, m_loss])
    d_summary_a = ops.summary(d_loss_a)
    d_summary_b = ops.summary(d_loss_b)
    metric_summary = ops.summary(m_loss)

    ''' optim '''
    t_var = tf.trainable_variables()
    d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
    d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
    g_var = [var for var in t_var if 'a2b_generator' in var.name or 'b2a_generator' in var.name]
    metric_var = [var for var in t_var if 'metric_discriminator' in var.name]
	
    d_a_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
    d_b_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
    g_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(g_loss, var_list=g_var)
    metric_train_op = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(m_loss, var_list=metric_var)

""" train """
''' init '''
# session
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# counter
it_cnt, update_cnt = ops.counter()

'''data'''
a_img_paths = glob('./datasets/' + dataset + '/bounding_box_train-Market/*.jpg')
b_img_paths = glob('./datasets/' + dataset + '/bounding_box_train-Duke/*.jpg')
a_data_pool = data.ImageData(sess, a_img_paths, batch_size, load_size=load_size, crop_size=crop_size)
b_data_pool = data.ImageData(sess, b_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

a_test_img_paths = glob('./datasets/' + dataset + '/bounding_box_train-Market/*.jpg')
b_test_img_paths = glob('./datasets/' + dataset + '/bounding_box_train-Duke/*.jpg')
a_test_pool = data.ImageData(sess, a_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)
b_test_pool = data.ImageData(sess, b_test_img_paths, batch_size, load_size=load_size, crop_size=crop_size)

a2b_pool = utils.ItemPool()
b2a_pool = utils.ItemPool()

'''summary'''
summary_writer = tf.summary.FileWriter('./summaries/' + dataset + '_spgan' , sess.graph)

'''saver'''
ckpt_dir = './checkpoints/' + dataset + '_spgan'
utils.mkdir(ckpt_dir + '/')

saver = tf.train.Saver(max_to_keep=30)
ckpt_path = utils.load_checkpoint(ckpt_dir, sess, saver)
if ckpt_path is None:
    sess.run(tf.global_variables_initializer())
else:
    print('Copy variables from % s' % ckpt_path)

'''train'''
try:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_epoch = min(len(a_data_pool), len(b_data_pool)) // batch_size
    max_it = epoch * batch_epoch

    now = time.strftime("%c")
    print('================ Begining Training time (%s) ================\n' % now)
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)
        # prepare data
        a_real_ipt = a_data_pool.batch()
        b_real_ipt = b_data_pool.batch()
        a2b_opt, b2a_opt = sess.run([a2b, b2a], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        a2b_sample_ipt = np.array(a2b_pool(list(a2b_opt)))
        b2a_sample_ipt = np.array(b2a_pool(list(b2a_opt)))
        
        # train G
        g_summary_opt, _  = sess.run([g_summary, g_train_op], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
        summary_writer.add_summary(g_summary_opt, it)
        # train D_b
        d_summary_b_opt, _ = sess.run([d_summary_b, d_b_train_op], feed_dict={b_real: b_real_ipt, a2b_sample: a2b_sample_ipt})
        summary_writer.add_summary(d_summary_b_opt, it)
        # train D_a
        d_summary_a_opt, _ = sess.run([d_summary_a, d_a_train_op], feed_dict={a_real: a_real_ipt, b2a_sample: b2a_sample_ipt})
        summary_writer.add_summary(d_summary_a_opt, it)
		# train metric
        metric_summary_opt, _ = sess.run([metric_summary, metric_train_op], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt, a2b: a2b_opt, b2a: b2a_opt})
        summary_writer.add_summary(metric_summary_opt, it)
        

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1
	
        # display
        if it % 20 == 0:
            print('-------------------------------------------------------')
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

	    # save
        if (((it + 1) % 1000 == 0) or (it + 1 == batch_epoch)):
            now = time.strftime("%c")
            print('================ Training time (%s) ================\n' % now)
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)
   
        # sample
        if (it + 1) % 100 == 0:
            a_real_ipt = a_test_pool.batch()
            b_real_ipt = b_test_pool.batch()
            [a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt] = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
            sample_opt = np.concatenate((a_real_ipt, a2b_opt, a2b2a_opt, b_real_ipt, b2a_opt, b2a2b_opt), axis=0)
            save_dir = './sample_images_while_training/' + dataset + '_spgan'
            utils.mkdir(save_dir + '/')
            im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))
			
except Exception, e:
    coord.request_stop(e)
finally:
    print("Stop threads and close session!")
    coord.request_stop()
    coord.join(threads)
    sess.close()
