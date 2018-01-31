from __future__ import absolute_import, division, print_function

import os
import Utils.utils as utils
import models.models_spgan as models
import argparse
import numpy as np
import tensorflow as tf
import utils.image_utils as im
from glob import glob


""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', dest='dataset', default='market2duke', help='which dataset to use')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
args = parser.parse_args()

dataset = args.dataset
crop_size = args.crop_size


""" run """
with tf.Session() as sess:
    a_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])
    b_real = tf.placeholder(tf.float32, shape=[None, crop_size, crop_size, 3])

    a2b = models.generator(a_real, 'a2b')
    b2a = models.generator(b_real, 'b2a')
    b2a2b = models.generator(b2a, 'a2b', reuse=True)
    a2b2a = models.generator(a2b, 'b2a', reuse=True)

    #--retore--#
    saver = tf.train.Saver()
    ckpt_path = utils.load_checkpoint('./checkpoints/' + dataset + '_spgan', sess, saver)
    saver.restore(sess, ckpt_path)

    if ckpt_path is None:
        raise Exception('No checkpoint!')
    else:
        print('Copy variables from % s' % ckpt_path)

    #--test--#
    a_list = glob('./datasets/' + dataset + '/bounding_box_train-Market/*.jpg')
    b_list = glob('./datasets/' + dataset + '/bounding_box_train-Duke/*.jpg')

    a_save_dir = './test_predictions/' + dataset + '_spgan' + '/bounding_box_train_market2duke/'
    b_save_dir = './test_predictions/' + dataset + '_spgan' + '/bounding_box_train_duke2market/'
    utils.mkdir([a_save_dir, b_save_dir])

		
    for i in range(len(a_list)):
        a_real_ipt = im.imresize(im.imread(a_list[i]), [crop_size, crop_size])
        a_real_ipt.shape = 1, crop_size, crop_size, 3
        a2b_opt = sess.run(a2b, feed_dict={a_real: a_real_ipt}) 
        a_img_opt =  a2b_opt

        img_name = os.path.basename(a_list[i])
        img_name = 'duke_'+img_name
        im.imwrite(im.immerge(a_img_opt, 1, 1), a_save_dir + img_name)
        print('Save %s' % (a_save_dir + img_name))


    
    for i in range(len(b_list)):
        b_real_ipt = im.imresize(im.imread(b_list[i]), [crop_size, crop_size])
        b_real_ipt.shape = 1, crop_size, crop_size, 3
        b2a_opt = sess.run(b2a, feed_dict={b_real: b_real_ipt})
        b_img_opt = b2a_opt
        img_name = os.path.basename(b_list[i])
        img_name = 'market_'+img_name
        im.imwrite(im.immerge(b_img_opt, 1, 1), b_save_dir + img_name)
        print('Save %s' % (b_save_dir + img_name))
		
