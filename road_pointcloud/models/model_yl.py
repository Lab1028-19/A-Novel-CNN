import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util

def placeholder_inputs(batch_size, image_size_h, image_size_w, point_channel):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, image_size_h,image_size_w, point_channel))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, image_size_h*image_size_w))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    image_h = point_cloud.get_shape()[1].value
    image_w = point_cloud.get_shape()[2].value
    num_point = image_h*image_w
    input_image = point_cloud
    print('input_image',input_image.shape)
    # CONV
    net = tf_util.conv2d(input_image, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    print('conv4',net.shape)
    points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    print('feat1',points_feat1.shape)
    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [image_h,image_w], padding='VALID', scope='maxpool1')
    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    print('feat fc1',pc_feat1.shape)
   
    # CONCAT 
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, image_h, image_w, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
    print('concat1 ',points_feat1_concat.shape)
    # CONV 
    net = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 5, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv8')
    print('convf ',net.shape)
    net = tf.reshape(net,[batch_size,num_point,5])
   # net = tf.squeeze(net, [2])
    print('output ',net.shape)
    return net

def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,64,360,7))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,64,360,7)})
            print(time.time() - start)
