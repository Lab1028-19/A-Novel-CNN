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
    num_classes=9
    input_image = point_cloud
    print('___________________________model_structure____________________________________')
    print('input_image',input_image.shape)
    # CONV
#    net = tf_util.conv2d(input_image, 64, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
#    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
#    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
#    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
#    print('conv4',net.shape)
#    points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
#    print('feat1',points_feat1.shape)
#    # MAX
#    pc_feat1 = tf_util.max_pool2d(points_feat1, [image_h,image_w], padding='VALID', scope='maxpool1')
#    # FC
#    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
#    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
#    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
#    print('feat fc1',pc_feat1.shape)
#   
#    # CONCAT 
#    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, image_h, image_w, 1])
#    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])
#    print('concat1 ',points_feat1_concat.shape)
#    # CONV 
#    net = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv6')
#    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
#                         bn=True, is_training=is_training, scope='conv7')
#    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
#    net = tf_util.conv2d(net, 5, [1,1], padding='VALID', stride=[1,1],
#                         activation_fn=None, scope='conv8')
#    print('convf ',net.shape)
#    res_1 = tf_util.conv2d(input_image, 7, [1,1],
#                         padding='SAME', stride=[1,1],
#                         bn=True, is_training=is_training,
#                         scope='conv_input', bn_decay=bn_decay)
    res_1 = tf_util.max_pool2d(input_image, [2,2],stride=[1,1],
                             padding='SAME', scope='res_1')
    print('res_1',np.shape(res_1))
    #---------------------------------------------------------
    print('input image',np.shape(input_image))
    net = tf_util.conv2d(input_image, 8, [1,1],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    print('conv1',np.shape(net))
    net = tf_util.conv2d(net, 8, [1,1],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    print('conv2',np.shape(net))
    net = tf_util.max_pool2d(net, [2,2],stride=[1,1],
                             padding='SAME', scope='maxpool1')
    print('after pooling1',np.shape(net))
    convtp_1 = tf_util.conv2d_transpose(net,8,[1,1],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_1', bn_decay=bn_decay)
    print('convtp_1',np.shape(convtp_1))
    convtp_1p = tf_util.conv2d_transpose(res_1,16,[2,2],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_1p', bn_decay=bn_decay)
    print('convtp_1p',np.shape(convtp_1p))
    #--------------------------------------------------------------
    
    res_2 = tf_util.max_pool2d(net, [2,3],stride=[1,1],
                             padding='SAME', scope='res_2')
    print('res_2',np.shape(res_2))
    net = tf.concat([net,res_1],axis=3)
    print('hebing_1',np.shape(net))
    net = tf_util.conv2d(net, 8, [2,2],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    print('conv3',np.shape(net))
    net = tf_util.conv2d(net, 16, [2,2],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    print('conv4',np.shape(net))
    net = tf_util.max_pool2d(net, [2,3],stride=[1,1],
                             padding='SAME', scope='maxpool2')
    print('after pooling2',np.shape(net))
    convtp_2 = tf_util.conv2d_transpose(net,16,[2,3],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_2', bn_decay=bn_decay)
    print('convtp_2',np.shape(convtp_2))
    convtp_2p = tf_util.conv2d_transpose(res_2,16,[2,3],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_2p', bn_decay=bn_decay)
    print('convtp_2p',np.shape(convtp_2p))
    #-----------------------------------------------------------
    
    res_3 = tf_util.max_pool2d(net, [2,2],stride=[1,1],
                             padding='SAME', scope='res_3')
    print('res_3',np.shape(res_3))
    net = tf.concat([net,res_2],axis=3)
    print('hebing_2',np.shape(net))
    net = tf_util.conv2d(net, 16, [2,2],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    print('conv5',np.shape(net))
    net = tf_util.conv2d(net, 16, [2,2],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    print('conv6',np.shape(net))
    net = tf_util.max_pool2d(net, [3,3],stride=[1,1],
                             padding='SAME', scope='maxpool3')
    print('after pooling3',np.shape(net))
    convtp_3 = tf_util.conv2d_transpose(net,16,[4,6],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_3', bn_decay=bn_decay)
    print('convtp_3',np.shape(convtp_3))
    convtp_3p = tf_util.conv2d_transpose(res_3,32,[4,6],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_3p', bn_decay=bn_decay)
    print('convtp_3p',np.shape(convtp_3p))
    #----------------------------------------------------------
    
    res_4 = tf_util.max_pool2d(net, [2,2],stride=[1,1],
                             padding='SAME', scope='res_4')
    print('res_4',np.shape(res_4))
    
    net = tf.concat([net,res_3],axis=3)
    print('hebing_3',np.shape(net))
    net = tf_util.conv2d(net, 16, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    print('conv7',np.shape(net))
    net = tf_util.conv2d(net, 32, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    print('conv8',np.shape(net))
    net = tf_util.max_pool2d(net, [3,3],stride=[1,1],
                             padding='SAME', scope='maxpool4')
    print('after pooling4',np.shape(net))
    dp_net4 = tf_util.dropout(net, keep_prob=0.8, is_training=is_training, scope='dp_net4')
    convtp_4 = tf_util.conv2d_transpose(dp_net4,32,[8,12],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_4', bn_decay=bn_decay)
    print('convtp_4',np.shape(convtp_4))
    convtp_4p = tf_util.conv2d_transpose(res_4,64,[8,24],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_4p', bn_decay=bn_decay)
    print('convtp_4p',np.shape(convtp_4p))
    
    #-----------------------------------------------------------
    net = tf.concat([net,res_4],axis=3)
    print('hebing_4',np.shape(net))
    net = tf_util.conv2d(net, 64, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    print('conv9',np.shape(net))
    net = tf_util.conv2d(net, 64, [3,3],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv10', bn_decay=bn_decay)
    print('conv10',np.shape(net))
    net = tf_util.conv2d(net, 64, [4,6],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv10p', bn_decay=bn_decay)
    print('conv10p',np.shape(net))
    net = tf_util.max_pool2d(net, [6,9],stride=[1,1],
                             padding='SAME', scope='maxpool5')
    print('after pooling5',np.shape(net))
    dp_net5 = tf_util.dropout(net, keep_prob=0.8, is_training=is_training, scope='dp_net5')
    convtp_5 = tf_util.conv2d_transpose(dp_net5,64,[16,54],
                                       padding='SAME', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='convtp_5', bn_decay=bn_decay)
    print('convtp_5',np.shape(convtp_5))
    #----------------------------------------------------------
    net = tf.concat([convtp_5,convtp_4,convtp_3,convtp_2,convtp_1,convtp_1p,convtp_2p,convtp_3p,convtp_4p],axis=3)
    print('tranpose_hebing',np.shape(net))
    net = tf_util.dropout(net, keep_prob=0.8, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv11', bn_decay=bn_decay)
    print('conv11',np.shape(net))
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp2')
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='SAME', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv12', bn_decay=bn_decay)
    print('conv12',np.shape(net))
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp3')
    net = tf_util.conv2d(net, num_classes, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv13')
    print('conv13',np.shape(net))
    
    #-------------------------dev_code----------------------------------
    
    #---------------------------------------------------------
    net = tf.reshape(net,[batch_size,num_point,num_classes])
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
        a = tf.placeholder(tf.float32, shape=(32,48,216,7))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,48,216,7)})
            print(time.time() - start)
