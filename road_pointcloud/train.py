import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import points_to_2d
import get_points_gt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
from model import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--image_height', type=int, default=48, help='image_height [default: 64]')
parser.add_argument('--image_width', type=int, default=216, help='image_width [default: 360]')
parser.add_argument('--image_channel', type=int, default=7, help='image_channel [default: 7]')
#parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()


IMAGE_H = FLAGS.image_height
IMAGE_W = FLAGS.image_width
IMAGE_C = FLAGS.image_channel


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = IMAGE_H*IMAGE_W
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp models/model.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#MAX_NUM_POINT = 4096
NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
#
#ALL_FILES = provider.getDataFiles('indoor3d_sem_seg_hdf5_data/all_files.txt')
#room_filelist = [line.rstrip() for line in open('indoor3d_sem_seg_hdf5_data/room_filelist.txt')]
train_datafiles = provider.getDataFiles('data/traindata/train_datafiles.txt')
train_labelfiles = provider.getDataFiles('data/traindata/train_labelfiles.txt')
#test_datafiles = provider.getDataFiles('data/traindata/test_datafiles.txt')
#test_labelfiles = provider.getDataFiles('data/traindata/test_labelfiles.txt')
# Load ALL data
#data_batch_list = []
#label_batch_list = []
#for h5_filename in ALL_FILES:
#    data_batch, label_batch = provider.loadDataFile(h5_filename)
#    #print('h5file_data',data_batch.shape)
#    data_batch_list.append(data_batch)
#    label_batch_list.append(label_batch)
#data_batches = np.concatenate(data_batch_list, 0)
#label_batches = np.concatenate(label_batch_list, 0)
#print('data_batches',data_batches.shape)
#print('label_batches',label_batches.shape)
#
#test_area = 'Area_'+str(FLAGS.test_area)
#train_idxs = []
#test_idxs = []
#for i,room_name in enumerate(room_filelist):
#    if test_area in room_name:
#        test_idxs.append(i)
#    else:
#        train_idxs.append(i)
train_data=[]
train_label=[]
for i in range(len(train_datafiles)):
    data,label = provider.load_npdata(train_datafiles[i],train_labelfiles[i])
    train_data.append(data)
    train_label.append(label)
train_data=np.concatenate(train_data,0)
train_label=np.concatenate(train_label,0)
print('input_data',train_data.shape, 'input_label',train_label.shape)




#------------------------------------processing multi-channel maps-----------------
process_data=np.zeros((train_data.shape[0]*2,IMAGE_H,IMAGE_W,IMAGE_C))
process_label=np.zeros((train_data.shape[0]*2,NUM_POINT))
#to save super images
for id_vcc in range(train_data.shape[0]):
    data=train_data[id_vcc]
    label=train_label[id_vcc]
    
    #append super_image
    x, y, z ,intense=data[:,0],data[:,1],data[:,2],data[:,3]
    st=time.clock()
    m,n,d,r=points_to_2d.xyz_to_ra(points_xyz=data[:,:3],h_sy=82.08,h_angle=0.38,v_sy=20.16,v_angle=0.42)
    
    super_image=points_to_2d.make_2d_image(m=m,d=d,n=n,r=r,x=x,y=y,z=z,intense=intense,
                                       label=label,angle_view=216,lines_view=48)
    print(time.clock()-st)
    t_super_image=np.copy(super_image)
    for i in range(164):
        t_super_image[:,i,:]=super_image[:,163-i,:]
    t_gt = super_image[:,:,4]
    super_image = np.delete(super_image,4,2)
    process_data[id_vcc*2]=super_image[:,:216,:7]
    process_label[id_vcc*2]=t_gt.reshape((-1))
    
    #append super_image_tranpose hirozental
          
    t_gt = t_super_image[:,:,4]
    t_super_image = np.delete(t_super_image,4,2)
    process_data[id_vcc*2+1]=t_super_image[:,:216,:7]
    process_label[id_vcc*2+1]=t_gt.reshape((-1))
    
    print(id_vcc,'processing trainning data')
#--------------------------------------divide test and train data-----------
test_data=process_data[:114]
test_data=test_data[::2]
test_label=process_label[:114]
test_label=test_label[::2]
test_label=test_label.astype(np.int32)
train_data=process_data[114:]
train_data=train_data[::2]
train_label=process_label[114:]
train_label=train_label[::2]
train_label=train_label.astype(np.int32)
#test_data=[]
#test_label=[]
#for file in range(len(train_datafiles)):
#    data,label = provider.load_npdata(test_datafiles[i],test_labelfiles[i])
#    test_data.append(data)
#    test_label.append(label)
#test_data=np.concatenate(test_data,0)
#test_label=np.concatenate(test_label,0)
#test_label=test_label.reshape((-1,NUM_POINT)).astype(np.int32)
print('____________________________start____________________________________________')
print('train_data',train_data.shape, 'train_label',train_label.shape)
print('test_data',test_data.shape,'test_label', test_label.shape)




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, IMAGE_H,IMAGE_W,IMAGE_C)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print('pointclouds_pl',pointclouds_pl.shape)
            print('labels_pl',labels_pl.shape)
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            
            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
           # print('labels_pl',labels_pl.shape)
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data, train_label)
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 5 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        jitter_data = provider.jitter_point_cloud(current_data[start_idx:end_idx, :, :,:])
        feed_dict = {ops['pointclouds_pl']: jitter_data,
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0.00000001 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_recall=0
    
    log_string('----')
    current_data = test_data
    current_label = np.squeeze(test_label)
    eval_time=0
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :,:],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        eval_start=time.clock()
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        eval_end=time.clock()
        eval_time+=eval_end-eval_start
        
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        total_recall += np.sum((pred_val == current_label[start_idx:end_idx])*(pred_val==1))
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
    log_string('eval time: %f' % (eval_time/file_size))       
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval recall: %f'% (total_recall / float(np.sum(current_label==1))))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
