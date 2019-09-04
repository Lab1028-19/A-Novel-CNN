# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.image as mpimg # mpimg 用于读取图片


#read folder list
def list_filesname(folder_dir):
    files_name=[]
    for files in os.listdir(folder_dir):
        files_name.append(os.path.join(folder_dir,files))
    return files_name

def read_calibration(files_name,cam_n):#解析标定的投影矩阵
    with open(files_name, 'r') as f:  
        data = f.readlines()  #read txt by lines
        tm_P=np.array(data[cam_n].split(' ')[1:],dtype=float).reshape((3,4))
        tm_R0_rect=np.array(data[4].split(' ')[1:],dtype=float).reshape((3,3))
        tm_R0_rect=np.hstack((tm_R0_rect,np.array([0,0,0]).reshape((3,1))))
        tm_R0_rect=np.vstack((tm_R0_rect,np.array([0,0,0,1]).reshape((1,4))))
        tm_Tr_velo_to_cam=np.array(data[5].split(' ')[1:],dtype=float).reshape((3,4))
        tm_Tr_velo_to_cam=np.vstack((tm_Tr_velo_to_cam,np.array([0,0,0,1]).reshape((1,4))))
    return tm_P,tm_R0_rect,tm_Tr_velo_to_cam


def read_img(files_name):
    img = mpimg.imread(files_name)
    size=img.shape
    return img,size


def read_velxyz(files_name):
    data = np.fromfile(files_name,dtype=np.float32)
    data = data.reshape([-1,4])
    points = data
    pl = data[data[:,0]>0,:]
    x, y, z, intense= data[:,0], data[:,1], data[:,2], data[:,3]
    return x,y,z,intense,points

def get_points_gt(id_vcc,calib_files,gt_files,vel_files):
    #read the calib
    P,R0,Tr=read_calibration(calib_files[id_vcc],2)
    #read the image
    gt_img,img_size=read_img(gt_files[id_vcc])
    gt=gt_img[:,:,2]-(gt_img[:,:,0]-1)
    #计算联合标定后的点云矩阵pj
    p_x,p_y,p_z,pointcloud=read_velxyz(vel_files[id_vcc])
    pc=pointcloud.transpose()
    #pj为经过联合标定过后投影到图片上的x,y,label,及对应的点云x,y,z,intense
    pj=np.dot(np.dot(np.dot(P,R0),Tr),pc)
    pj[0,:] /= pj[2,:]
    pj[1,:] /= pj[2,:]
    pj=np.round(pj)
    ####vstack()在列上合并
    pj=np.vstack((pj,pc))
    pj=pj[:,pj[0,:]>0]#x>0
    pj=pj[:,pj[0,:]<img_size[1]]#x<img_size[1]
    pj=pj[:,pj[1,:]>0]#y>0
    pj=pj[:,pj[1,:]<img_size[0]]#y<img_size[0]
    for i in range(pj.shape[1]):
        pj[2,i] = 1 if gt[int(pj[1,i]),int(pj[0,i])]>0 else 0
    return pj[[3,4,5,6,2],:].transpose()

def get_label_location(labelfile):
    file = open(labelfile,"r")  
    list_arr = file.readlines()  
    l = len(list_arr)
    label_location=[]
    type_dic={'pedestrian':1,'vehicle':2,'cyclist':3,'dontCare':4}
    for item in list_arr:   
        temp = item.split(' ')
        temp_label = np.append(type_dic[temp[0]],np.float32(temp[1:]))
        label_location.append(temp_label)
    label_location=np.array(label_location,dtype=np.float32)
    return label_location

def get_points_label(points,label_location):
    label=np.zeros((len(points),2),dtype=np.float32)
    for p in range(len(label_location)):
        x_center=label_location[p][1]
        y_center=label_location[p][2]
        z_center=label_location[p][3]
        length=label_location[p][4]
        width=label_location[p][5]
        height=label_location[p][6]
        yaw=2*np.pi-label_location[p][7]
        for i in range(len(points)):
            x1=points[i,0]
            y1=points[i,1]
            x = (x1-x_center)*np.cos(yaw) - (y1-y_center)*np.sin(yaw) + x_center
            y = (x1-x_center)*np.sin(yaw) + (y1-y_center)*np.cos(yaw) + y_center
            z = points[i,2]
            if abs(x-x_center)<=length/2 and abs(y-y_center)<=width/2 and abs(z-z_center)<=height/2 :
                label[i,1]=p+1
                label[i,0]=label_location[p][0]
    return label
'''
def get_points_label(points,label_location):
    label=np.zeros(len(points))
    for i in range(len(points)):
        for p in range(len(label_location)):
            x_center=label_location[p][1]
            y_center=label_location[p][2]
            z_center=label_location[p][3]
            length=label_location[p][4]
            width=label_location[p][5]
            height=label_location[p][6]
            yaw=2*np.pi-label_location[p][7]
            x1=points[i,0]
            y1=points[i,1]
            x = (x1-x_center)*np.cos(yaw) - (y1-y_center)*np.sin(yaw) + x_center
            y = (x1-x_center)*np.sin(yaw) + (y1-y_center)*np.cos(yaw) + y_center
            z = points[i,2]
            if abs(x-x_center)<=length/2 and abs(y-y_center)<=width/2 and abs(z-z_center)<=height/2 :
                label[i]=label_location[p][0]
    return label
'''
def save_points_gt(id_vcc,out_dir,vel_files):
    #vel_gt : x,y,z,label
    vel_gt=get_points_gt(id_vcc)
    out_filename=out_dir+vel_files[id_vcc].split('/vel/')[1].split('.')[0]+".npy"
    np.save(out_filename, vel_gt)
    print('saved:'+out_filename)
    
def read_poinst_gt(out_filename):
    data=np.load(out_filename).reshape([-1,5])
    return data[:,:-1],data[:,-1]