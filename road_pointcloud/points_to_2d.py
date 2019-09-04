import numpy as np
import matplotlib.pyplot as plt
import time
def xyz_to_ra(points_xyz,v_sy,v_angle,h_sy,h_angle):
    v_angle/=180*np.pi
    h_angle/=180*np.pi
    h_sy/=180*np.pi
    v_sy/=180*np.pi
    x=points_xyz[:,0]
    y=points_xyz[:,1]
    z=points_xyz[:,2]
    a_h=np.arctan2(x,y+0.00000000000000001)
    #print((a_h.max()-a_h.min())*180/np.pi)
    #水平角按h_angle度划分
    a_h=np.floor(a_h/h_angle)
    d=np.sqrt(np.square(x)+np.square(y)+np.square(z))
    a_v=np.arccos(z/(d+0.00000000000000001))
    #a_v=a_v*180/np.pi
    #垂直角按v_angle划分
    a_v=np.round(a_v/v_angle)
    #64线取中间段的64个区间,多余的两端向中间收拢
    xh=np.unique(a_v)
    #print(len(xh))
    #mid=np.ceil(np.median(xh))
    #if np.size(xh)>64:
    #    a_v[a_v>=mid+31]=mid+31
    #    a_v[a_v<=mid-32]=mid-32
    
    #转换到0递增
    a_h=a_h-a_h.min()
    a_h=a_h.astype(int)
    a_h[a_h>=round(h_sy/h_angle)]=round(h_sy/h_angle)-1
    a_v=a_v-a_v.min()
    a_v=a_v.astype(int)
    a_v[a_v>=round(v_sy/v_angle)]=round(v_sy/v_angle)-1
    #print(a_v.max(),a_h.max())
    r=np.sqrt(np.square(x)+np.square(y)+np.square(z))
    #返回垂直角序，水平角序，xoy距离，矢量距离
    return a_v,a_h,d,r
def make_2d_image(m,n,d,r,x,y,z,intense,label,angle_view,lines_view):
    
    point_image=np.zeros((lines_view,angle_view))
    range_image=np.zeros((lines_view,angle_view))
    intense_image=np.zeros((lines_view,angle_view))
    x_image=np.zeros((lines_view,angle_view))
    y_image=np.zeros((lines_view,angle_view))
    height_image=np.zeros((lines_view,angle_view))
    dis_image=np.zeros((lines_view,angle_view))
    gt_image=np.zeros((lines_view,angle_view))
    #ss=time.clock()
    point_image[m,n]+=1
    range_image[m,n]+=np.log2(r)
    intense_image[m,n]+=intense
    x_image[m,n]+=x
    y_image[m,n]+=y
    height_image[m,n]+=z
    dis_image[m,n]+=np.log2(d)
    gt_image[m,n]=label
    '''
    for i in range(np.size(m)):
        point_image[m[i],n[i]]+=1
        range_image[m[i],n[i]]+=np.log2(r[i])
        intense_image[m[i],n[i]]+=intense[i]
        x_image[m[i],n[i]]+=x[i]
        y_image[m[i],n[i]]+=y[i]
        height_image[m[i],n[i]]+=z[i]
        dis_image[m[i],n[i]]+=np.log2(d[i])
        gt_image[m[i],n[i]]=label[i]
    '''
    #print(time.clock()-ss)
    num_p=np.copy(point_image)
    num_p[num_p==0]=1
    intense_image/=num_p
    #range_image/=num_p
    y_image/=num_p
    x_image/=num_p
    height_image/=num_p
    #dis_image/=num_p
    super_image=np.zeros((lines_view,angle_view,8))
    #super_image[:,:,0]=point_image
    #super_image[:,:,1]=range_image
    #super_image[:,:,2]=dis_image
    super_image[:,:,3]=intense_image
    super_image[:,:,4]=gt_image
    super_image[:,:,5]=x_image
    super_image[:,:,6]=y_image
    super_image[:,:,7]=height_image
    return super_image

def xyz_to_mn(points_xyz,v_sy,v_angle,h_sy,h_angle):
    x=points_xyz[:,0]
    y=points_xyz[:,1]
    z=points_xyz[:,2]
    a_h=np.arctan2(x,y+0.00000000000000001)
    print((a_h.max()-a_h.min())*180/np.pi)
    #水平角按h_angle度划分
    a_h=np.floor(a_h*180/np.pi/h_angle)
    d=np.sqrt(np.square(x)+np.square(y)+np.square(z))
    a_v=np.arccos(z/(d+0.00000000000000001))
    a_v=a_v*180/np.pi
    #垂直角按v_angle划分
    a_v=np.round(a_v/v_angle)
    #64线取中间段的64个区间,多余的两端向中间收拢
    xh=np.unique(a_v)
    print(len(xh))
    
    #转换到0递增
    a_h=a_h-a_h.min()
    a_h=a_h.astype(int)
    a_h[a_h>=h_sy/h_angle]=h_sy/h_angle-1
    a_v=a_v-a_v.min()
    a_v=a_v.astype(int)
    a_v[a_v>=v_sy/v_angle]=v_sy/v_angle-1
    r=np.sqrt(np.square(x)+np.square(y)+np.square(z))
    #返回垂直角序，水平角序，xoy距离，矢量距离
    return a_v,a_h
def mn_to_img(m,n,m_img):
    output_img=np.zeros((m.max()+1,n.max()+1,7))
    for i in range(len(m)):
        output_img[m[i],n[i],0]=m_img[i,0]
        output_img[m[i],n[i],1]=m_img[i,1]
        output_img[m[i],n[i],2]=m_img[i,2]
        output_img[m[i],n[i],3]=m_img[i,3]
        output_img[m[i],n[i],4]=m_img[i,5]
        output_img[m[i],n[i],5]=m_img[i,6]
        output_img[m[i],n[i],6]=m_img[i,7]
    return output_img