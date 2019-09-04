import numpy as np
import matplotlib.pyplot as plt
def xyz_to_ra(points_xyz,v_sy,v_angle,h_sy,h_angle):
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
    mid=np.ceil(np.median(xh))
    if np.size(xh)>64:
        a_v[a_v>=mid+31]=mid+31
        a_v[a_v<=mid-32]=mid-32
    
    #转换到0递增
    a_h=a_h-a_h.min()
    a_h=a_h.astype(int)
    a_h[a_h>=h_sy/h_angle]=h_sy/h_angle-1
    a_v=a_v-a_v.min()
    a_v=a_v.astype(int)
    a_v[a_v>=v_sy/v_angle]=v_sy/v_angle-1
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
    instance_image=np.zeros((lines_view,angle_view))
    for i in range(np.size(m)):
        point_image[m[i],n[i]]+=1
        range_image[m[i],n[i]]+=np.log2(r[i])
        intense_image[m[i],n[i]]+=intense[i]
        x_image[m[i],n[i]]+=x[i]
        y_image[m[i],n[i]]+=y[i]
        height_image[m[i],n[i]]+=z[i]
        dis_image[m[i],n[i]]+=np.log2(d[i])
        gt_image[m[i],n[i]]=label[i,0]
        instance_image[m[i],n[i]]=label[i,1]
    num_p=np.copy(point_image)
    num_p[num_p==0]=1
    intense_image/=num_p
    range_image/=num_p
    y_image/=num_p
    x_image/=num_p
    height_image/=num_p
    dis_image/=num_p
    super_image=np.zeros((64,720,9))
    super_image[:,:,0]=point_image
    super_image[:,:,1]=range_image
    super_image[:,:,2]=dis_image
    super_image[:,:,3]=intense_image
    super_image[:,:,4]=gt_image
    super_image[:,:,5]=x_image
    super_image[:,:,6]=y_image
    super_image[:,:,7]=height_image
    super_image[:,:,8]=instance_image
    return super_image