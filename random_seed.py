import cv2
import numpy as np
import json
import base64
import os
from labelme import utils
import matplotlib.pyplot as plt
import random
import math
#import torch


global image_end,json_end,mask_end

def add_obj(background, img, mask, x, y,json1,json11):
    '''
    Arguments:
    background - background image in CV2 RGB format
    img - image of object in CV2 RGB format
    mask - mask of object in CV2 RGB format
    x, y - coordinates of the center of the object image
    0 < x < width of background
    0 < y < height of background
    
    Function returns background with added object in CV2 RGB format
    
    CV2 RGB format is a numpy array with dimensions width x height x 3
    '''
    
    bg = background.copy()
    
    h_bg, w_bg = bg.shape[0], bg.shape[1]
    
    h, w = img.shape[0], img.shape[1]
    
    # Calculating coordinates of the top left corner of the object image
    x = x - int(w/2)
    y = y - int(h/2)    
    
    mask_boolean = mask[:,:] != 0
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
    
    if x >= 0 and y >= 0:
    
        h_part = h - max(0, y+h-h_bg) # h_part - part of the image which overlaps background along y-axis
        w_part = w - max(0, x+w-w_bg) # w_part - part of the image which overlaps background along x-axis

        bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
        
    elif x < 0 and y < 0:
        
        h_part = h + y
        w_part = w + x
        
        bg[0:0+h_part, 0:0+w_part, :] = bg[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_boolean)[h-h_part:h, w-w_part:w, :]
        
    elif x < 0 and y >= 0:
        
        h_part = h - max(0, y+h-h_bg)
        w_part = w + x
        
        bg[y:y+h_part, 0:0+w_part, :] = bg[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[0:h_part, w-w_part:w, :] + (img * mask_rgb_boolean)[0:h_part, w-w_part:w, :]
        
    elif x >= 0 and y < 0:
        
        h_part = h + y
        w_part = w - max(0, x+w-w_bg)
        
        bg[0:0+h_part, x:x+w_part, :] = bg[0:0+h_part, x:x+w_part, :] * ~mask_rgb_boolean[h-h_part:h, 0:w_part, :] + (img * mask_rgb_boolean)[h-h_part:h, 0:w_part, :]
    #cv2.imwrite(temp, bg)
    json_bg = json_file(json11,json1,x,y)
    return bg,json_bg

def clamp_number(num,a,b):
    return max(min(num, max(a, b)), min(a, b))

def json_file(data,data1,global_x0,global_y0):
    """
    json1是种子列表标签，json11是背景列表标签，global_x0,global_y0是种子左上角坐标
    """
    
    version = data['version']
    flags = data['flags']
    shapes = data['shapes']
    imageHeight = data['imageHeight']
    imageWidth = data['imageWidth']
    imageData = data['imageData']
    imagePath = data['imagePath']
    
    json_content = []
    obj1=data1['shapes']
    label1=obj1[0]['label']
    points1=obj1[0]['points']
    for num, info_point in enumerate(points1):
        info_point[0]=info_point[0]+global_x0
        info_point[1]=info_point[1]+global_y0
        info_point[0]=clamp_number(info_point[0],0,imageWidth)
        info_point[1]=clamp_number(info_point[1],0,imageHeight)
        
        
    
                
    item_dict = {
        "version": version,
        "flags": flags,
        "shapes": shapes,
        "imageData": imageData,
        "imagePath": imagePath,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }
    item_dict["shapes"].append(obj1[0])
    #print(item_dict["shapes"])
    json_content.append(item_dict)
    json_list=json_content[0]
    return json_list

    


    
    
    
def imagedata_list(img,json_list):
    image='cache.jpg'
    cv2.imwrite(image, img)
    with open(image,'rb') as image_file:#读取图片文件
        image_text_data = image_file.read()
        image_text_bytes = base64.b64encode(image_text_data)#用base64API编码
        imageData_end = image_text_bytes.decode('utf-8')
    """
    with open(json_file,'r') as load_f:
        data=json.load(load_f)
        imageData = data['imageData']
        imageData=imageData_end 
    with open(json_file,'w') as dump_f:
        json.dump(data, dump_f,indent=4)
    """
    json_list['imageData']=imageData_end
    return json_list


def mask_res(json_list):
    #data = json.load(json_file)
    data = json_list
    img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符可以得到原图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
    lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])   # data['shapes']是json文件中记录着标注的位置及label等信息的字段
    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw.draw_label(lbl, img, captions)
    mask = []
    class_id = []
    class_name = [key for key in lbl_names.keys()]
    img_mask_new = np.zeros(img_gray.shape).astype(np.uint8)
    for i in range(1, len(lbl_names)):  # 跳过第一个class（默认为背景）
        img_mask_new[lbl == i] = img_gray[lbl == i]
    #获取标记区域
    retval, im_at_fixed = cv2.threshold(img_mask_new, 0, 255, cv2.THRESH_BINARY)
    return im_at_fixed
    """
    filename = os.path.splitext(temp)[0]
    mask_result = filename+'.png'
    cv2.imwrite(mask_result, im_at_fixed)
    """

        
        


def edge_blur(img,mask):
    mask= Edge_Extract(mask)
    dst_TELEA = cv2.inpaint(img,mask,8,cv2.INPAINT_TELEA)
    return dst_TELEA

def Edge_Extract(image):
    img = cv2.imread(image,0)
    mask=cv2.Canny(img,30,100)
    return mask

def read_json(json_file):
    json_content=[]
    with open(json_file,'r') as load_f:
        data=json.load(load_f)
        version = data['version']
        flags = data['flags']
        shapes = data['shapes']
        imagePath = data['imagePath']
        imageData = data['imageData']
        imageHeight = data['imageHeight']
        imageWidth = data['imageWidth']
        
        item_dict = {
            "version": version,
            "flags": flags,
            "shapes": shapes,
            "imageData": imageData,
            "imagePath": imagePath ,
            "imageHeight": imageHeight,
            "imageWidth": imageWidth
        }
        json_content.append(item_dict)
        json_list=json_content[0]
        return json_list
    
    
def json_scale(json_list,img,scalex,scaley,w,h):
    list01 = json_list['shapes'][0]['points']
    num=len(list01)
    for i in range(num):
        json_list['shapes'][0]['points'][i][0]=list01[i][0]*scalex
        json_list['shapes'][0]['points'][i][1]=list01[i][1]*scaley
    json_list['imageHeight']=h
    json_list['imageWidth']=w
    json_list = imagedata_list(img,json_list)
    return json_list

def json_affine(json_son,M,img):
    list01 = json_son['shapes'][0]['points']
    num=len(list01)
    for i in range(num):
        json_son['shapes'][0]['points'][i][0]=list01[i][0]*M[0][0]+list01[i][1]*M[0][1]+M[0][2]
        json_son['shapes'][0]['points'][i][1]=list01[i][0]*M[1][0]+list01[i][1]*M[1][1]+M[1][2]
    h, w = img.shape[0], img.shape[1]
    json_son['imageHeight']=h
    json_son['imageWidth']=w
    json_son = imagedata_list(img,json_son)
    return json_son

def json_rotation_affine(json_son,center,angle,scale,img):
    list01 = json_son['shapes'][0]['points']
    #print(list01)
    #print(center)
    #json1['shapes'][0]['points']=[]
    #json2['shapes'][0]['points']=[]
    h, w = img.shape[0], img.shape[1]
    angle=math.pi*(-angle/180)
    cos=math.cos(angle)
    sin=math.sin(angle)
    num=len(list01)
    list02=[[0,0] for i in range(0,num)]
    list03=[[0,0] for i in range(0,num)]
    list04=[[0,0] for i in range(0,num)]
    for i in range(0,num):
        #print(i)
        #print(list01[i])
        #print(center)
        list02[i][0]=list01[i][0]-center[0]
        list02[i][1]=list01[i][1]-center[1]
        #print(list02[i])
        list03[i][0]=list02[i][0]*cos*scale-list02[i][1]*sin*scale
        list03[i][1]=list02[i][0]*sin*scale+list02[i][1]*cos*scale
        #print(list03[i])
        #print(w,h)
        list04[i][0]=list03[i][0]+w/2
        list04[i][1]=list03[i][1]+h/2
        #print(list04[i])
        
    
    #print(list02)
    #print(list04)
    json_son['shapes'][0]['points']=list04
    #print(list01)
    #print(list02)
    #print(list03)
    #print(list04)
    #print(json1['shapes'][0]['points'])
    #print(json2['shapes'][0]['points'])
    #print(json_son['shapes'][0]['points'])
    json_son['imageHeight']=h
    json_son['imageWidth']=w
    json_son = imagedata_list(img,json_son)
    return json_son

def RotationMatrix2D(center,M,w,h):
    M01=np.float32(-M[0,0]*center[0]-M[0,1]*center[1]+w/2)
    M02=np.float32(-M[0,0]*center[1]+M[0,1]*center[0]+h/2)
    #angle *= math.pi/180
    """
    alpha=np.float32(scale*math.cos(angle))
    beta=np.float32(scale*math.sin(angle))
    M01=np.float32((1-alpha)*center[0]-beta*center[1])
    M02=np.float32((1-alpha)*center[1]+beta*center[0])
    M1=np.float32([[alpha,beta,M01],[-beta,alpha,M02]])
    """
    M1=np.float32([[M[0,0],M[0,1],M01],[M[1,0],M[1,1],M02]])
    return M1

def randfloat(num, l, h):
    if l > h:
        return None
    else:
        a = h - l
        b = h - a
        out = np.random.rand(num) * a + b
        #out = np.array(out)
        out1=out[0]
        return out1
    

def bboxes_diou(boxes1,boxes2):
    '''
    cal DIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    #cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    #cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    #cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])

    #cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
                 np.square(boxes1_center[...,1]-boxes2_center[...,1])

    #cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious


def get_rec(json_son,x_left,y_left,w_bg,h_bg) :
    #print('json_son:',json_son['shapes'][0]['points'])
    corner = [x_left,y_left]
    json_point = json_son['shapes'][0]['points']
    #print('(x_left,y_left):',x_left,y_left)
    point_=[]
    for num, info_point in enumerate(json_point):
        point_1=[]
        for e in range(len(info_point)):
            point_1.append(info_point[e]+corner[e])
        point_.append(point_1)
    #print('point:',point_)
    x_max = max([i[0] for i in point_])
    x_min = min([i[0] for i in point_])
    y_max = max([i[1] for i in point_])
    y_min = min([i[1] for i in point_])
    if x_min<0:
        x_min=0
    if x_max>=w_bg:
        x_max=w_bg-1
    if y_min<0:
        y_min=0
    if y_max>=h_bg:
        y_max=h_bg-1
    return [x_min,y_min,x_max,y_max]

def mirror_matrix(mirror,w_son,h_son):
    if mirror == 0:
        return np.float32([[1,0,0],[0,1,0]])
    elif mirror == 1:
        return np.float32([[-1,0,w_son],[0,1,0]])
    elif mirror == 2:
        return np.float32([[1,0,0],[0,-1,h_son]])
    else:
        return np.float32([[-1,0,w_son],[0,-1,h_son]])
        

def rand_circul(xcenter,ycenter,r1,r2,w_bg,h_bg):#在圆环内采样
    x = random.uniform(r1, r2)
    y = random.uniform(math.sqrt(x*x-r1*r1), math.sqrt(r2*r2-x*x))
    if random.uniform(-1,1) > 0:
        x=round(xcenter+x)
    else:
        x=round(xcenter-x)
    if x<50:
        x=50
    elif x>=w_bg-50:
        x=w_bg-50
    if random.uniform(-1,1) > 0:
        y=round(ycenter+y)
    else:
        y=round(ycenter-y)
    if y<50:
        y=50
    elif y>=h_bg-50:
        y=h_bg-50
    return [x,y]

background1 ='son/gray_bg.jpg'
#background1 ='son/yellow_bg.jpg'
#image='2-60_03_1.jpg'
#mask1='2-60_03_1.png'
#json1='2-60_03_1.json'
json11= 'son/gray_bg.json'
#json11= 'son/yellow_bg.json'
son_pathname = 'son/gray'
#son_pathname = 'son/yellow'
count = 0
d=0
for filename in os.listdir(son_pathname):
    name,extension = os.path.splitext(filename)
    if extension == '.jpg':
        count=count+1
image_list=['0']*count
json_list = ['0']*count
for filename in os.listdir(son_pathname):
    name,extension = os.path.splitext(filename)
    if extension == '.jpg':
        #print(d)
        image_list[d] =son_pathname+'/'+filename
        json_list[d] = son_pathname+'/'+name+'.json'
        d=d+1
    

    
background= cv2.imread(background1)
#background=cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
#img = cv2.imread(image)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
#mask = cv2.imread(mask1)
#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

"""
写个函数对img，mask，和json1做放缩，镜像和旋转的变换

"""
num=100#生成图片数
p1=0.8
p2=0.8
p3=0.8
N=50#网格
#一张图片的随机粒子数
for i in range(418,num+400):
    head=0#代表下一次采样时参考的框的json_rec_list下标
    image_end = background.copy()
    json_end=read_json(json11)
    n=random.randint(50,80)#粒子数
    json_rec_list=[]
    h_bg, w_bg = background.shape[0], background.shape[1]
    w_s=w_bg/float(N)
    h_s=h_bg/float(N)
    json_son_all=[]
    son_rotate_all=[]
    son_mask_all=[]
    for j in range(n):
        s_rand = random.randint(0,count-1)
        son = cv2.imread(image_list[s_rand])
        #son = img.copy()#随机选用粒子
        json_son = read_json(json_list[s_rand])
        #json_son = json_list_01.copy()
        """
        mask1 = mask_res(json_son)
        with open('son1'+str(j+1)+'_1.json','w') as dump_f:
            json.dump(json_son, dump_f,indent=4)
        cv2.imwrite('son1'+str(j+1)+'_1.jpg', son)
        cv2.imwrite('son1'+str(j+1)+'_1.png', mask1)
        """
        #放缩
        p11=random.random()
        if p11<=p1:
            a = randfloat(1,-0.1,0.15)
            b = randfloat(1,-0.1,0.15)
            scalex=0.9+a#粒子长缩放比例随机数
            scaley=0.9+b
            son_scale =  cv2.resize(son, (0, 0), fx=scalex, fy=scaley, interpolation=cv2.INTER_AREA)
            h_son, w_son = son_scale.shape[0], son_scale.shape[1]
            json_son11 = json_scale(json_son,son_scale,scalex,scaley, w_son,h_son)#son对应的json
        else:
            son_scale=son
            h_son, w_son = son_scale.shape[0], son_scale.shape[1]
            json_son11=json_son
        #镜像
        p21=random.random()
        if p21<=p2:
            #产生随机数，1，2，3，1代表水平翻转
            mirror = random.randint(1,3)
            M_flip = mirror_matrix(mirror,w_son,h_son)
            #M_flip = np.float32([[-1,0,w_son],[0,1,0]])
            cos = np.abs(M_flip[0, 0])
            sin = np.abs(M_flip[0, 1])
            new_w = h_son * sin + w_son * cos
            new_h = h_son * cos + w_son * sin
            M_flip[0, 2] += (new_w - w_son) * 0.5
            M_flip[1, 2] += (new_h - h_son) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))

            #son_flip = cv2.warpAffine(src=son_rotation, M=M_flip, dsize=(h_son, w_son), borderValue=(255, 255, 255))
            son_flip = cv2.warpAffine(src=son_scale, M=M_flip, dsize=(w, h), borderValue=(255, 255, 255))
            json_son21 = json_affine(json_son11,M_flip,son_flip)
        else:
            son_flip=son_scale
            json_son21=json_son11
        h_son, w_son = son_flip.shape[0], son_flip.shape[1]
        """
        mask2 = mask_res(json_son)
        with open('son1'+str(j+1)+'_2.json','w') as dump_f:
            json.dump(json_son, dump_f,indent=4)
        cv2.imwrite('son1'+str(j+1)+'_2.jpg', son_flip)
        cv2.imwrite('son1'+str(j+1)+'_2.png', mask2)
        """
        #旋转
        p31=random.random()
        if p31<=p3:
            center = (w_son / 2, h_son / 2)   # 绕图片中心进行旋转
            angle = random.randint(-180, 180)  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
            scale = randfloat(1,0.8,1.2)                        # 固定，将图像缩放为80%
            M = cv2.getRotationMatrix2D(center, angle, scale)# 获得旋转矩阵
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = h_son * sin + w_son * cos
            new_h = h_son * cos + w_son * sin
            M[0, 2] += (new_w - w_son) * 0.5
            M[1, 2] += (new_h - h_son) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))  
            M1=RotationMatrix2D(center,M,w,h)
            son_rotation = cv2.warpAffine(src=son_flip, M=M1, dsize=(w, h), borderValue=(255, 255, 255)) # 进行仿射变换，边界填充为255，即白色，默认为0，即黑色
            json_son31 = json_rotation_affine(json_son21,center,angle,scale,son_rotation)
        else:
            son_rotation=son_flip
            json_son31 =json_son21
        son_mask = mask_res(json_son31)
        json_son=json_son31
        
        json_son_all.append(json_son31)
        son_rotate_all.append(son_rotation)
        son_mask_all.append(son_mask)
        """
        with open('son1'+str(j+1)+'.json','w') as dump_f:
            json.dump(json_son, dump_f,indent=4)
        cv2.imwrite('son1'+str(j+1)+'.jpg', son_rotation)
        cv2.imwrite('son1'+str(j+1)+'.png', son_mask)
        """
        #此时对种子的放缩、镜像、旋转处理已完成，此时的信息存储在son_rotation,json_son,son_mask
    index=0
    grid = [[[-1] for y_ in range(N)]for _ in range(N)]#二维数组一定要这样创建，否则无法部分赋值
    #print(grid)
    while index<n and head<=index:#代表粒子列表下标
        #用dIOU阈值，先新建一个列表储存所有框，再判断这个框的位置信息，跟所有框比较
        if index >=1:
            i_1=0
            while i_1<=20:
                if index>=n or head>=index:
                    break
                json_son=json_son_all[index]#index表示当前要撒的粒子
                son_rotation=son_rotate_all[index]
                h_son, w_son = son_rotation.shape[0], son_rotation.shape[1]
                son_mask=son_mask_all[index]
                w1=json_rec_list[head][2]-json_rec_list[head][0]#参考框长宽
                h1=json_rec_list[head][3]-json_rec_list[head][1]
                xcenter_cir=(json_rec_list[head][2]+json_rec_list[head][0])/2
                ycenter_cir=(json_rec_list[head][3]+json_rec_list[head][1])/2#圆心
                rec=get_rec(json_son,0,0,w_bg,h_bg)
                w2=rec[2]-rec[0]
                h2=rec[3]-rec[1]
                rmin=min((w1+w2)/2,(h1+h2)/2)#圆环半径
                rmax=math.sqrt(w1*w1+h1*h1)+math.sqrt(w2*w2+h2*h2)
                xy=rand_circul(xcenter_cir,ycenter_cir,rmin,rmax,w_bg,h_bg)
                x=xy[0]
                y=xy[1]
                p=math.floor(x/w_s)
                q=math.floor(y/h_s)#格子编号
                print(p,q)
                print(grid[p][q][0])
                while grid[p][q][0]!=-1:
                    #print("寻找无占据格子")
                    xy=rand_circul(xcenter_cir,ycenter_cir,rmin,rmax,w_bg,h_bg)
                    x=xy[0]
                    y=xy[1]
                    p=math.floor(x/w_s)
                    q=math.floor(y/h_s)#格子编号
                    i_1=i_1+1
                    print(i_1)
                    if i_1>20:
                        print("未找到")
                        break
                print("结束寻找")
                #以上位置格子一定没被占据
                x_left = x - int(w_son/2)
                y_left = y - int(h_son/2)
                rec=get_rec(json_son,x_left,y_left,w_bg,h_bg)
                p1=math.floor(rec[0]/w_s)
                q1=math.floor(rec[1]/h_s)#格子编号
                p2=math.floor(rec[2]/w_s)
                q2=math.floor(rec[3]/h_s)
                print("p1,p2,q1,q2 ",p1,p2,q1,q2)
                compare=[]
                for a in range(p1,p2+1):
                    for b in range(q1,q2+1):
                        if grid[a][b][0]!=-1:
                            compare=compare+grid[a][b]
                compare=sorted(list(set(compare)))
                print("compare ",compare)
                if len(compare)==0:
                    json_rec_list.append(rec)
                    image_end,json_end = add_obj(image_end, son_rotation, son_mask,x, y,json_son,json_end)#种子按照指定位置投放并且生成json
                    for a in range(p1,p2+1):
                        for b in range(q1,q2+1):
                            if grid[a][b][0]==-1:
                                grid[a][b][0]=index
                            else:
                                grid[a][b].append(index)
                    #print("grid[p1:p2,q1:q2] ",grid[p1:p2,q1:q2])
                    index=index+1
                    i_1=i_1+1
                else:
                    diou_list=[]
                    for n_ in compare:
                        rec1=np.asarray([json_rec_list[n_]])
                        rec2=np.asarray([rec])
                        diou = bboxes_diou(rec1, rec2)
                        diou_list.append(diou)
                    if max(diou_list)[0]<=-0.5:#iou阈值
                        json_rec_list.append(rec)
                        image_end,json_end = add_obj(image_end, son_rotation, son_mask,x, y,json_son,json_end)#种子按照指定位置投放并且生成json
                        for a in range(p1,p2+1):
                            for b in range(q1,q2+1):
                                if grid[a][b][0]==-1:
                                    grid[a][b][0]=index
                                else:
                                    grid[a][b].append(index)
                        index=index+1
                        i_1=i_1+1
                    else:
                        i_1=i_1+1
            head=head+1
        else:
            json_son=json_son_all[index]
            son_rotation=son_rotate_all[index]
            h_son, w_son = son_rotation.shape[0], son_rotation.shape[1]
            son_mask=son_mask_all[index]
            x=random.randint(400, w_bg-400)
            y=random.randint(300, h_bg-300)#中心点
            x_left = x - int(w_son/2)
            y_left = y - int(h_son/2)
            rec=get_rec(json_son,x_left,y_left,w_bg,h_bg) 
            json_rec_list.append(rec)
            p1=math.floor(rec[0]/w_s)
            q1=math.floor(rec[1]/h_s)#格子编号
            p2=math.floor(rec[2]/w_s)
            q2=math.floor(rec[3]/h_s)
            #print("p1,p2,q1,q2 ",p1,p2,q1,q2)
            for a in range(p1,p2+1):
                for b in range(q1,q2+1):
                    grid[a][b][0]=0#代表ab编号格子被0号框所占据
            print(grid)
            image_end,json_end = add_obj(image_end, son_rotation, son_mask,x, y,json_son,json_end)#种子按照指定位置投放并且生成json
            #print("index ",index)
            index=index+1
        #print(x,y)
            
    
        
    json_end = imagedata_list(image_end,json_end)
    out = 'out/parameter/scale/1/'#修改
    #out = 'out/yellow/'
    imgname='composition_1_'+str(i+1)+'.jpg'#修改
    maskname='composition_1_'+str(i+1)+'.png'#修改
    jsonname='composition_1_'+str(i+1)+'.json'#修改
    json_end['imagePath']=imgname
    mask_end = mask_res(json_end)
    cv2.imwrite(os.path.join(out,maskname), mask_end)
    with open(os.path.join(out,jsonname),'w') as dump_f:
        json.dump(json_end, dump_f,indent=4)
    edge_result=edge_blur(image_end,os.path.join(out,maskname))
    cv2.imwrite(os.path.join(out,imgname), edge_result)
    print("生成粒子数：",n)
    print("已生成图片",i+1)
