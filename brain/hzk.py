#-*-coding:utf-8-*-
import socket
import numpy as np

def hzk():
    LABELS = 100  # 8363

    f=open("HZK16", 'rb')
    try:
        allhz = f.read(32 * 203)
        allhz=f.read(32*LABELS)#203-113 0-9
        #allhz=f.read(32*LABELS)
    finally:
        f.close()
    data=np.frombuffer(allhz,dtype=np.uint16)
    cnt=int(data.size/16)
    data=data.reshape(cnt,16)
    imgs=[]
    for i in range(cnt):
        img=[]
        for j in range(16):
            d=socket.ntohs(data[i][j])
            for k in range(16):
                flag = (d & (0x8000 >> (15 - k)))>>k
                if flag!=0:
                    img.append([(j-7),(8-k),1])
        imgs.append(img)
    return imgs

def hz2img_org(hz):#16x16
    img=np.array([[0]*16]*16)
    for i in range(len(hz)):
        img[hz[i][0]+7,hz[i][1]+7]=1
    return img

def hz2img(hz):#28x28
    img=np.array([[0]*28]*28)
    for i in range(len(hz)):
        img[(hz[i][0]+7)+6,(hz[i][1]+7)+6]=1
    return img

def hz2img28(hz):#28x28
    img=np.array([[0]*28]*28)
    for i in range(len(hz)):
        img[(hz[i][0]+7)*28//16,(hz[i][1]+7)*28//16]=1
    return img
