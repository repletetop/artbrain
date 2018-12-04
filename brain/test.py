#-*-coding:utf-8-*-

import numpy as np
import socket
from BrainLayer import BrainLayer
from matplotlib import pyplot as plt
from FuncArea import FCompare2D



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

def hz2img(hz):
    img=np.array([[0]*16]*16)
    for i in range(len(hz)):
        img[hz[i][0]+7,hz[i][1]+7]=1
    return img

def rotate():
    allhz=hzk()
    img=hz2img(allhz[0])
    la = BrainLayer(16, 16)
    lb= BrainLayer(16,16)
    la.rotate(lb,45,7,7)
    lc= BrainLayer(16,16)
    la.rotate(lc,90,7,7)

    la.input(img, 0, 0)
    #m=lb.getMat()
    #la.input(m)#you leiji wucha
    m=lb.getMat()
    plt.imshow(m)
    plt.show()


if __name__ == "__main__":
    allhz=hzk()
    i0=hz2img(allhz[0])
    i1=hz2img(allhz[1])
    i2=hz2img(allhz[2])
    i3=hz2img(allhz[3])
    i4=hz2img(allhz[4])
    i5=hz2img(allhz[5])
    i6=hz2img(allhz[6])
    i7=hz2img(allhz[7])
    i8=hz2img(allhz[8])
    i9=hz2img(allhz[9])

    c2d=FCompare2D()
    c2d.la.input(i0)
    r=c2d.like(i0,i1)
    plt.imshow(r)
    plt.show()
    r=c2d.like(i2,i1)
    plt.imshow(r)
    plt.show()
    r=c2d.like(i1,i7)
    plt.imshow(r)
    plt.show()
    print(r)

