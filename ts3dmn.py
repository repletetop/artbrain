#coding=utf-8
import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import tensorflow.examples.tutorials.mnist.input_data as input_data
import socket
import math

from collections import Counter
from brain.neuron import  *
from brain.FuncArea import *

LABELS=100#8363
HZ=16*16#16*16


def hzk():
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


def showhz_img(hz):
    img=np.array([[0]*16]*16)
    for i in range(len(hz)):
        img[hz[i][0],hz[i][1]]=1
    plt.imshow(img)

def showhz(hz):
    plt.xlim(-20,20)
    plt.ylim(-20,20)
    for i in range(len(hz)):
        plt.scatter(hz[i][1],hz[i][0])

def showallhz(hzs):
    c=len(hzs)
    img=np.array([[0]*16*20]*16*(c/20))
    for i in range(c/20):
        for j in range(20):
            hz=hzs[i*10+j];
            for k in range(len(hz)):
                img[i*16+hz[k][0],j*16+hz[k][1]] = 1
    plt.imshow(img)



def unique(a):
 order = np.lexsort(a.T)
 a = a[order]
 diff = np.diff(a, axis=0)
 ui = np.ones(len(a), 'bool')
 ui[1:] = (diff != 0).any(axis=1)
 return a[ui]

def trans(hz,t):
    for i in range(len(hz)):
        hz[i]=np.dot(t,hz[i])
    return unique(hz)

#最好都是0
def uniquex(hz):
    a,b,c=hz.T
    ndx=np.lexsort((a,b,c))
    hz=hz[ndx,]
    A1=np.diff(hz,axis=0)
    ndx=np.any(A1,axis=1)
    return hz[1:,:][ndx,]


def rotate(hz,a):
    cosa=math.cos(a)
    sina=math.sin(a)
    t = np.array([cosa,-sina,sina,cosa])
    t=t.reshape(2,2)
    trans(hz,t)

def zoom(hz,z):
    t=np.array([[z,0,0],[0,z,0],[0,0,z]])
    if z<=1:# Add point
        return trans(hz,t)
    else:
        return trans(hz,t)
        new=np.array([[0,0,0]]*z*z*len(hz))
        for i in range(len(ts)):
            for j in range(z):
                for k in range(z):
                    new[i * z * z + j * z + k] = np.array([hz[i][0] + j, hz[i][1] + k, hz[i][2]])
                    #new[i*z*z+j*z+k]=hz[i]
        return uniquex(new)

def zoomx(hz,z):
    t=np.array([[z,0,0],[0,1,0],[0,0,1]])
    return trans(hz,t)
def zoomy(hz,z):
    t=np.array([[1,0,0],[0,z,0],[0,0,1]])
    return trans(hz,t)

#chuoqie,qiebian
def qbx(hz,k):
    t=np.array([[1,k,0],[0,1,0],[0,0,1]])
    trans(hz,t)
def qby(hz,k):
    t=np.array([[1,0,0],[k,1,0],[0,0,1]])
    trans(hz,t)
def fans(hz,x,y):
    t=np.array([[2*x*x-1,2*x*y],[2*x*y,2*y*y-1]])
    trans(hz,t)
def touy(hz,x,y):
    t=np.array([[x*x,x*y],[x*y,y*y]])
    trans(hz,t)
def piny(hz,x,y):
    t=np.array([[1,0,x],[0,1,y]])
    for i in range(len(hz)):
        a=hz[i]
        a=np.resize(a,3)
        a[2]=1
        a=np.dot(t,a)
        hz[i][0]=a[0]
        hz[i][1] = a[1]


def distxy(x,y):
    xy=0.0;
    for i in range(len(x)):
        xy=xy+(x[i]-y[i])*(x[i]-y[i])
    return xy


def cosVector(xx,yy):
    x=xx.copy()
    y=yy.copy()
    xy=0.0;
    xx=0.0;
    yy=0.0;
    for i in range(len(x)-1):
        xy+=x[i]*y[i]   #sum(X*Y)
        xx+=x[i]**2     #sum(X*X)
        yy+=y[i]**2     #sum(Y*Y)
    ret=xy/((xx*yy)**0.5)
    return ret

def cosVector_o(x,y):
    if(len(x)!=len(y)):
        #print('error input,x and y is not in the same space')
        return;
    xy=0.0;
    xx=0.0;
    yy=0.0;
    for i in range(len(x)):
        xy+=x[i]*y[i]   #sum(X*Y)
        xx+=x[i]**2     #sum(X*X)
        yy+=y[i]**2     #sum(Y*Y)
    ##print(result1)
    ##print(result2)
    ##print(result3)
    ret=xy/((xx*yy)**0.5)
    ##print("result is "+str(ret)) #结果显示
    return ret

def findByRow(mat,row):
    return np.where((mat==row).all(1))[0]

#tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#    wa = np.array([[[[0.0] * 1] * 1] * 3] * 3)
#    r=sess.run(c2db,feed_dict={x: batch[0],w:wa}) x:-1,28,28,1

def connect(hz,mn):
    cj=[]
    to=[]
    for i in range(len(hz)):
        cm = 99999
        jmin=0
        for j in range(len(mn)):
            mn3 = mn[j]
            #c = distxy(hz[i], mn3)
            c = (hz[i][0]-mn3[0])*(hz[i][0]-mn3[0])\
              + (hz[i][1]-mn3[1])*(hz[i][1]-mn3[1])
            if c < cm:
                cm = c
                jmin = j
                if(cm==0):break
        #x,y,z=zip(hz[i],mn[imax])
        #ax.plot(x,y,z)
        ##print cm,cj
        #if(cm!=0):#Counter =0 也计算
        #    cj.append(cm)
        #    to.append([mn[jmin][0]-hz[i][0],mn[jmin][1]-hz[i][1]])
        cj.append(cm)
        to.append([mn[jmin][0]-hz[i][0],mn[jmin][1]-hz[i][1]])

    ##print ("sang:%.2f%%" %(len(Counter(cj)) * 100.0 / len(hz)),
    ##print ("std:%.2f,mean:%.2f,sang:%.2f%%"%(np.std(cj), np.mean(cj), len(Counter(cj)) * 100.0 / len(hz)),
    meancj=np.mean(cj)
    stdcj=np.std(cj)
    sang= len(Counter(cj)) * 100.0 / len(hz) #sang
    bianyixishu= stdcj/meancj ##变异系数
    ##print ("std:%.2f,mean:%.2f,sang:%.2f%%,by:%.2f" % (stdcj,meancj,sang,bianyixishu),
    ##print meancj/(28*4*28*4)*100 #平均比最长
    #if(len(Counter(cj))*100/len(hz)<30):
    #    #print ("     ",Counter(cj)
    #return (1+np.std(cj))*(1+np.mean(cj))
    #return  np.std(cj) * np.mean(cj)
    #return np.std(cj)/np.mean(cj)  #变异系数
    #return sang #* stdcj/meancj # *stdcj*meancj #sang/()
    #return sang
    ##########return sang*bianyixishu #!!!!!

    meanCos=np.mean(cj)
    stdCos=np.std(cj)
    sangCos= len(Counter(cj)) * 100.0 / len(hz) #sang
    bianyixishuCos= stdCos/meanCos ##变异系数

    #print ("s:%.2f%%,b:%.2f" % (sang, bianyixishu)),
    x,y=zip(*to)
    #print (np.std(x),np.std(y),np.mean(x),np.mean(y))

    return sang*bianyixishu

def connect_o(hz,mn):
    cj=[]
    for i in range(len(hz)):
        #if(len(findByRow(hz, [hz[i][0] +0, hz[i][1] + 1, hz[i][2]]))
        #    and len(findByRow(hz,[hz[i][0]+0,hz[i][1]-1,hz[i][2]]))
        #    and len(findByRow(hz,[hz[i][0]-1,hz[i][1]+0,hz[i][2]]))
        #    and len(findByRow(hz,[hz[i][0]+1,hz[i][1]+0,hz[i][2]]))):
        #    continue

        cm = 0
        imax = 0
        #hz3 = hz[i]
        for j in range(len(mn)):
            mn3 = mn[j]
            c = cosVector(hz[i], mn3)
            if c > cm:
                cm = c
                imax = j
        #x,y,z=zip(hz[i],mn[imax])
        #ax.plot(x,y,z)
        ##print cm,cj
        cj.append(cm*100)
    ##print np.std(cj),np.mean(cj),":",cj
    return 1-np.std(cj)/np.mean(cj)
        #ax.plot([hz3[0], mn[imax][0]], [hz3[1], mn[imax][1]], [hz3[2], mn[imax][2]])



def fullconectab(a,b):
    la=len(a)
    lb=len(b)
    if(la>lb):
        return connect(a,b)
    else:
        return connect(b,a)



def inputmn(mnist):
    xs, ys = mnist.train.next_batch(100)  # 100
    imgs = []
    for m in range(len(xs)):
        img = []
        for i in range(28):
            for j in range(28):
                if xs[m][i*28+j]>0.2 :
                    img.append([(i-14), ( j-14),1])
        imgs.append(img)
    return imgs,ys



def ffankui(a,b):
    dir=1
    step=0.1
    r1 = fullconectab(a, b)
    ao=a.copy()
    sadd=1
    while True:
        #a=ao.copy()
        r = fullconectab(a,b)
        zoom(a,1+dir*step)
        sadd=sadd*(1+dir*step)
        #print ("sadd",sadd)
        r1=fullconectab(a,b)
        if r1<r:
            dir = -dir
            if(step>0.01):
                step=step-0.01
            #else:
            #    break
        dr=r1-r
        #print ("dr:",(dr))
        #print ("r,r1",r,r1)
        if(abs(dr)<0.00001):
            break
    return sadd


def ffankuizoomx(a,b):
    dir=1
    step=0.1
    r1 = fullconectab(a, b)
    ao=a.copy()
    sadd=1
    while True:
        #a=ao.copy()
        r = fullconectab(a,b)
        zoomx(a,1+dir*step)
        sadd=sadd*(1+dir*step)
        ##print ("sadd",sadd
        r1=fullconectab(a,b)
        if r1<r:
            dir = -dir
            if(step>0.01):
                step=step-0.01
            #else:
            #    break
        dr=r1-r
        #print ("dr:",(dr))
        #print ("r,r1",r,r1)
        if(abs(dr)<0.00001):
            break
    return sadd


def fufankui(a,b,func):
    dir=1
    step=0.1
    r1 = fullconectab(a, b)
    ao=a.copy()
    sadd=1
    while True:
        #a=ao.copy()
        r = fullconectab(a,b)
        func(a,dir*step)
        sadd=sadd*(1+dir*step)
        ##print ("sadd",sadd
        r1=fullconectab(a,b)
        if r1<r:
            dir = -dir
            if(step>0.02):
                step=step/2
            else:
                step = step - 0.01
        #if( step<0.01):
        #    break
        dr=r1-r
        ##print ("dr:",(dr)
        ##print ("r,r1,step",r,r1,step
        if(abs(dr)<=0.001):
            break
    return sadd

def showab(mn,hz):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    ax.view_init(-90, 0)
    xx,yy,zz=zip(*hz)
    ax.scatter(xx,yy,zz,c='r')
    mx,my,mz=zip(*mn)
    ax.scatter(mx,my,mz,c='b')
    plt.show()

def show3dimg(img):#hz16x16img
    rows=len(img)
    cols=len(img[0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.set_xlim(0, rows)
    #ax.set_ylim(0, cols)
    ax.set_zlim(0, 10)
    ax.view_init(60, 0)

    #xx, yy, zz = zip(*hz)
    for i in range(rows):
        for j in range(cols):
            if(img[i][j]!=0):
                ax.scatter(i, j, img[i][j], c='r') #,marker='.'
    plt.show()

def findz(hz,x,y):
    for i in range(len(hz)):
        if(hz[i][0]==x and hz[i][1]==y):
            return hz[i][2]
            break;
    return -1;

def show3d(hz):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xx, yy, zz = zip(*hz)
    ax.scatter(xx, yy, zz,marker='+', c='b')
    cnt=len(hz)
#    for i in range(cnt):
#        f1=findz(hz,xx[i]-1,yy[i])
#        f2 = findz(hz, xx[i] + 1, yy[i])
#        f3=findz(hz,xx[i],yy[i]-1)
#        f4 = findz(hz, xx[i], yy[i]+1)
#        if (zz[i]>=f1 and zz[i]>=f2) or (zz[i]>=f3 and zz[i]>=f4):
#            ax.scatter(xx[i],yy[i],zz[i],c='r')
#        if (zz[i]>=f1 and zz[i]>=f2) :
#            ax.scatter(xx[i],yy[i],zz[i],c='r')
#        if  (zz[i]>=f3 and zz[i]>=f4):
#            ax.scatter(xx[i],yy[i],zz[i],c='r')

    plt.show()

def cvd(img,i,j,cmat):
    c=0
    for x in xrange(-1,2,1):
        for y in xrange(-1,2,1):
            xx=i+x
            yy=j+y
            c=c+img[xx][yy]*cmat[x][y]
    return int(c)

def conv2d(hz,w,cmat):
    #w=len(hz)
    img=np.array([[0]*w]*w)
    cvdimg=np.array([[0]*w]*w)
    for i in range(len(hz)):
        img[hz[i][0]+w/2,hz[i][1]+w/2]=1.0*hz[i][2]
    cnd=[]
    for i in range(1,w-1):
        for j in range(1,w-1):
            cvdimg[i][j]=cvd(img,i,j,cmat)
            if(cvdimg[i][j] != 0):
                cnd.append([(w/2- i), (0+ j-w/2),cvdimg[i][j]])

    hz=np.array(cnd)
    return hz


def high(img,i,j):
    cmat=[[1,1,1],
          [1,1,1],
          [1,1,1]]
    c=0
    for x in xrange(-1,2,1):
        for y in xrange(-1,2,1):
            xx=i+x
            yy=j+y
            c=c+img[xx][yy]*cmat[x][y]
    #if(c!=1):c=0  #只取1，表示周围为1
    return 9-c

def convtoxi(hz):
    w=len(hz)
    img=np.array([[0]*w]*w)
    cvdimg=np.array([[0]*w]*w)
    for i in range(len(hz)):
        img[hz[i][0]+w/2,hz[i][1]+w/2]=1
    for i in range(1,w-1):
        for j in range(1,w-1):
#            nc=img[i-1][j-1]+img[i][j-1]+img[i+1][j-1]+\
#               img[i-1][j]+img[i][j]+img[i+1][j]+\
#               img[i-1][j+1]+img[i][j+1]+img[i+1][j+1]
            nc=            img[i][j-1] +\
               img[i-1][j]+img[i][j]+img[i+1][j]+\
                           img[i][j+1]
            cvdimg[i][j]=nc

    for i in range(1,w-1):
        for j in range(1,w-1):
            if cvdimg[i][j]<cvdimg[i][j-1]\
                    or cvdimg[i][j]<cvdimg[i-1][j-1] \
                    or cvdimg[i][j]<cvdimg[i+1][j-1] \
                    or cvdimg[i][j]<cvdimg[i][j+1] \
                    or cvdimg[i][j]<cvdimg[i-1][j+1] \
                    or cvdimg[i][j]<cvdimg[i+1][j+1] \
                    or cvdimg[i][j]<cvdimg[i-1][j] \
                    or cvdimg[i][j] <cvdimg[i+1][j]:
                cvdimg[i][j]=0


    cnd=[]
    for i in range(1, w - 1):
        for j in range(1, w - 1):
            if(cvdimg[i][j]!=0):
                cnd.append([(w/2- i), (0+ j-w/2),w/4])

    return np.array(cnd)


def conv(img,i,j):
    cmat=[[-1,-1,-1],
          [-1,9,-1],
          [-1,-1,-1]]
    c=0
    for x in xrange(-1,2,1):
        for y in xrange(-1,2,1):
            xx=i+x
            yy=j+y
            c=c+img[xx][yy]*cmat[x][y]
    #if(c!=1):c=0  #只取1，表示周围为1
    return c





def convline(hz,w,conv):
    img=np.array([[0]*w]*w)
    cvdimg=np.array([[0]*w]*w)
    for i in range(len(hz)):
        img[hz[i][0]+w/2,hz[i][1]+w/2]=1.0*hz[i][2]
    cnd=[]
    for i in range(1,w-1):
        for j in range(1,w-1):
            cvdimg[i][j]=conv(img,i,j)
            if(cvdimg[i][j]!=0):
                cnd.append([(w/2- i), (0+ j-w/2),cvdimg[i][j]])

    hz=cnd
    return np.array(cnd)


def toushi(mn,w,h,f,n):
    cmat=[[2*n/w,0,0,0],
          [0,2*n/h,0,0],
          [0,0,f/(f-n),-f*n/(f-n)],
          [0,0,1,0]]
    o=np.ones(len(mn))
    mn=np.c_[mn,o]
    for i in range(len(mn)):
        p=mn[i]
        n=np.dot(p,cmat)
        n=n/n[3]
        mn[i]=n

    mn=np.delete(mn,3,1)
    return mn

def runkuo(img,i,j):
    cmat=[[-1,-1,-1],
          [-1,8,-1],
          [-1,-1,-1]]
    c=0
    for x in xrange(-1,2,1):
        for y in xrange(-1,2,1):
            xx=i+x
            yy=j+y
            c=c+img[xx][yy]*cmat[x][y]
    if(c!=0):c=1  #只取1，表示周围为1
    return c

def hexin(img,i,j):
    cmat=[[-1,-1,-1],
          [-1,9,-1],
          [-1,-1,-1]]
    c=0
    for x in xrange(-1,2,1):
        for y in xrange(-1,2,1):
            xx=i+x
            yy=j+y
            c=c+img[xx][yy]*cmat[x][y]
    if(c!=1):c=0  #只取1，表示周围为1
    #if (c != 0): c = 0
    #else:c=1
    return c

def hz2array(hz):
    img=np.array([[0]*16]*16)
    for i in range(len(hz)):
        img[hz[i][0]+7,hz[i][1]+7]=1
    return img

def hz2img(hz):
    img=np.array([[0]*16]*16)
    for i in range(len(hz)):
        img[hz[i][0]+7,hz[i][1]+7]=1
    return img




def test1():
    cmat = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]



    for i in range(10):
        nb=int(np.dot(ys[i],[0,1,2,3,4,5,6,7,8,9]))
        if(nb!=0):
            continue
        ##print ("mn:",nb
        mx=99999
        idx=0
        for j in range(10):
            mn = np.array(allmn[i]).copy()
            hz = np.array(allhz[j])
            #mn=conv2d(mn,28,cmat)
            #mn=conv2d(mn,28,cmat)
            #show3d(mn)
            #hz=conv2d(hz,16,cmat)
            #show3d(hz)
            #hz=conv2d(hz,16,cmat)
            #showab(mn,hz)

            mn=zoom(mn,4)
            hz=zoom(hz,7)

    #        show3d(mn)

            #mn=convline(mn,28)
            #w=28.0;h=28.0;f=28*9;n=28.0
            #mn=toushi(mn,w,h,f,n)
            #show3d(mn)
            #hz=conv2d(hz,16)

            #mn=zoom(mn,1.0/3)
            #hz=zoom(hz,1.0/3)
            #showab(mn,hz)
            #mn=zoom(mn,1.0/2)
            #hz=zoom(hz,1.0/2)
            #showab(mn,hz)

            #fufankui(mn,hz,qby)
            #fufankui(mn,hz,qbx)# 0.549579196779
            #mn=conv2d(mn)
            #hz=conv2d(hz)
            c=fullconectab(mn,hz)
            #print ("%d:c:%.2f"%( j, c))

            if(c<35): #28*4/12=9  9*9  9*9
                showab(mn,hz)
                #f=fufankui(mn, hz, qby)
                #f=fufankui(mn, hz, qbx)  # 0.549579196779
                #f=fufankui(mn, hz, zoom)
                #f = fufankui(mn, hz, zoomy)
                #c=fullconectab(mn,hz)
                ##print c
                #showab(mn,hz)
            ##print j,c
            if(c<mx):
                mx=c
                idx=j
        #print ("nd:%d,hz:%d "%(nb,idx))
        showab(mn, np.array(allhz[idx]))
        #print ("---------------")
        #break

    #std,mean: 69.1277542943 58.4258373206
    #0.0 0 4167.40051835
    #0.0 5 7067.19496396
    #std,mean: 58.4587160961 49.8181818182
    #0.0 6 3021.58384525
    #std,mean: 64.6529481826 55.0765550239
    #0.0 8 3681.59116125
    #std,mean: 70.4380518626 51.5598086124
    #0.0 9 3754.77033355
    #std,mean: 225.629084822 207.167464115
    #std,mean: 140.522246709 103.976076555
    #0.0 3 14856.4502047
    #std,mean: 325.509532938 317.057416268
    #0.0 1 103848.778433
    #std,mean: 179.899223434 113.85645933
    #0.0 2 20777.4442992
    #std,mean: 110.996133758 83.4449760766
    #0.0 4 9457.51083586
    #std,mean: 88.8554415055 77.6507177033
    #0.0 7 47176.801882

    def test():
        hz=np.array( allhz[3])
        mn=np.array( allmn[5])
        for i in range(0,10):

            nb=np.dot(ys[i],[0,1,2,3,4,5,6,7,8,9])
            mn = np.array(allmn[i])
            #print ("mn:",nb)
            #print ("befor:",fullconectab(hz,mn))
        #    #print fufankui(mn,hz,qby)
        #    #print fufankui(mn,hz,qbx)# 0.549579196779
        #    #print fufankui(mn,hz,zoomx)
        #    #print fufankui(mn,hz,zoomy)
        #    #print ("after:",fullconectab(hz,mn)
        #    #print ("----------------------"
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_zlim(-20, 20)
            ax.view_init(90, 0)
            xx,yy,zz=zip(*hz)
            ax.scatter(xx,yy,zz,c='r')
            mx,my,mz=zip(*mn)
            ax.scatter(mx,my,mz,c='b')
            plt.show()


    ##print ffankui(mn,hz)
    #zoom(mn,1.04190586955)
    #0.49
    ##print ffankuizoomx(mn,hz)
    #zoomx(mn,1.06370544902) #0.494689799169
    #zoomy(mn,1.03392169645) #0.486397037285
    ##print fufankui(mn,hz,qby)
    ##print fufankui(mn,hz,qbx)# 0.549579196779
    #qbx(mn,-1)
    #qby(mn,0.663123269835)
    ##print fufankui(mn,hz,zoomx)
    ##print fufankui(mn,hz,zoomy)

    ##print ("after:",fullconectab(hz,mn)
    #showhz(mn)
    #for i in range(55,56):
    #    mn=np.array( allmn[i])
    #    ret=fullconectab(hz,mn)
    #    #print i,ret
        #if(ret>0.6):
            #showhz(mn)
            #plt.show()

    #hz=np.array( allmn[9])
    #showhz(mn)

    #cos 0-45, 360-315
    #sin 0-45   360-315
    #rotate(hz,-math.pi/8)
    #zoom(hz,2.5)
    #qby(hz,1)
    #piny(hz,0,-10)
    #showhz(hz)
    #showhz(mn)
    #hz=zoom(hz,0.5)
    #hz=zoom(hz,0.5)
    #hz=zoom(hz,0.5)
    #hz=zoom(hz,0.5)
    #fans(hz,1,1)
    #touy(hz,1,2)
    #piny(hz,15,15)
    #showhz(hz)
    #plt.show()


    #xx,yy,zz=zip(*hz)
    #ax.scatter(xx,yy,zz,c='r')
    #mx,my,mz=zip(*mn)
    #ax.scatter(mx,my,mz,c='b')

    #    plt.plot(hz3,[mn[imax][0], mn[imax][1], 14])
    #ax = fig.gca(projection='3d')
    #plt.plot([0,0,0],[5,5,5],[8,8,8])

def test1():
    allhz = hzk()

    b = brain()
    hz0 = hz2array(allhz[0])
    hz1 = hz2array(allhz[0])
    #print (b.comp(hz0, hz1))


def mn2array16(mn):
    img=np.array([[0]*28]*28)
    for i in range(len(mn)):
        img[int(mn[i][0]/2)+14,int(mn[i][1]/2)+14]=1
    mn=img
    left = 0;
    right = 29;
    top = 0;
    bottom = 29
    for n in range(28):
        if (mn[n, :].max() > 0 and left == 0):
            left = n
        if (mn[-n, :].max() > 0 and right == 29):
            right = 28 - n
        if (mn[:, n].max() > 0 and top == 0):
            top = n
        if (mn[:, -n].max() > 0 and bottom == 29):
            bottom = 30 - n
    i=int((16-(right-left))/2)
    j=int((16-(bottom-top)+1)/2)

    new = mn[top-j:bottom+j, left-i:right+i]
    return new

def mn2array28(mn):
    img=np.array([[0]*28]*28)
    for i in range(len(mn)):
        img[mn[i][0]/2+14,mn[i][1]/2+14]=mn[i][2]
    return img

def mn2array(mn):
    mi=np.min(mn)
    mx=np.max(mn)
    cnt=mx-mi+1
    img=np.array([[0]*cnt]*cnt)
    for i in range(len(mn)):
        img[mn[i][0]-mi,mn[i][1]-mi]=1
    return img


def Convolution(hz,mat):

    b = brain(2,16,16)
    b.input(hz, 0,0,0)

    b.convolutionfun(0, 0,0,16, 16, 1,0,0, mat,actvand)
    b.conduct()
    r = b.getAxonMat(1,0,0,16,16)
    return r

def testConvolution():
    allhz = hzk()
    hz = hz2array(allhz[0])
    cmat=[[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
    r=Convolution(hz,cmat)
    plt.imshow(r)
    plt.show()


def circle(r):
    cmat=np.array([[0]*(2*r+1)]*(2*r+1))
    for i in range(2*r+1):
        x=i-r
        for j in range(2*r+1):
            y=j-r
            if(abs(np.sqrt(x*x+y*y)-np.sqrt(r*r))<0.5):
                cmat[i][j]=1
    return cmat
def semicircleA(r):
    cmat=np.array([[0]*(2*r+1)]*(r))
    for i in range(r):
        x=i-r
        for j in range(2*r+1):
            y=j-r
            if(abs(np.sqrt(x*x+y*y)-np.sqrt(r*r))<0.5):
                cmat[i][j]=1
    return cmat

def semicircleV(r):
    cmat=np.array([[0]*(2*r+1)]*(r))
    for i in range(r+1,2*r+1):
        x=i-r
        for j in range(2*r+1):
            y=j-r
            if(abs(np.sqrt(x*x+y*y)-np.sqrt(r*r))<0.5):
                cmat[i-r-1][j]=1
    return cmat

def testFeature():
    allhz = hzk()

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)
    b = brain()
    cmat = semicircleV(3)
    cmatA=semicircleA(3)

    mn = mn2array16(allmn[7])# 4:1 7:0
    plt.imshow(mn)
    plt.show()
    b.input(mn, 10, 0, 0)
    b.convolution(10, 16, 16, 13, cmat)
    b.convolution(10, 16, 16, 14, cmatA)
    b.neuronsconduct(10,0,0,16,16)
    r = b.getValue(13, 16, 16)
    plt.imshow(r)
    plt.show()
    r = b.getValue(14, 16, 16)
    plt.imshow(r)
    plt.show()

    for j in range(0, 10):
        hz = hz2array(allhz[j])
        b.input(hz, j,0,0)
        b.convolution(j, 16, 16, 12, cmat)
        b.conduct()
        r = b.getValue(12, 16, 16)
        plt.imshow(r)
        plt.show()

        #cmat=[[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
    #cmat=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    #cmat=[[1],[1],[1],[1]]
    #cmat=[[1,1,1,1]]


def test3():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allhz = hzk()
    allmn,ys=inputmn(mnist)

    b = brain()
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        b.input(hz, j)

    b.setcompsynaspse16(10)


    for i in range(0,10):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue
        mn=mn2array16(allmn[i])
        plt.imshow(mn)
        plt.show()

        b.input(mn,10)
        b.clear()
        b.conduct()
        img,r=b.getresult()
        plt.imshow(img)
        plt.show()

def test4():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allhz = hzk()
    allmn,ys=inputmn(mnist)

    b = brain()

    for i in range(5):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 1):
            continue
        mn=mn2array16(allmn[i])
        plt.imshow(mn)
        plt.show()
        b.input(mn,0)
        img=b.connect22(0,1)
        plt.imshow(img)
        plt.show()

def center(mn):
    left = 0;
    right = len(mn[0]);
    top = 0;
    bottom = len(mn)
    for n in range(right):
        if (mn[n, :].max() > 0 and top == 0):
            top = n
        if (mn[-n, :].max() > 0 and bottom == len(mn[0])):
            bottom = len(mn[0]) - n
        if (mn[:, n].max() > 0 and left == 0):
            left = n
        if (mn[:, -n].max() > 0 and right == len(mn)):
            right = len(mn) - n

    new = mn[top-1:bottom+1+1, left-1:right+1+1]
    return new


def test5():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allhz = hzk()
    allmn,ys=inputmn(mnist)

    b = brain()
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        b.input(hz, j,0,0)

    b.setcompsynaspse16(10)

    for i in range(0,9):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue
        #mn=mn2array28(allmn[i])
        mn = mn2array16(allmn[i])
        #plt.imshow(mn)
        #plt.show()
        mn=center(mn)
        b.input(mn,10,(16-(right-left+1))/2,(16-(bottom-top+1))/2)
        b.clear()
        b.conduct()
        img,r=b.getresult()
        #print ("r:",r)
        plt.imshow(img)
        plt.show()

def testComplayer():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allhz = hzk()
    allmn,ys=inputmn(mnist)

    b = brain()
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        b.input(hz, j,0,0)


    for i in range(0,9):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue
        mn28=mn2array28(allmn[i])
        #plt.imshow(mn)
        #plt.show()
        mn=center(mn28)

        b.input(mn,10,(16-len(mn))/2,(16-len(mn))/2)
        ##print b.getValue(10,28,28)
        #cmat=[[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
        #cmat=[[1,0],[0,-1]]
        #cmat=[[1]]
        #cmat = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
        #        [0, 0, 0], [0, 0, 0], [0, 0, 0],
        #        [0, 0, 0], [0, 0, 0], [0, 0, 1]]
        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        cmat = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        b.convolution(10,28,28,11,cmat)
        cmat = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        b.convolution(11,28,28,12,cmat)
        b.conduct()
        r = b.getValue(12,16,16)
        plt.imshow(r)
        plt.show()

def diff(hz):
    df=[]
    for r in range(15):
        dr=[]
        for c in range(15):
            dr.append(0)
            dr.append(hz[r][c+1]-hz[r][c])
        dr.append(0)
        dr.append(0)
        df.append(dr)
        lastdr=dr
        dr=[]
        for c in range(16):
            if(hz[r+1][c]-hz[r][c]):
                dr.append(hz[r + 1][c] - hz[r][c])
                if(hz[r + 1][c+1] - hz[r][c+1]):
                    if(hz[r][c+1]-hz[r][c]==0):
                        dr.append(hz[r+1][c]-hz[r][c])
                    else:
                        dr.append(0)
                else:
                    dr.append(0)
            else:
                dr.append(lastdr[c*2])
                dr.append(lastdr[c * 2+1])
        #dr.append(hz[r + 1][15] - hz[r][15])
        #dr.append(hz[r + 1][15] - hz[r][15])
        df.append(dr)
    dr=[]
    for c in range(15):
        dr.append(0)
        dr.append(hz[15][c+1]-hz[15][c])
    dr.append(0)
    dr.append(0)
    df.append(dr)
    new=np.array(df)
    return new*new


def testDiff():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allhz = hzk()
    allmn,ys=inputmn(mnist)

    hz16=[]
    b = brain(16,32,32)
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        hz16.append(hz)
        b.input(hz, j,0,0)
    hz=hz16[0]

    new=diff(hz)

    plt.imshow(new)
    plt.show()
    return

    hz=center(hz)
    h=len(hz);w=len(hz[0])
    cmat = hz[0:h/2,0:w/2]
    cmat1 = hz[h/2:h,0:w/2]
    cmat2 = hz[h/2:h,w/2:w]
    cmat3 = hz[0:h/2,w/2:w]

    b.convolution(10, 16, 16, 11, cmat)
    b.convolution(10, 16, 16, 12, cmat1)
    b.convolution(10, 16, 16, 13, cmat2)
    b.convolution(10, 16, 16, 14, cmat3)
    b.maxpooling(11,16,16,15,4,4)
    b.maxpooling(12,16,16,15,4,4)
    b.maxpooling(13,16,16,15,4,4)
    b.maxpooling(14,16,16,15,4,4)

    for i in range(0,9):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue
        b.clear(11,0,0,16,16)
        b.clear(12,0,0,16,16)
        b.clear(13,0,0,16,16)
        b.clear(14,0,0,16,16)
        mn=mn2array16(allmn[i])
        tmn=diff(mn)
        b.input(tmn,15,0,0)
        b.pooling(15,32,32,16,2,2,actvsum)
        b.conduct(15,32,32)
        r=b.getValue(15,32,32)
        plt.imshow(r)
        plt.show()
        r=b.getValue(16,16,16)
        plt.imshow(r)
        plt.show()
        return
        b.input(mn, 10, 0, 0)
        b.conduct(10, 16, 16)

        r = b.getValue(15, 4, 4)
        #plt.imshow(r)
        #plt.show()

    return


def test9():
    allhz = hzk()
    hz16=[]
    b = brain()
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        hz16.append(hz)
        b.input(hz, j,0,0)

    cmat = hz16[0][0:8,0:8]

    for j in range(10):
        b.convolution(j, 16, 16, j+10, cmat)
        #b.maxpooling(j+10,16,16,j+20,4,4)
        b.conduct(j, 16, 16)
        r = b.getValue(j+10, 4, 4)
        plt.imshow(r)
        plt.show()

    return


def testRotate():

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    b = brain()
    b.rotate(10, 10, 7, 7, 11)
    b.rotate(10, 20, 7, 7, 12)
    b.rotate(10, 30, 7, 7, 13)
    b.rotate(10, 40, 7, 7, 14)
    b.rotate(10, -10, 7, 7, 15)
    b.rotate(10, -20, 7, 7, 16)
    b.rotate(10, -30, 7, 7, 17)
    b.rotate(10, -40, 7, 7, 18)
    b.getArea(10)
    b.getArea(11)
    b.getArea(12)
    b.getArea(13)
    b.getArea(14)
    b.getArea(15)
    b.getArea(16)
    b.getArea(17)
    b.getArea(18)

    nmin = neuron(-1,-1,-1)
    nmin.axon.activation = actvminidx
    layer = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    angle=[0,10,20,30,40,-10,-20,-30,-40]
    #layer = [1,11]
    for la in layer:
        b.getNeuron(la, 16, 16).connectto(nmin)

    for i in range(0,52):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 1):
            continue

        mn=mn2array16(allmn[i])
        #mn=center(mn)
        #b.input(mn,0,(16-len(mn))/2,(16-len(mn))/2)
        #b.clear(10,0,0,16,16)
        b.input(mn,10,0,0)
        plt.imshow(mn)
        plt.show()

        b.conduct()

        idx= nmin.axon.getValue()
        r=b.getValue(layer[idx], 16, 16)

        plt.imshow(r)
        plt.show()

    return


def testRotate():
    allhz = hzk()

    b = brain(28,28,28)
    b.rotate(10, 10, 7, 7, 11)
    b.rotate(10, 20, 7, 7, 12)
    b.rotate(10, 30, 7, 7, 13)
    b.rotate(10, 40, 7, 7, 14)
    b.rotate(10, -10, 7, 7, 15)
    b.rotate(10, -20, 7, 7, 16)
    b.rotate(10, -30, 7, 7, 17)
    b.rotate(10, -40, 7, 7, 18)
    b.getArea(10)
    b.getArea(11)
    b.getArea(12)
    b.getArea(13)
    b.getArea(14)
    b.getArea(15)
    b.getArea(16)
    b.getArea(17)
    b.getArea(18)

    nmin = neuron(-1,-1,-1)
    nmin.axon.activation = actvminidx
    layer = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    angle=[0,10,20,30,40,-10,-20,-30,-40]
    #layer = [1,11]
    for la in layer:
        b.getNeuron(la, 16, 16).connectto(nmin)

    for i in range(0,9):
        #nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #if (nb != 1):
        #    continue

        #mn=mn2array16(allmn[i])
        #mn=center(mn)
        #b.input(mn,0,(16-len(mn))/2,(16-len(mn))/2)
        #b.clear(10,0,0,16,16)
        mn=allhz[i]
        b.input(mn,10,0,0)
        plt.imshow(mn)
        plt.show()

        b.conduct()

        idx= nmin.axon.getValue()
        r=b.getValue(layer[idx], 16, 16)

        plt.imshow(r)
        plt.show()

    return


def testSelect():

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    b = brain()
    b.rotate(10, 10, 7, 7, 11)
    b.rotate(10, 20, 7, 7, 12)
    b.rotate(10, 30, 7, 7, 13)
    b.rotate(10, 40, 7, 7, 14)
    b.rotate(10, -10, 7, 7, 15)
    b.rotate(10, -20, 7, 7, 16)
    b.rotate(10, -30, 7, 7, 17)
    b.rotate(10, -40, 7, 7, 18)
    b.getArea(10)
    b.getArea(11)
    b.getArea(12)
    b.getArea(13)
    b.getArea(14)
    b.getArea(15)
    b.getArea(16)
    b.getArea(17)
    b.getArea(18)

    nmin = neuron(-1,-1,-1)
    nmin.axon.activation = actvminidx
    layer = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    angle=[0,10,20,30,40,-10,-20,-30,-40]
    #layer = [1,11]
    for la in layer:
        b.getNeuron(la, 16, 16).connectto(nmin)

    #selected
    for i in range(16):
        for j in range(16):
            nselect=neuron(-1,i,j)
            nselect.dendritic.selected=nmin.axon
            nselect.axon.activation=actvselect
            for k in range(len(layer)):
                n=b.getNeuron(layer[k],i,j)
                n.connectto(nselect)
            nselect.connectto(b.getNeuron(19,i,j))

    cmat = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    b.convolution(19, 16, 16, 20, cmat)

    for i in range(0,52):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue

        mn=mn2array16(allmn[i])
        #mn=center(mn)
        #b.input(mn,0,(16-len(mn))/2,(16-len(mn))/2)
        #b.clear(10,0,0,16,16)
        b.input(mn,10,0,0)
        plt.imshow(mn)
        plt.show()

        b.conduct()

        idx= nmin.axon.getValue()
        r=b.getValue(layer[idx], 16, 16)


        r=b.getValue(19, 16, 16)
        plt.imshow(r)
        plt.show()

        r=b.getValue(20, 16, 16)
        plt.imshow(r)
        plt.show()


    return


def testComplayer():

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    b = brain(16,16,16)
    b.rotate(10, 10, 7, 7, 11)
    b.rotate(10, 20, 7, 7, 12)
    b.rotate(10, 30, 7, 7, 13)
    b.rotate(10, 40, 7, 7, 14)
    b.rotate(10, -10, 7, 7, 15)
    b.rotate(10, -20, 7, 7, 16)
    b.rotate(10, -30, 7, 7, 17)
    b.rotate(10, -40, 7, 7, 18)
    b.getArea(10)
    b.getArea(11)
    b.getArea(12)
    b.getArea(13)
    b.getArea(14)
    b.getArea(15)
    b.getArea(16)
    b.getArea(17)
    b.getArea(18)

    nmin = neuron(-1,-1,-1)
    nmin.axon.activation = actvminidx
    layer = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    angle=[0,10,20,30,40,-10,-20,-30,-40]
    #layer = [1,11]
    for la in layer:
        b.getNeuron(la, 16, 16).connectto(nmin)

    #selected
    for i in range(16):
        for j in range(16):
            nselect=neuron(-1,i,j)
            nselect.dendritic.selected=nmin.axon
            nselect.axon.activation=actvselect
            for k in range(len(layer)):
                n=b.getNeuron(layer[k],i,j)
                n.connectto(nselect)
            nselect.connectto(b.getNeuron(19,i,j))

    cmat = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    b.convolution(19, 16, 16, 20, cmat)
    #b.convolution(0, 16, 16, 21, cmat)

    allhz=hzk()
    hz16=[]
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        hz16.append(hz)
        b.input(hz, j,0,0)

    for i in range(0,10):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue

        mn=mn2array16(allmn[i])
        #mn=center(mn)
        #b.input(mn,0,(16-len(mn))/2,(16-len(mn))/2)
        #b.clear(10,0,0,16,16)
        b.input(mn,10,0,0)
        plt.imshow(mn)
        plt.show()

        b.conduct()

        idx= nmin.axon.getValue()
        r=b.getValue(layer[idx], 16, 16)


        r=b.getValue(19, 16, 16)

        plt.imshow(r)
        plt.show()


        #r=b.getValue(20, 16, 16)
        ##print r
        #plt.imshow(r)
        #plt.show()

        for n in range(10):
            rc=b.complayer(19,n,16,16)
            ##print rc

        for n in range(10):
            rc=b.complayer(n,19,16,16)

    return





def testComplayerFeature():

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    b = brain()


    cmatV = semicircleV(3)
    b.convolution(10, 28, 28, 20, cmatV)
    cmatA = semicircleA(3)
    b.convolution(10, 28, 28, 21, cmatA)
    cmatI = [[1,1,1,1]]
    b.convolution(10, 28, 28, 22, cmatI)
    cmat_ = [[1],[1],[1],[1]]
    b.convolution(10, 28, 28, 23, cmat_)
    cmat4 = [[0,0,1,0,0],
             [0,0,1,0,0],
             [1,1,1,1,1],
             [0,0,1,0,0],
             [0,0,1,0,0]]
    b.convolution(10, 28, 28, 24, cmat4)

    allhz=hzk()
    hz16=[]
    for j in range(0, 10):
        hz = hz2array(allhz[j])
        hz16.append(hz)
        b.input(hz, j,0,0)
        mn=hz
        b.input(mn, 10, 0, 0)
        b.conduct()
        r1 = b.getValue(20, 16, 16)
        r2 = b.getValue(21, 16, 16)
        r3 = b.getValue(22, 16, 16)
        r4 = b.getValue(23, 16, 16)
        r5 = b.getValue(24, 16, 16)

        # if(r1.sum()*r2.sum()>0):
        #    #print mn
        plt.imshow(mn)
        plt.show()

    return

    for i in range(0,100):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
#        if (nb != 0):
#            continue

        mn=mn2array28( allmn[i])
        b.input(mn,10,0,0)

        b.conduct()


        r1=b.getValue(20, 28, 28)
        r2=b.getValue(21, 28, 28)
        r3=b.getValue(22, 28, 28)
        r4=b.getValue(23, 28, 28)
        r5=b.getValue(24, 28, 28)

        #if(r1.sum()*r2.sum()>0):
        #    #print mn
        plt.imshow(mn)
        plt.show()


    return


def testConvolutionEqual():
    allhz = hzk()

    cnt=23
    b = brain(2,16*cnt,16*cnt)

    cmat=[[0]*3]*3
    layer=1
    x=0
    for n in range(1,np.power(2,16)):
        cmat[0][0]=int(n & 1>0)
        cmat[0][1]=int(n & 2>0)
        cmat[0][2]=int(n & 4>0)
        cmat[1][0]=int(n & 8>0)
        cmat[1][1]=int(n & 16>0)
        cmat[1][2]=int(n & 32>0)
        cmat[2][0]=int(n & 64>0)
        cmat[2][1]=int(n & 128>0)
        cmat[2][2]=int(n & 256>0)
        #cmat[2][1]=int(n & 512>0)
        #cmat[2][2]=int(n & 1024>0)
        #cmat[2][3]=int(n & 2048>0)
        #cmat[3][0]=int(n & 4096>0)
        #cmat[3][1]=int(n & 8192>0)
        #cmat[3][2]=int(n & 16384>0)
        #cmat[3][3]=int(n & 32768>0)

        if(np.array(cmat).sum()!=3):
            continue

        i=x/cnt
        j=x%cnt

        b.input(cmat, layer, i * 16, j * 16)
        b.convolution(0,0,0, 16, 16, layer,i*16,j*16, cmat)


        x+=1
        if(x==cnt*cnt):
            x=0
            layer+=1
            break




    for j in range(0, 10):
        hz = hz2array(allhz[j])
        b.input(hz, 0,0,0)

        b.conduct()
        r = b.getValue(1, 16*cnt, 16*cnt)
        plt.imshow(r)
        plt.show()


def ConvolutionAnd(mn):

    allhz = hzk()
    cols = 10
    rows = 11
    wd = 16
    b = brain(1, wd * rows, wd * cols)

    cmat = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]  # 9/6
    # cmat=[[1],[1]]
    b.convolution(0, 0, 0, wd, wd * rows, 0, 0, wd, cmat)
    cmat = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]  # 9/6
    # cmat=[[1,1]]
    b.convolution(0, 0, 0, wd, wd * rows, 0, 0, wd * 2, cmat)
    cmat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 9/6
    b.convolution(0, 0, 0, wd, wd * rows, 0, 0, wd * 3, cmat)
    cmat = [[0, 0, 1], [0, 1, 0], [0, 1, 0]]  # 9/6
    b.convolution(0, 0, 0, wd, wd * rows, 0, 0, wd * 4, cmat)

    #    cmat=[[1,1],[1,1]] #9/6
    #    b.convolution(0, 0, 0,wd,wd, 0,wd,0, cmat)
    #    cmat=[[0,1,0],[0,1,0],[0,1,0]]#9/6
    #    b.convolution(0, wd, 0,wd,wd, 0,wd,wd, cmat)
    #    cmat=[[0,0,0],[1,1,1],[0,0,0]]#9/6
    #    b.convolution(0, wd,0,wd,wd, 0,wd,wd*2, cmat)
    #    cmat=[[1,0,0],[0,1,0],[0,0,1]]#9/6
    #    b.convolution(0, wd,0,wd,wd, 0,wd,wd*3, cmat)
    #    cmat=[[0,0,1],[0,1,0],[1,0,0]]#9/6
    #    b.convolution(0, wd,0,wd,wd, 0,wd,wd*4, cmat)

    ########
    #    hh=wd*1
    #    cmat=[[0,1,0],[0,1,0],[0,1,0]]#9/6
    #    b.convolution(0, 0+hh, 0,wd,wd*10, 0,0+hh,wd, cmat)
    #    cmat=[[0,0,0],[1,1,1],[0,0,0]]#9/6
    #    b.convolution(0, 0+hh,0,wd,wd*10, 0,0+hh,wd*2, cmat)
    #    cmat=[[1,0,0],[0,1,0],[0,0,1]]#9/6
    #    b.convolution(0, 0+hh,0,wd,wd*10, 0,0+hh,wd*3, cmat)
    #    cmat=[[0,0,1],[0,1,0],[1,0,0]]#9/6
    #    b.convolution(0, 0+hh,0,wd,wd*10, 0,0+hh,wd*4, cmat)

    #    cmat=[[1,1],[1,1]] #9/6
    #    b.convolution(0, 0+hh, 0,wd,wd, 0,wd+hh,0, cmat)
    #    cmat=[[0,1,0],[0,1,0],[0,1,0]]#9/6
    #    b.convolution(0, wd+hh, 0,wd,wd, 0,wd+hh,wd, cmat)
    #    cmat=[[0,0,0],[1,1,1],[0,0,0]]#9/6
    #    b.convolution(0, wd+hh,0,wd,wd, 0,wd+hh,wd*2, cmat)
    #    cmat=[[1,0,0],[0,1,0],[0,0,1]]#9/6
    #    b.convolution(0, wd+hh,0,wd,wd, 0,wd+hh,wd*3, cmat)
    #    cmat=[[0,0,1],[0,1,0],[1,0,0]]#9/6
    #    b.convolution(0, wd+hh,0,wd,wd, 0,wd+hh,wd*4, cmat)

    b.pooling(0, 0, 0, wd * 5, wd * rows, 0, 0, wd * 5, 2, 2)
    b.pooling(0, 0, wd * 5, wd / 2 * 5, wd * rows, 0, 0, wd * 5 + wd / 2 * 5, 2, 2)

    #   b.pooling(0,wd*3+hh,0,wd*5,wd/2,0,wd*3+hh+wd/2,0,2,2)
    #   b.pooling(0,0+hh,0,wd*5,wd,0,wd*2+hh,0,2,2)
    #   b.pooling(0,wd*2+hh,0,wd*5,wd/2,0,wd*2+wd/2+hh,0,2,2)


    for i in range(10):
        hz = hz2array(allhz[i])
        b.input(hz, 0, wd * (i + 1),0)
    # b.conduct()

    # b.reset(0,0,wd,wd*(cols-1),wd*rows)
    b.input(mn, 0, 0, 0)
    b.conduct()
    # b.reset(0, 0, wd, wd * 4, wd)
    # b.reset(0, 0, wd*5, wd/2 * 5, wd/2)
    # b.reset(0, 0, wd*5+wd/2*5, wd/4 * 5, wd/4)
    # b.neuronsconduct(0,0,0,wd*5,wd)
    # b.neuronsconduct(0, 0,wd*5, wd/2 * 5, wd/2)
    # b.neuronsconduct(0,0,0,wd,wd)

    for j in range(10):  # tz

        fd, dista = b.compare(0, 0, 0, wd, wd, 0, 0, wd * (j + 1))
        fdb, distb = b.compare(0, 0 + wd, 0, wd, wd, 0, 0 + wd, wd * (j + 1))
        fdc, distc = b.compare(0, 0 + wd * 2, 0, wd, wd, 0, 0 + wd * 2, wd * (j + 1))
        fdb, distd = b.compare(0, 0 + wd * 5, 0, wd / 2, wd / 2, 0, 0 + wd * 5, wd / 2 * (j + 1))


    r = b.getAxonMat(0, 0, 0, wd * 11, wd * 9)
    plt.imshow(r)
    plt.show()


def testConvolutionAnd():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    for i in range(10, 42):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #if (nb != 0):
        #    continue
        #print ("i=%d,nb=%d" %(i,nb))
        mn = mn2array16(allmn[i])
        ConvolutionAnd(mn)



def testDiffConvolution():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    allhz = hzk()

    cols=10
    rows=11
    wd=32
    b = brain(1,wd*rows,wd*cols)


    cmat=[[0,1,0],[0,1,0],[0,1,0],[0,1,0]]#9/6
    b.convolution(0, 0, 0,wd,wd*rows, 0,0,wd, cmat)
    cmat=[[0,0,0,0],[1,1,1,1],[0,0,0,0]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*2, cmat)
    cmat=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*3, cmat)
    cmat=[[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*4, cmat)


    b.pooling(0,0,0,wd*5,wd*rows,0,0,wd*5,2,2)
    b.pooling(0,0,wd*5,wd/2*5,wd*rows,0,0,wd*5+wd/2*5,2,2)


    for i in range(10):
        hz = hz2array(allhz[i])
        hz=diff(hz)
        ##print hz
        b.input(hz, 0,0,wd*(i+1))
    b.conduct()
    for i in range(10, 42):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #if (nb != 0):
        #    continue

        mn = mn2array16(allmn[i])
        mn=diff(mn)
        #b.reset(0,0,wd,wd*(cols-1),wd*rows)
        b.input(mn, 0,0,0)
        #b.conduct()
        b.reset(0, 0, wd, wd * 4, wd)
        b.reset(0, 0, wd*5, wd/2 * 5, wd/2)
        b.reset(0, 0, wd*5+wd/2*5, wd/4 * 5, wd/4)
        b.neuronsconduct(0,0,0,wd*5,wd)
        b.neuronsconduct(0, 0,wd*5, wd/2 * 5, wd/2)
        #b.neuronsconduct(0,0,0,wd,wd)

        for j in range(10): #tz
            fd,dista=b.compare(0,0,0,wd,wd,0,0,wd*(j+1))
            fdb,distb=b.compare(0,0+wd,0,wd,wd,0,0+wd,wd*(j+1))
            fdc,distc=b.compare(0,0+wd*2,0,wd,wd,0,0+wd*2,wd*(j+1))
            fdb,distd=b.compare(0,0+wd*5,0,wd/2,wd/2,0,0+wd*5,wd/2*(j+1))




        r = b.getAxonMat(0, 0,0,wd*11, wd*9)
        plt.imshow(r)
        plt.show()

def showneurons(brain,layer,row,col,rows,cols):
    img = brain.getMat(layer, row, col, rows,cols)
    show3dimg(img)
    return

def testConvolutionDelay():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    allhz = hzk()

    cols=5
    rows=3*3
    wd=16
    b = brain(1,wd*rows,wd*cols)


    cmat=[[1,1,1],[1,1,1],[1,1,1]]#9/6
    b.convolution(0, 0, 0,wd,wd, 0,0,wd, cmat)
    b.convolution(0, 0,wd,wd,wd, 0,0,wd*2, cmat)
    b.convolution(0, 0,wd*2,wd,wd, 0,0,wd*3, cmat)
    b.convolution(0, 0,wd*3,wd,wd, 0,0,wd*4, cmat)
    #
    b.convolution(0, wd*3, 0,wd,wd, 0,wd*3,wd, cmat)
    b.convolution(0, wd*3,wd,wd,wd, 0,wd*3,wd*2, cmat)
    b.convolution(0, wd*3,wd*2,wd,wd, 0,wd*3,wd*3, cmat)
    b.convolution(0, wd*3,wd*3,wd,wd, 0,wd*3,wd*4, cmat)
    #
    cmat=[[0,1,0],[1,1,1],[0,1,0]]#9/6
    b.convolution(0, 0, 0,wd*5,wd, 0,wd,0, cmat)
    #
    b.convolution(0, wd*3, 0,wd*5,wd, 0,wd+wd*3,0, cmat)
    #
    cmat=[[1]]
    b.convolution(0, wd, 0,wd*5,wd, 0,wd*2,0, cmat)
    #
    b.convolution(0, wd+wd*3, 0,wd*5,wd, 0,wd*2+wd*3,0, cmat)
    #
    b.diff(0,0,0,wd*5,wd*3,0,wd*3,0,0,wd*6,0)

    for i in range(10, 42):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue


        mn = mn2array16(allmn[i])
        hz=hz2array(allhz[0])
        b.reset(0,0,0,wd*cols,wd*rows)
        b.input(mn, 0, 0, 0)
        b.input(hz, 0, wd*3, 0)
        for xx in range(99):
            b.neuronsconduct(0,0,0,wd,wd)
            b.neuronsconduct(0,0,1*wd,wd,wd)
            b.neuronsconduct(0,0,2*wd,wd,wd)
            b.neuronsconduct(0,0,3*wd,wd,wd)
            b.neuronsconduct(0,0,4*wd,wd,wd)

            b.neuronsconduct(0,wd,0,wd*5,wd)
            #
            b.neuronsconduct(0,wd*3,0,wd,wd)
            b.neuronsconduct(0,wd*3,1*wd,wd,wd)
            b.neuronsconduct(0,wd*3,2*wd,wd,wd)
            b.neuronsconduct(0,wd*3,3*wd,wd,wd)
            b.neuronsconduct(0,wd*3,4*wd,wd,wd)

            b.neuronsconduct(0,wd+wd*3,0,wd*5,wd)
            #################3

            b.input(mn, 0, 0, 0)
            b.input(hz, 0, wd * 3, 0)

            #showneurons(b,0, wd,wd,wd, wd)
            r = b.getAxonMat(0, 0,0,wd*rows, wd*5)
            plt.imshow(r)
            plt.show()
            #for j in range(r.max(),r.min(),-1):
            #    #print j
            #    r=np.array([[ int(b.getNeuron(0, row, col).getValue()==j) for col in range(0,wd*5) ]for row in range(0,wd*rows)])
            #    plt.imshow(r)
            #    plt.show()

def setbrain(b,col):
    wd=16
    cmat=[[1,1,1],[1,1,1],[1,1,1]]#9/6
    b.convolution(0, 0,col+ 0,wd,wd, 0,0,col+wd, cmat)
    b.convolution(0, 0,col+wd,wd,wd, 0,0,col+wd*2, cmat)
    b.convolution(0, 0,col+wd*2,wd,wd, 0,0,col+wd*3, cmat)
    b.convolution(0, 0,col+wd*3,wd,wd, 0,0,col+wd*4, cmat)
    #
    b.convolution(0, wd*3,col+ 0,wd,wd, 0,wd*3,col+wd, cmat)
    b.convolution(0, wd*3,col+wd,wd,wd, 0,wd*3,col+wd*2, cmat)
    b.convolution(0, wd*3,col+wd*2,wd,wd, 0,wd*3,col+wd*3, cmat)
    b.convolution(0, wd*3,col+wd*3,wd,wd, 0,wd*3,col+wd*4, cmat)
    #
    cmat=[[0,1,0],[1,1,1],[0,1,0]]#9/6
    b.convolution(0, 0, col+0,wd*5,wd, 0,wd,col+0, cmat)
    #
    b.convolution(0, wd*3, col+0,wd*5,wd, 0,wd+wd*3,col+0, cmat)
    #
    cmat=[[1]]
    b.convolution(0, wd, col+0,wd*5,wd, 0,wd*2,col+0, cmat)
    #
    b.convolution(0, wd+wd*3, col+0,wd*5,wd, 0,wd*2+wd*3,col+0, cmat)
    #
    b.diff(0,0,col+0,wd*5,wd*3,0,wd*3,col+0,0,wd*6,col+0)

def testConvolutionDelayDiff():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    allhz = hzk()

    cols=5*3
    rows=3*3
    wd=16
    b = brain(1,wd*rows,wd*cols)

    setbrain(b,0)
    setbrain(b,wd*5)
    setbrain(b,wd*10)

    for i in range(10, 42):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        if (nb != 0):
            continue


        mn = mn2array16(allmn[i])
        b.reset(0,0,0,wd*cols,wd*rows)
        for xx in range(99):
            b.input(mn, 0, 0, 0)
            b.input(hz2array(allhz[0]), 0, wd * 3, 0)
            b.input(mn, 0, 0, wd*5)
            b.input(hz2array(allhz[6]), 0, wd * 3, wd*5)
            b.input(mn, 0, 0, wd*10)
            b.input(hz2array(allhz[8]), 0, wd * 3, wd*10)
            b.reset(0, wd*6, 0, wd * cols, wd * 3)
            b.conduct()


            #showneurons(b,0, wd,wd,wd, wd)
            r = b.getAxonMat(0, 0,0,wd*rows, wd*cols)
            plt.imshow(r)
            plt.show()

def adj(hz,r,c):
    i=0
    v=0
    for x in range(r-1,r+2):
        for y in range(c-1,c+1):
            if (hz[x][y]!=0):
                v-=np.power(2,i)
            i+=1
    return v

def adjoins(hz):
    w=len(hz[0])
    h=len(hz)
    for i in range(h):
        for j in range(w):
            if(hz[i][j]):
                hz[i][j]=adjcnt(hz,i,j)
    return hz

def adjcnt(hz,r,c):
    return int(hz[r-1][c]!=0)+int(hz[r+1][c]!=0)+int(hz[r][c-1]!=0)+int(hz[r][c+1]!=0)

def adjoincnts(hz):
    w=len(hz[0])
    h=len(hz)
    #adjcnts=[0,1,2,3,4,5,6,7,8]
    adjcnts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(h):
        for j in range(w):
            if(hz[i][j]):
                adjcnts[adjcnt(hz,i,j)]+=1
    return adjcnts

def testAdjoins():
    allhz = hzk()


    cols=5
    rows=3
    wd=16
    b = brain(1,wd*rows,wd*cols)


    cmat=[[1,1,1],[1,1,1],[1,1,1]]#9/6
    b.convolution(0, 0, 0,wd,wd, 0,0,wd, cmat)
    b.convolution(0, 0,wd,wd,wd, 0,0,wd*2, cmat)
    b.convolution(0, 0,wd*2,wd,wd, 0,0,wd*3, cmat)
    b.convolution(0, 0,wd*3,wd,wd, 0,0,wd*4, cmat)
    cmat=[[0,1,0],[1,1,1],[0,1,0]]#9/6
    b.convolution(0, 0, 0,wd*4,wd, 0,wd,0, cmat)
    cmat=[[1]]
    #b.convolution(0, wd, 0,wd*4,wd, 0,wd*2,0, cmat)


    mn = hz2array(allhz[0])
    mn=adjoins(mn)


    b.reset(0, 0, 0, wd * cols, wd * rows)
    for xx in range(99):

        b.input(mn, 0, 0, 0)

        b.neuronsconduct(0, 0, 0, wd, wd)
        b.neuronsconduct(0, 0, 1 * wd, wd, wd)
        b.neuronsconduct(0, 0, 2 * wd, wd, wd)
        b.neuronsconduct(0, 0, 3 * wd, wd, wd)
        b.neuronsconduct(0, 0, 4 * wd, wd, wd)

        b.neuronsconduct(0, wd, 0, wd * 4, wd)

        showneurons(b, 0, 0, 0, wd, wd)

        r = b.getMat(0, 0, 0, wd * rows, wd * 5)
        plt.imshow(r)
        plt.show()


def testConvolutionFeature():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    allhz = hzk()

    cols=10
    rows=11
    wd=16
    b = brain(1,wd*rows,wd*cols)


    cmat=[[0,1,0],[0,1,0],[0,1,0]]#9/6
    b.convolution(0, 0, 0,wd,wd*rows, 0,0,wd, cmat)
    cmat=[[0,0,0],[1,1,1],[0,0,0]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*2, cmat)
    cmat=[[1,0,0],[0,1,0],[0,0,1]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*3, cmat)
    cmat=[[0,0,1],[0,1,0],[0,1,0]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*4, cmat)
    cmat=[[0,1,0],[1,0,1],[0,1,0]]#9/6
    b.convolution(0, 0,0,wd,wd*rows, 0,0,wd*5, cmat)



    for i in range(10):
        hz = hz2array(allhz[i])
        b.input(hz, 0,wd*(i+1),0)
    #b.conduct()
    for i in range(10, 42):
        nb = int(np.dot(ys[i], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        #if (nb != 0):
        #    continue

        mn = mn2array16(allmn[i])
        #b.reset(0,0,wd,wd*(cols-1),wd*rows)
        b.input(mn, 0,0,0)
        b.conduct()

        for j in range(10): #tz

            fd,dista=b.compare(0,0,0,wd,wd,0,0,wd*(j+1))
            fdb,distb=b.compare(0,0+wd,0,wd,wd,0,0+wd,wd*(j+1))
            fdc,distc=b.compare(0,0+wd*2,0,wd,wd,0,0+wd*2,wd*(j+1))
            fdb,distd=b.compare(0,0+wd*5,0,wd/2,wd/2,0,0+wd*5,wd/2*(j+1))


        r = b.getAxonMat(0, 0,0,wd*11, wd*9)
        plt.imshow(r)
        plt.show()


def testFilter0():
    allhz = hzk()
    wd=16
    b = brain(1,16*2,16*5)
    for j in range(0, 1):
        hz = hz2array(allhz[j])
        b.input(hz, j,0,0)

    cmat=[[0,1,1,0]]
    b.filter(0, 0,0,16, 16, 0,0,16, cmat)
    cmat=[[0],[1],[0]]
    b.filter(0, 0,0,16, 16, 0,0,16*2, cmat)
    cmat=[[1]]
    b.convolution(0, 0,16,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*2,16, 16, 0,16,0, cmat)

    b.conduct()

    #b.neuronsconduct(0,0,0,16,16)
    r = b.getAxonMat(0, 0, 0, wd * 2, wd * 3)
    plt.imshow(r)
    plt.show()

def testFilter():
    wd = 16
    b = brain(1, 16 * 3, 16 * 5)

    allhz = hzk()
    for j in range(0, 1):
        hz = hz2array(allhz[j])
        b.input(hz, j,0,0)

    #mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    #allmn, ys = inputmn(mnist)
    #mn = mn2array16(allmn[10])
    #b.input(mn, 0, 0, 0)

    cmat = [[0, 1, 1, 0]]
    b.filter(0, 0, 0, 16, 16, 0, 0, 16 * 1, cmat)
    b.defilter(0, 0, 16 * 1, 16, 16, 0, 16, 16, cmat)
    cmat = [[0], [1], [1], [0]]
    b.filter(0, 0, 0, 16, 16, 0, 0, 16 * 2, cmat)
    b.defilter(0, 0, 16 * 2, 16, 16, 0, 16, 16 * 2, cmat)
    cmat = [[0, 1, 0]]
    b.filter(0, 0, 0, 16, 16, 0, 0, 16 * 3, cmat)
    b.defilter(0, 0, 16 * 3, 16, 16, 0, 16, 16 * 3, cmat)
    cmat = [[0], [1], [0]]
    b.filter(0, 0, 0, 16, 16, 0, 0, 16 * 4, cmat)
    b.defilter(0, 0, 16 * 4, 16, 16, 0, 16, 16 * 4, cmat)
    cmat = [[1]]
    b.convolution(0, 0, 16 * 1, 16, 16, 0, 16, 0, cmat)
    b.convolution(0, 0, 16 * 2, 16, 16, 0, 16, 0, cmat)
    b.convolution(0, 0, 16 * 3, 16, 16, 0, 16, 0, cmat)
    b.convolution(0, 0, 16 * 4, 16, 16, 0, 16, 0, cmat)

    b.convolution(0, 16, 16 * 1, 16, 16, 0, 16 * 2, 0, cmat)
    b.convolution(0, 16, 16 * 2, 16, 16, 0, 16 * 2, 0, cmat)
    b.convolution(0, 16, 16 * 3, 16, 16, 0, 16 * 2, 0, cmat)
    b.convolution(0, 16, 16 * 4, 16, 16, 0, 16 * 2, 0, cmat)

    b.conduct()

    # b.neuronsconduct(0,0,0,16,16)
    # b.neuronsconduct(0,16,0,16,16)

    r = b.getAxonMat(0, 0, 0, wd * 3, wd * 5)
    plt.imshow(r)
    plt.show()

def OldHandwritingExtraction():
    wd=16
    b = brain(1,16*3,16*5)

    #allhz = hzk()
    #for j in range(0, 1):
    #    hz = hz2array(allhz[j])
    #    b.input(hz, j,0,0)

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)
    mn = mn2array16(allmn[10])
    b.input(mn,0,0,0)

    cmat=[[0,1,1,0]]
    b.filter(0, 0,0,16, 16, 0,0,16*1, cmat)
    b.defilter(0, 0,16*1,16, 16, 0,16,16, cmat)
    cmat=[[0],[1],[1],[0]]
    b.filter(0, 0,0,16, 16, 0,0,16*2, cmat)
    b.defilter(0, 0,16*2,16, 16, 0,16,16*2, cmat)
    cmat=[[0,1,0]]
    b.filter(0, 0,0,16, 16, 0,0,16*3, cmat)
    b.defilter(0, 0,16*3,16, 16, 0,16,16*3, cmat)
    cmat=[[0],[1],[0]]
    b.filter(0, 0,0,16, 16, 0,0,16*4, cmat)
    b.defilter(0, 0,16*4,16, 16, 0,16,16*4, cmat)
    cmat=[[1]]
    b.convolution(0, 0,16*1,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*2,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*3,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*4,16, 16, 0,16,0, cmat)

    b.convolution(0, 16,16*1,16, 16, 0,16*2,0, cmat)
    b.convolution(0, 16,16*2,16, 16, 0,16*2,0, cmat)
    b.convolution(0, 16,16*3,16, 16, 0,16*2,0, cmat)
    b.convolution(0, 16,16*4,16, 16, 0,16*2,0, cmat)

    b.conduct()

    #b.neuronsconduct(0,0,0,16,16)
    #b.neuronsconduct(0,16,0,16,16)

    r = b.getAxonMat(0, 0, 0, wd * 3, wd * 5)
    plt.imshow(r)
    plt.show()


def Adj(mn):
    wd = 16
    b = brain(1, 16 * 4, 16 * 4)
    b.adj(0, 0, 0, 16, 16, 0, 0, 16)

    for i in range(3, 40):
        # b.reset(0,0,0,wd*4,wd*4)
        for n in neuron.neuronset:
            n.value = 0

        b.input(mn, 0, 0, 0)
        b.conduct()
        ##print mn
        ##print ("-----"
        #r = b.getMat(0, 0, 0, wd * 2, wd)
        ##print r
        r = b.getAxonMat(0, 0, 0, wd * 3, wd * 4)
        plt.imshow(r)
        plt.show()

def HandwritingExtraction(mn):
    wd = 16
    b = brain(1, 16 * 4, 16 * 4)
    b.Extraction(0, 0, 0, 16, 16, 0, 16, 0)

    for n in neuron.neuronset:
        ##print n.xyz, n.value
        n.value = 0

    b.input(mn, 0, 0, 0)
    b.conduct()
    return b.getAxonMat(0, 16, 0, wd , wd )

def testHandwritingExtraction():
    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)

    mn = mn2array16(allmn[10])
    r = HandwritingExtraction(mn)
    plt.imshow(r)
    plt.show()
    ConvolutionAnd(r)


def testEdge():
    wd=16
    b = brain(1,16*3,16*5)

    #allhz = hzk()
    #for j in range(0, 1):
    #    hz = hz2array(allhz[j])
    #    b.input(hz, j,0,0)

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)
    mn = mn2array16(allmn[10])
    b.input(mn,0,0,0)

    cmat=[[0,1,1],[0,1,1],[0,1,1]]
    b.filter(0, 0,0,16, 16, 0,0,16*1, cmat)
    b.defilter(0, 0,16*1,16, 16, 0,16,16, cmat)
    cmat=[[1,1,0],[1,1,0],[1,1,0]]
    b.filter(0, 0,0,16, 16, 0,0,16*2, cmat)
    b.defilter(0, 0,16*2,16, 16, 0,16,16*2, cmat)
    cmat=[[0,0,0],[1,1,1],[1,1,1]]
    b.filter(0, 0,0,16, 16, 0,0,16*3, cmat)
    b.defilter(0, 0,16*3,16, 16, 0,16,16*3, cmat)
    cmat=[[1,1,1],[1,1,1],[0,0,0]]
    b.filter(0, 0,0,16, 16, 0,0,16*4, cmat)
    b.defilter(0, 0,16*4,16, 16, 0,16,16*4, cmat)
    cmat=[[1]]
    b.convolution(0, 0,16*1,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*2,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*3,16, 16, 0,16,0, cmat)
    b.convolution(0, 0,16*4,16, 16, 0,16,0, cmat)

    b.convolution(0, 16,16*1,16, 16, 0,16*2,0, cmat)
    b.convolution(0, 16,16*2,16, 16, 0,16*2,0, cmat)
    b.convolution(0, 16,16*3,16, 16, 0,16*2,0, cmat)
    b.convolution(0, 16,16*4,16, 16, 0,16*2,0, cmat)

    b.conduct()

    #b.neuronsconduct(0,0,0,16,16)
    #b.neuronsconduct(0,16,0,16,16)

    r = b.getAxonMat(0, 0, 0, wd * 3, wd * 5)
    plt.imshow(r)
    plt.show()



def testPooling():
    allhz = hzk()
    hz=hz2array(allhz[0])
    hz=hz-1
    hz=hz*hz
    ##print hz

    b = brain(1,16*5,16*2)
    b.input(hz, 0, 0, 0)

    #cmat=[[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
    #b.convolutionfun(0, 0,0,16, 16, 12,0,0, cmat,actvand)

    b.pooling(0,0,0,16,16,0,0,16,4,4)

    b.pooling(0,0,0,16,16,0,16,0,2,2)
    b.pooling(0,16,0,16,16,0,16*2,0,2,2)

    b.conduct()
    r = b.getAxonMat(0,0,0,16*3,16*2)
    plt.imshow(r)
    plt.show()

def getLine(angle):
    a=np.tan(angle)
    wd=15
    o=(wd)/2
    hz = np.array([[0] * wd] * wd)
    for x in range(-o,o):
        y =  int(a*x)
        if (y < o and y >= -o):
            hz[y+o][x+o] = 1
    for y in range(-o,o):
        x =  int(y/a)
        if (x < o and x >= -o):
            hz[y+o][x+o] = 1

    return hz

def getArc(a,b,angle1,angle2):
    if(a>b):
        wd=a*2+1
    else:
        wd=b*2+1
    o=(wd)/2
    hz = np.array([[0] * wd] * wd)
    for angle in range(angle1,angle2):
        t=np.pi*angle/180
        x=int(round(b*np.sin(t)))
        y=int(round(a*np.cos(t)))
        hz[x+o][y+o]=1


    return hz


def testLine():
    for a in range(1,180,15):
        hz=getLine(np.pi*a/180)
        plt.imshow(hz)
        plt.show()

def testArc():
    for a in range(8,1,-1):
        for b in range(8,1,-1):
            hz=getArc(a,b,180,360)
            plt.imshow(hz)
            plt.show()

def Bezier(x,t):#3次 Bezier
    xt = 1.0/6*(
        (-x[0] + 3 * x[1] -3* x[2]+x[3]) * t * t*t +\
         (3*x[0]-6*x[1]+3*x[2])*t*t+\
         (-3 * x[0] + 3 * x[2]) * t +\
         (x[0]+4*x[1]+x[2])
    )
    return xt



def getBezier(points):
    wd=max(max(points))
    hz = np.array([[0] * wd] * wd)
    segs=len(points)-3
    for m in range(segs):
        n=wd*3
        x,y=zip(*points[m:m+4])
        #hz[x[0]][y[0]]=1
        for tt in range(n):
            t=1.0*tt/n
            xt = Bezier(x,t)
            yt = Bezier(y, t)
            hz[int(xt)][int(yt)]=1
    return hz

def Bezier2(x,yt,t):  #2次Bezier
    xt = x[0]*yt*yt+x[1]*2*yt*t+x[2]*t*t
    return xt

def getBezier2(points):
    wd=max(max(points))*2
    hz = np.array([[0] * wd] * wd)
    x, y = zip(*points)
    hz[x[0]][y[0]]=1
    rate=wd*2
    for tt in range(rate):
        t=1.0*tt/rate
        yt=1-t
        xt=Bezier2(x,yt,t)
        yt=Bezier2(y,yt,t)
        ##print t,xt,yt
        try:
            hz[int(yt)][int(xt)]=1
        except:
            print (wd,yt,xt)
    ##print hz
    hz=center(hz)
    ##print -1
    ##print hz
    return hz



def testBezier():
    for n in range(16,8,-1):
        for b in range(n+1,16+1):
            a=n
            #p = [[0, 0], [0, b], [a, b], [a, 0], [0, 0], [0, b], [a, b], [a, 0]]
            #hz=getBezier(p)
            #plt.imshow(hz)
            #plt.show()

            #
            for x in range(a):
                p = [[0, 0], [0, b], [a, b], [x, 0], [0, 0], [0, b], [a, b]]
                hz=getBezier(p)
                plt.imshow(hz)
                plt.show()
            #

            #p = [[0, 0], [0, a], [b, a], [b, 0], [0, 0], [0, a], [b, a], [b, 0]]
            #hz=getBezier(p)
            #plt.imshow(hz)
            #plt.show()


def testBezierFilter():
    wd = 16

    allhz = hzk()
    hz = hz2array(allhz[9])
    hz = Convolution(hz, [[1, 1], [1,  1]])
    #hz=Convolution(hz,[[1,1,1],[1,1,1],[1,1,1]])


    for n in range(16,5,-1):
        for b in range(n+1,16+1):
            a=n
            #p = [[0, 0], [0, b], [a, b], [a, 0], [0, 0], [0, b], [a, b], [a, 0]]
            p = [[0, 0], [0, b], [a, b], [a, 0], [0, 0], [0, b], [a, b]]
            cmat=getBezier(p)
            br = brain(1, 16 * 3, 16 * 5)
            br.input(hz, 0, 0, 0)
            br.input(cmat,0,16,0)
            br.convolution(0, 0, 0, 16, 16, 0, 0, 16 * 1, cmat)
            br.conduct()
            r = br.getAxonMat(0, 0, wd, wd , wd )
            if(r.sum()>0):
                r = br.getAxonMat(0, 0, 0, wd * 2, wd * 2)
                plt.imshow(r)
                plt.show()

            p = [[0, 0], [0, a], [b, a], [b, 0], [0, 0], [0, a], [b, a], [b, 0]]
            cmat=getBezier(p)
            br = brain(1, 16 * 3, 16 * 5)
            br.input(hz, 0, 0, 0)
            br.input(cmat,0,16,0)
            br.convolution(0, 0, 0, 16, 16, 0, 0, 16 * 1, cmat)
            br.conduct()
            r = br.getAxonMat(0, 0, wd, wd , wd )
            if(r.sum()>0):
                r = br.getAxonMat(0, 0, 0, wd * 2, wd * 2)
                plt.imshow(r)
                plt.show()

def testBezier2():
    for n in range(16,5,-1):
        for b in range(n+1,16+1):
            a=n
            p = [[0, 0], [a/2, b], [a, 0]]
            mat=getBezier2(p)
            plt.imshow(mat)
            plt.show()
            p = [[0, 0], [b/2, a], [b, 0]]
            mat=getBezier2(p)
            plt.imshow(mat)
            plt.show()

def testBezier2Filter():
    wd = 16

    allhz = hzk()
    hz = hz2array(allhz[0])
    hz = Convolution(hz, [[1, 1], [1,  1]])
    #hz=Convolution(hz,[[1,1,1],[1,1,1],[1,1,1]])


    for n in range(16,6,-1):
        for b in range(n+1,16+1):
            a=n
            #p = [[0, 0], [0, b], [a, b], [a, 0], [0, 0], [0, b], [a, b], [a, 0]]
            p = [[0, b], [a/2, 0], [a, b]]
            cmat=getBezier2(p)
            br = brain(1, 16 * 3, 16 * 5)
            br.input(hz, 0, 0, 0)
            br.input(cmat,0,16,0)
            br.convolution(0, 0, 0, 16, 16, 0, 0, 16 * 1, cmat)
            br.conduct()
            r = br.getAxonMat(0, 0, wd, wd , wd )
            if(r.sum()>0):
                r = br.getAxonMat(0, 0, 0, wd * 2, wd * 2)
                plt.imshow(r)
                plt.show()

            p = [[0, a], [b/2, 0], [b, a]]
            cmat=getBezier2(p)
            br = brain(1, 16 * 3, 16 * 5)
            br.input(hz, 0, 0, 0)
            br.input(cmat,0,16,0)
            br.convolution(0, 0, 0, 16, 16, 0, 0, 16 * 1, cmat)
            br.conduct()
            r = br.getAxonMat(0, 0, wd, wd , wd )
            if(r.sum()>0):
                #print r
                r = br.getAxonMat(0, 0, 0, wd * 2, wd * 2)
                plt.imshow(r)
                plt.show()




def testArcFilter():
    wd = 16

    allhz = hzk()
    hz = hz2array(allhz[0])
    hz=Convolution(hz,[[1,1,1],[1,1,1],[1,1,1]])


    for a in range(8,2,-1):
        for b in range(8,2,-1):
            cmat=getArc(a,b,180,360)
            #print cmat
            #br.clear(0,0,0,wd*2,wd*2)
            br = brain(1, 16 * 3, 16 * 5)
            br.input(hz, 0, 0, 0)
            br.input(cmat,0,16,0)
            br.convolution(0, 0, 0, 16, 16, 0, 0, 16 * 1, cmat)
            br.conduct()
            r = br.getMat(0, 0, wd, wd , wd )
            #print r
            r = br.getAxonMat(0, 0, 0, wd * 2, wd * 2)
            plt.imshow(r)
            plt.show()

def testRemember(br):
    allhz = hzk()
    for i in range(10):
        hz = hz2array(allhz[i])
        br.remember(hz,i)

def testRecognize(br):
    #allhz = hzk()
    #img = hz2array(allhz[3])

    mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    allmn,ys=inputmn(mnist)
    r=0
    for i in range(100):
        img = mn2array16(allmn[i])
        nb=int(np.dot(ys[i],[0,1,2,3,4,5,6,7,8,9]))
        try:
            re=br.recognize(img)
            #print img
            #print nb,re
            if(nb==re):r=r+1
        except:
            continue

    #print r

def zjj(u,v,x,y,M,N):
    if( (((u) * x*N + (v) * y*M)%(M*N))*2<M*N):  #(ux/M+vb/N)*2*PI
        R=-1
        I=1
    else:
        R=1
        I=-1
#    T=M/(u+1)
#    if((x%T)>T):#cos
#        R=1;I=-1
#    else:
#        R=-1;I=1

#    T=N/(v+1)
#    if((y%T)*2>T):#sin
#        I=-1
#    else:
#        I=1
#    ys=(((u) * x*N + (v) * y*M)%(M*N))*2
#    R=np.sin(ys/(M*N)*np.pi)
#    I=np.cos(ys/(M*N)*np.pi)

    ##print u,v,x,y,R,I

    #R=np.cos(2*np.pi*(1.0*u*x/M+1.0*v*y/N))
    #I=np.sin(2*np.pi*(1.0*u*x/M+1.0*v*y/N))
    #return complex(0,2*np.pi*(1.0*u*x/M+1.0*v*y/N))
    return R,I



def DFT(img):
    row=len(img)
    col=len(img[0])
    f=np.array([[complex(0,0)]*col]*row)
    for u in range(row):
        for v in range(col):
            R=0
            I=0
            for x in range(row):
                for y in range(col):
                    R,I=zjj(u,v,x,y,row,col)
                    ##print u,v,x,y,R,I
                    f[u, v] += img[x,y] * complex(R,-I)
                    ##print f[u,v]
#            R += img[x, y] * (np.cos(2 * np.pi * (u * x + v * y)))
#            I += img[x, y] * (np.sin(2 * np.pi * (u * x + v * y)))
#            if v<=n/2 : vv=v+n/2
#            else:   vv=v-(n+1)/2
#            if u<=n/2 : uu=u+n/2
#            else:   uu=u-(n+1)/2
#            #print R,I,v,u,vv,uu

            #f[u,v]=f[u,v]/row/col/2
    return f

def IDFT(X):
    row = len(X)
    col =   len(X[0])
    f = np.array([[0] * col] * row)
    for x in range(row):
        for y in range(col):
            xy=complex(0,0)
            for u in range(row):
                for v in range(col):
                    R,I=zjj(u,v,x,y,row,col)
                    xy+=X[u,v]*complex(R,I)
            f[x,y]=abs(xy)/row/col/2
            #print xy
    return f

def testDFT():
    row=3
    col=3
    img=np.array([[0]*col]*row)

#    img[0,0]=1
#    img[0,1]=1
#    img[1, 0] = 1
#    img[1, 1] = 1

#    img[0:9,0:4]=1
#    img[0:9,1:2]=1
#    img[0:1,0:9]=1
    img[2:3,0:3]=1
#    img[0:8,0:1]=1
#    img[6:7,0:9]=1
#    img[8:9,0:9]=1

    allhz = hzk()
    img = hz2array(allhz[3])
    #img = np.array([[0] * 32] * 32)
    #img=np.resize(hz,(32,32))

    plt.imshow(img)
    plt.show()
    #f=np.fft.fft2(img)
    f=DFT(img)
    #print f
    plt.imshow(abs(f))
    plt.show()
    #d=np.fft.ifft2(f)
    d = IDFT(f)
    #print d
    plt.imshow(abs(d))
    plt.show()

def NDFT(img):
    br = brain()

def testNDFT():
    row=3
    col=3
    img=np.array([[0]*col]*row)

#    img[0,0]=1
    img[0,1]=1
    img[0, 2] = 1
#    img[1, 1] = 1

#    img[0:9,0:4]=1
#    img[0:9,1:2]=1
#    img[0:1,0:9]=1
#    img[2:3,0:2]=1
#    img[0:8,0:1]=1
#    img[6:7,0:9]=1
#    img[8:9,0:9]=1

    br = brain(3, 16 , 16 )
    #print img
    br.input(img, 0, 0, 0)
    for u in range(row):
        # NDFT ROW
        for v in range(col):
            #NDFT ITEM
            nX=br.getNeuron(1,u,v)
            for c in range (0,col,v+1):
                nx = br.getNeuron(0, u,  c)
                nx.connectto(nX)

    for u in range(row):
        # NDFT ROW
        for v in range(col):
            #NDFT ITEM
            nX=br.getNeuron(1,u,v)
            for c in range(0,col,v+1):
                nx = br.getNeuron(2, u,  c)
                nx.connectfrom(nX)

    br.conduct()

    img=br.getMat(1,0,0,row,col)
    #print img

    plt.imshow(img)
    plt.show()



    img=br.getMat(2,0,0,row,col)
    #print img

    plt.imshow(img)
    plt.show()


def testGetMatFilter(br):
    def f(v):
        if v==0:return 1
        else:return 0
    #print br.getMatFilter(br.knowledges[0].layer,0,0,16,16,f)


if __name__ == "__main__":
    #testConvolution()
    #testRotate()
    #testSelect()
    #testComplayer()
    #testFeature()
    #testComplayerFeature()
    #testConvolutionEqual()
    #testDiff()
    #testDiffConvolution()
    #testConvolutionDelay()
    #testConvolutionDelayDiff()
    #testAdjoins()
    #testConvolutionFeature()
    #testFilter()
    # testEdge()
    #testConvolutionAnd()
    #testHandwritingExtraction()
    #testPooling()
    #testLine()
    #testArc()
    #testArcFilter()
    #testBezier()
    #testBezierFilter()
    #testBezier2()
    #testBezier2Filter()
    #testDFT()
    #testNDFT()



    #testRemember(br)
    allhz=hzk()
    img=hz2img(allhz[0])
    la = brain.BrainLayer(16, 16)

    lb=brain.BrainLayer(16,16)
    la.rotate(lb,5,7,7)
    lc=brain.BrainLayer(16,16)
    la.rotate(lc,90,7,7)

    la.input(img, 0, 0)
    m=lb.getMat()
    #la.input(m)#you leiji wucha
    m=lb.getMat()
    plt.imshow(m)

    m=lc.getMat()
    #plt.imshow(m)
    plt.show()
    c2d=FCompare2D()
    r=c2d.like(img,m)
    print(r)


    #testGetMatFilter(br)
    #testRecognize(br)
