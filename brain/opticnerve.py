#-*-coding:utf-8-*-
import numpy as np
from neuron import *
'''
2.促发记忆：这个记忆类型可能我们平时听的不多，但是也非常有用。例如我们读一段文字：
“今年有很多节日，春节、清明节，我们都非常喜欢节日，每次过节的时候节目也很多”是不是
后面的那个“节目”不注意很容易看错，看成“节日”？这个就是促发记忆的效果，它的作用就是
加速识别，不需要每次都完整调用大脑搜索机制来判别这个词语是什么。识别图像也是，如果
先看某个类型的图像，以后就容易把相似的图像识别为先前看到的。例如识别人脸。这个作用是
非常强大的，如果促发记忆出现了问题，那么快速阅读根本就是不可能的。促发记忆主要由大
脑皮层控制。

作者：宋文峰
链接：https://www.jianshu.com/p/d6aa6501c3e4
來源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
'''
COLS=28
ROWS=28

import matplotlib.pyplot as plt
def pltshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()


class opticnerve:
    def __init__(self):
        self.neurons=np.array([[neuron() for ii in range(COLS)]
                  for jj in range(ROWS)])# brains
        self.pallium=[] # 促发记忆主要由大脑皮层控制
        self.knowledges=[]
        self.threshold = 1

    def train(self,img,label):
        pass

    def remember(self,img,label):
        idx,vmax=self.look(img)
        #print(vmax)
        if(vmax/(ROWS*COLS)>=self.threshold):
            if (self.knowledges[idx] == label):
                #print(label,"Already in mermory!",vmax,vmax/786)
                #n=self.pallium[idx]
                return
            else:
                print("New", label, self.knowledges[idx],vmax,vmax/786)
                pltshow(img)
                n = neuron()
                self.pallium.append(n)
                self.knowledges.append(label)
        else:
            n = neuron()
            self.pallium.append(n)
            self.knowledges.append(label)

        r,c=img.shape
        for i in range(r):
            for j in range(c):
                if(img[i][j]!=0):
                    n.connectfrom(self.neurons[i,j],1)
                else:
                    n.connectfrom(self.neurons[i, j], 0)


    def center(self,mn):
        left = 0;
        right = 29;
        top = 0;
        bottom = 29
        for n in range(28):
            if (mn[n, :].max() > 0 and left == 0):
                left = n
            if (mn[-n, :].max() > 0 and right == 29):
                right = 29 - n
            if (mn[:, n].max() > 0 and top == 0):
                top = n
            if (mn[:, -n].max() > 0 and bottom == 29):
                bottom = 29 - n

        new = np.zeros((ROWS,COLS),np.uint8)
        nleft=(28-(right-left))//2
        nright=nleft+(right-left)
        ntop=(28-(bottom-top))//2
        nbottom=ntop+(bottom-top)
        new[nleft:nright,ntop:nbottom] = mn[left:right,top:bottom]
        return new

    def look(self,img):
        #img=self.center(img)
        #pltshow(img)
        for n in self.pallium:
            n.value=0
        r,c=img.shape
        for i in range(r):
            for j in range(c):
                self.neurons[i,j].value = img[i,j]
                self.neurons[i, j].conduct()
        idx=-1;vmax=0
        for i in range(len(self.pallium)):
            #print(self.pallium[i].value)
            if(self.pallium[i].value>vmax):
                idx=i;vmax=self.pallium[i].value
        return idx,vmax

    def predict(self,img):
        idx,vmax=self.look(img)
        return self.knowledges[idx],vmax*100//(ROWS*COLS)


if __name__ == "__main__":
    from hzk import *
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
    #print(i1)


    on = opticnerve()
    on.remember(i0, '0')
    on.remember(i1, '1')
    on.remember(i2, '2')
    on.remember(i3, '3')
    on.remember(i4, '4')
    on.remember(i5, '5')
    on.remember(i6, '6')
    on.remember(i7, '7')
    on.remember(i8, '8')
    on.remember(i9, '9')

    #print(on.knowledges)
    #print(on.neurons[0,0].axon.synapses)
    #print(len(on.pallium[0].dendritic.synapses))
    lb=on.predict(i2)
    print(lb)

    #import tensorflow.examples.tutorials.mnist.input_data as input_data

    #mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    #batch_xs, batch_ys = mnist.train.next_batch(100)

