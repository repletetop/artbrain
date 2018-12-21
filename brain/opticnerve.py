# -*-coding:utf-8-*-
import numpy as np
import shelve

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
# COLS=28
# ROWS=28
# NEURONSCOUNT = ROWS*COLS

import matplotlib.pyplot as plt


def pltshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()

class pallium:#局部回路神经元及抑制
    def __init__(self):
        self.neurons=[]
        self.maxvalue=0

class opticnerve:
    def __init__(self, ROWS, COLS):
        self.neurons = np.array([[neuron() for ii in range(COLS)]
                                 for jj in range(ROWS)])  # brains
        self.pallium=[] # 促发记忆主要由大脑皮层控制
        self.knowledges = {}
        self.dendritics = []
        self.COLS = COLS
        self.ROWS = ROWS
        self.NEURONSCOUNT = ROWS * COLS
        # self.threshold = int(0.01*NEURONSCOUNT)
        # self.threshold = 1

    def save(self):  #
        d = shelve.open("opticnv")
        # d['neurons']=self.neurons
        d['knowledges'] = self.knowledges
        d['dendritics'] = self.dendritics
        d.close()

    def load(self):
        f = shelve.open("opticnv")
        if len(f.dict.keys()) > 0:
            self.neurons = f.get('neurons')
            self.knowledges = f.get('knowledges')
            self.dendritics = f.get('dendritics')
        f.close()



    def clearneurons(self):
        for i in range(self.ROWS):
            for j in range(self.COLS):
                self.neurons[i,j].value = 0

    def reset(self):
        self.clearneurons()

        for k in self.knowledges:
            self.knowledges[k].value = 0
            self.knowledges[k].dendritic.value=0
        for n in self.pallium:
            n.value=0
            n.dendritic.value=0

        self.positive = []

        self.keys = list(self.knowledges.keys())
        self.values = list(self.knowledges.values())

    def center(self, mn):
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

        new = np.zeros((self.ROWS, self.COLS), np.uint8)
        nleft = (28 - (right - left)) // 2
        nright = nleft + (right - left)
        ntop = (28 - (bottom - top)) // 2
        nbottom = ntop + (bottom - top)
        new[nleft:nright, ntop:nbottom] = mn[left:right, top:bottom]
        return new


    def look(self, img):  # get max overlap
        # img=self.center(img)
        # pltshow(img)
        self.reset()
        r, c = img.shape
        for i in range(r):
            for j in range(c):
                # if img[i, j]!=0:
                self.neurons[i, j].value = img[i, j]
                self.neurons[i, j].conduct()
        # hengxiang yizhi
        #self.positive = []
        vmax = -1
        nmax = None
        # self.tmp=[]
        for n in self.pallium:
            n.calcValue()
            #if(n.dendritic.value==len(n.dendritic.synapses)):
            #    self.positive.append(n)
            n.conduct()
            if vmax < n.value:
                vmax = n.value
                nmax = n
        return nmax
        #if nmax!=None and nmax.dendritic.value/len(nmax.dendritic.synapses)>0.6:
        #    return nmax  # max overlap
        #return None


    def remember(self, img, label):
        nmax = self.look(img)  # found mutipy?
        if (nmax!=None):
            lb=nmax.axon.outneurons[0].label
            if lb==label:
                return

        if (self.knowledges.__contains__(label)):
            # print("Already in,create dendritic only", label)
            nlb = self.knowledges[label]
        else:
            nlb = neuron()
            #self.pallium.append(nlb)
            nlb.label=label
            self.knowledges[label] = nlb
        #how to create tree

        npt = neuron()
        self.pallium.append(npt)
        dnpt=npt.dendritic
        nn = neuron()
        self.pallium.append(nn)
        dn=nn.dendritic

        n = neuron()
        self.pallium.append(n)
        nlb.inaxon.append(n.axon)
        n.axon.outneurons.append(nlb)
        d=n.dendritic
        d.connectfrom(npt.axon,1)
        d.connectfrom(nn.axon,1)


        #imgleft=self.output()
        #if(imgleft.sum()==0 and len(self.positive)>0):
        #    for pn in self.positive:
        #        dnpt.connectfrom(pn.axon, 1)
        #        #pn.recall()
        #        #print(self.output())
        #    r, c = img.shape
        #    for i in range(r):
        #        for j in range(c):
        #            if (img[i][j] == 0):  # negative
        #                dn.connectfrom(self.neurons[i, j].axon, -1)
        #    return

        r, c = img.shape
        for i in range(r):
            for j in range(c):
                if (img[i][j] > 0):#positive
                    dnpt.connectfrom(self.neurons[i, j].axon, 1)
                else:#negative
                    dn.connectfrom(self.neurons[i, j].axon, -1)
    def think(self):
        imax=0
        for i in range(self.ROWS):
            for j in range(self.COLS):
                if imax<len(self.neurons[i,j].axon.synapses):
                    imax = len(self.neurons[i, j].axon.synapses)
                #print(i,j,len(self.neurons[i, j].axon.synapses))

                if len(self.neurons[i, j].axon.synapses)<10:#
                    continue
                isFirst=True
                for s in self.neurons[i, j].axon.synapses:
                    ns=[]
                    for ds in s.dendritic.synapses:
                        ns.append(ds.axon) # one dendritic
                    if isFirst:
                        isFirst=False
                        intersection=set(ns)
                    else:
                        intersection = intersection & set(ns)

                if(len(intersection)>10):
                    #print("intersction len:",len(intersection))
                    n=neuron()
                    self.pallium.append(n)
                    for axon in intersection:
                        n.dendritic.connectfrom(axon,1)

                    for s in self.neurons[i,j].axon.synapses:
                        for ds in s.dendritic.synapses:#every dendritic
                            if ds.axon in intersection:
                                s.dendritic.synapses.remove(ds)
                        s.dendritic.connectfrom(n.axon,1)

        #print(imax)


#synapses: 112398
#After think:
#synapses: 83982



    def remember_org(self, img, label):#train  500  Right: 83 100 83 %
        nmax = self.look(img)  # found mutipy?
        #if len(self.positive)>0:
        #    lb = self.positive[-1].axon.outneurons[0].label
        if (nmax!=None):
            lb=nmax.axon.outneurons[0].label
            if lb==label:
                return

        if (self.knowledges.__contains__(label)):
            # print("Already in,create dendritic only", label)
            # pltshow(img)
            nlb = self.knowledges[label]
            # if label already in memory?
            # not create n ,just create a dendritic ok
        else:
            nlb = neuron()
            #self.pallium.append(nlb)
            nlb.label=label
            self.knowledges[label] = nlb
        #how to create tree

        npt = neuron()
        self.pallium.append(npt)
        dnpt=npt.dendritic
        nn = neuron()
        self.pallium.append(nn)
        dn=nn.dendritic

        n = neuron()
        self.pallium.append(n)
        nlb.inaxon.append(n.axon)
        n.axon.outneurons.append(nlb)
        d=n.dendritic
        d.connectfrom(npt.axon,1)
        d.connectfrom(nn.axon,1)


        #imgleft=self.output()
        #if(imgleft.sum()==0 and len(self.positive)>0):
        #    for pn in self.positive:
        #        d.connectfrom(pn.axon, 1)
        #    return

        r, c = img.shape
        for i in range(r):
            for j in range(c):
                if (img[i][j] > 0):#positive
                    dnpt.connectfrom(self.neurons[i, j].axon, 1)
                else:#negative
                    dn.connectfrom(self.neurons[i, j].axon, -1)


    def predict(self, img):
        nmax = self.look(img)
        if(nmax !=None):
            return nmax.axon.outneurons[0].label
        else:
            return None

    def recall(self,label):
        n=self.knowledges[label]
        n.recall()
        img=self.output()
        print(img)
    def output(self):
        img=np.zeros((self.ROWS,self.COLS),np.uint8)
        for i in range(self.ROWS):
            for j in range(self.COLS):
                img[i,j]=self.neurons[i,j].value
        return img
    def status(self):
        s=0
        for n in self.pallium:
           s+= len(n.dendritic.synapses)
        print("synapses:",s)
        pass


if __name__ == "__main__":
    from hzk import *

    allhz = hzk()
    i0 = hz2img(allhz[0])
    i1 = hz2img(allhz[1])
    i2 = hz2img(allhz[2])
    i3 = hz2img(allhz[3])
    i4 = hz2img(allhz[4])
    i5 = hz2img(allhz[5])
    i6 = hz2img(allhz[6])
    i7 = hz2img(allhz[7])
    i8 = hz2img(allhz[8])
    i9 = hz2img(allhz[9])
    # print(i1)

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

    # print(on.knowledges)
    # print(on.neurons[0,0].axon.synapses)
    # print(len(on.pallium[0].dendritic.synapses))
    lb = on.predict(i2)
    print(lb)

    # import tensorflow.examples.tutorials.mnist.input_data as input_data

    # mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    # batch_xs, batch_ys = mnist.train.next_batch(100)

