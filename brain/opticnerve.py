# -*-coding:utf-8-*-
import numpy as np
import shelve

from neuron import *
import itertools

import pandas as pd
from pandas import Series,DataFrame
from goto import with_goto
import sys


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

    def compute(self):
        self.positive=[]
        for n in self.pallium:
            n.calcValue()
            if (n.dendritic.value>=len(n.dendritic.synapses)):
                self.positive.append(n)

    def feel(self,img):
        self.reset()
        r, c = img.shape
        for i in range(r):
            for j in range(c):
                if  img[i, j]==1:
                    aa=1
                self.neurons[i, j].value = img[i, j]
                self.neurons[i, j].conduct()
        #self.compute()
        postive=[]
        for n in self.pallium:
            n.calcValue()
            if(n.value>0):
                postive.append(n)
        lenp = len(postive)
        MINICNT=0
        if(lenp>MINICNT):
            #if lenp==1:
            #    n.dendritic.connectfrom(postive[0].axon,1)
            #    self.pallium.append(n)
            #else:
            n = neuron()
            for pn in postive:
                    n.dendritic.connectfrom(pn.axon,1)
                    for s in pn.axon.synapses:
                        s.dendritic.disconnect(s)
                        s.dendritic.connectfrom(n.axon)
            self.pallium.append(n)

        imgleft = self.output()
        if(imgleft.sum()<=MINICNT):
            return
        n = neuron()
        r, c = imgleft.shape
        for i in range(r):
            for j in range(c):
                if (imgleft[i][j] > 0):  # positive
                    nn=neuron()
                    self.pallium.append(nn)
                    nn.dendritic.connectfrom(self.neurons[i, j].axon, 1)
                    n.dendritic.connectfrom(nn.axon,1)
                #else:  # negative
                    #n.dendritic.connectfrom(self.neurons[i, j].axon, -1)
        self.pallium.append(n)

    def input(self,img):
        self.reset()
        r, c = img.shape
        for i in range(r):
            for j in range(c):
                # if img[i, j]!=0:
                self.neurons[i, j].value = img[i, j]
                #self.neurons[i, j].conduct()
    def conduct(self):
        for i in range(self.ROWS):
            for j in range(self.COLS):
                    self.neurons[i, j].conduct(self.actived)

    def look(self, img):  # get max overlap
        if len(self.knowledges)==0:
            self.actived=[]
            return None

        for n in self.pallium:
            n.calcValue()

        vmax=0
        nmax=None
        for k,v in self.knowledges:
            if vmax<v.value:
                vmax=v.value
                nmax=v

        vmax=self.actived[0].dendritic.value
        nmax=self.actived[0]
        for n in self.actived[1:]:
            #print(n.axon.outneurons[0].label,end=" ")
            if n.dendritic.value>vmax:
                vmax = n.dendritic.value
                nmax=n
        return nmax

        #vmax=self.actived[0].dendritic.value/len(self.actived[0].dendritic.synapses)
        #nmax=self.actived[0]
        #for n in self.actived[1:]:
        #    if self.actived[0].dendritic.value/len(self.actived[0].dendritic.synapses)>vmax:
        #        vmax = self.actived[0].dendritic.value/len(self.actived[0].dendritic.synapses)
        #        nmax=n
        #return nmax
    @with_goto
    def train(self,imgs,labels):
        label .trainlabel
        for i in range(imgs.shape[0]):
            if(i==16):
                b=0
                pass
            self.remember(imgs[i],labels[i])


        label .testlabel
        for i in range(imgs.shape[0]):
            lb=self.predict(imgs[i])
            if(labels[i]!=lb):
                print("predict error,remember again.")
                self.remember(imgs[i],labels[i])
                goto .trainlabel



    def remember(self, img, label):
        nmax = self.look(img)  # found mutipy?
        #remember every thing diffrence
        if (nmax!=None):
            lb=nmax.axon.outneurons[0].label
            #if lb==label and self.times<=1:
            if lb == label: #allow mistake ,train twice
                return

        if (self.knowledges.__contains__(label)):
            # print("Already in,create dendritic only", label)
            nlb = self.knowledges[label]
            n=neuron()
            self.pallium.append(n)
            nlb.inaxon.append(n)
            n.axon.outneurons.append(nlb)
            d=n.dendritic
        else:
            nlb = neuron()
            self.knowledges[label]=nlb
            nlb.label=label
            n=neuron()
            self.pallium.append(n)
            nlb.inaxon.append(n)
            n.axon.outneurons.append(nlb)
            d=n.dendritic


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
                    d.connectfrom(self.neurons[i, j].axon, 1)
                else:#negative
                    d.connectfrom(self.neurons[i, j].axon, -1)
    def reform(self,synapses):
        isFirst = True
        for s in synapses:
            ns = []
            for ds in s.dendritic.synapses:
                ns.append((ds.axon, ds.polarity))  # one dendritic
            if isFirst:
                isFirst = False
                intersection = set(ns)
                ns1=set(ns)
            else:
                intersection = intersection & set(ns)
                if len(intersection) < 30:
                    return False

        if (isFirst == False ):
            # print("intersction len:",len(intersection))
            n = neuron()
            self.pallium.append(n)

            for s in synapses:
                for idx in range( len(s.dendritic.synapses),0,-1):
                    ds=s.dendritic.synapses[idx-1]
                #for ds in s.dendritic.synapses:  # every dendritic
                    if (ds.axon, ds.polarity) in intersection:
                        s.dendritic.disconnect(ds)
                s.dendritic.connectfrom(n.axon, 1)

            for axon, polarity in intersection:
                n.dendritic.connectfrom(axon, polarity)
            # show
            self.clearneurons()
            n.recall()
            img = self.output()
            pltshow(img)
            return True
        return False

    def thinkneuron_first(self,nun):
        cnt = len(nun.axon.synapses)
        while cnt>5:
            print(cnt)
            ret = self.reform(nun.axon.synapses[0:cnt])
            if(ret):
                cnt = len(nun.axon.synapses)
            else:
                cnt = cnt-1

    def thinkneuron(self,nun):#random
        dictaxonpolarity={}
        for s in nun.axon.synapses:
            for ds in s.dendritic.synapses:
                axonpolarity = (ds.axon, ds.polarity)
                if(axonpolarity in dictaxonpolarity):
                    dictaxonpolarity[axonpolarity].append(s.dendritic)
                else:
                    dictaxonpolarity[axonpolarity] =[s.dendritic]
        print (dictaxonpolarity)



    def thinkneuron_slow(self,nun):#slow
        cnt=len(nun.axon.synapses)
        while cnt>5:
            print(len(nun.axon.synapses),cnt)
            for cs in itertools.combinations(nun.axon.synapses, cnt):
                ret=self.reform(cs)
                if ret:
                    cnt=len(nun.axon.synapses)
                    break;
            cnt = cnt - 1

    @with_goto
    def think_two_letter_common_pix(self):#_two letter common most pix
        lastmaxlen=99999
        label .begin
        maxlen=0
        for i in range(len(self.pallium)):
            for j in range(i+1,len(self.pallium)):
                ni=self.pallium[i]
                nj=self.pallium[j]
                ai=[]
                for s in ni.dendritic.synapses:
                    ai.append(s.axon)
                aj=[]
                for s in nj.dendritic.synapses:
                    aj.append(s.axon)
                intersection=set(ai) & set(aj)
                if(maxlen<len(intersection)):
                    maxlen=len(intersection)
                    mi=i;mj=j;da=intersection;
                    if maxlen==lastmaxlen:
                        goto .endfor

        label .endfor

        if maxlen>100:
            print(mi,mj,maxlen,end=", ")
            sys.stdout.flush()
            n = neuron()
            self.pallium.append(n)
            self.pallium[mi].dendritic.connectfrom(n.axon, 1)
            self.pallium[mj].dendritic.connectfrom(n.axon, 1)
            for a in da:
                n.dendritic.connectfrom(a, 1)
                self.pallium[mi].dendritic.disconnectfrom(a)
                self.pallium[mj].dendritic.disconnectfrom(a)
            #output
            #self.clearneurons()
            #n.recall()
            #img=self.output()
            #pltshow(img)
            #####
            lastmaxlen=maxlen
            goto .begin

    # <10
    # synapses: 68395
    #pallium: 483
    #synapses: 24882
    #pallium: 1302
    #13.527137994766235
    #train
    #500
    #Right: 500
    #500
    #100 %
    # <100
    # 28.1
    # 100 <2
    def think(self):
        dictaxonpolarity = {}
        dictlen = {}
        dictdendritic = {}
        for i in range(self.ROWS):
            for j in range(self.COLS):
                # print("R,C",i,j)
                axon = self.neurons[i, j].axon
                if (len(axon.synapses) == 0):
                    continue
                #calc axon,polarity in set
                for s in axon.synapses:
                    axonpolarity = (axon, s.polarity)
                    if (axonpolarity in dictaxonpolarity):
                        dictaxonpolarity[axonpolarity].append(s.dendritic)
                    else:
                        dictaxonpolarity[axonpolarity] = [s.dendritic]
        for (axon, polarity) in dictdendritic:
            dictlen[str(set(dictaxonpolarity[axonpolarity]))] = len(set(dictaxonpolarity[axonpolarity]))
            dictdendritic[str(set(dictaxonpolarity[axonpolarity]))] = set(dictaxonpolarity[axonpolarity])
            if str(set(dictaxonpolarity[axonpolarity])) in dictdendritic:
                dictset[str(set(dictaxonpolarity[axonpolarity]))].append((axon,polarity))
            else:
                dictset[str(set(dictaxonpolarity[axonpolarity]))] = [(axon, polarity)]

        list1 = sorted(dictlen.items(), key=lambda x: x[1], reverse=True)
        # print(list1)
        for (k, v) in list1: #
            if (v <= 4):# 4: 2*4s+4d => 2s+1d+4s+1d+1n
                break
            ds = dictdendritic[k]
            dset=dictset[k]
            if(len(ds)<=1):# dendritic must have up 2 synapses
                continue

            n = neuron()
            self.pallium.append(n)
            for (axon,polarity) in dset:
                n.dendritic.connectfrom(axon,polarity)

            for d in ds:
                d.connectfrom(n.axon,1)
                for (axon, polarity) in dset:
                    d.disconnectfrom(axon)




    def think_mostlettercommon(self):#think most letter comm
        dictaxonpolarity = {}
        dictlen={}
        dictdendritic={}
        for i in range(self.ROWS):
            for j in range(self.COLS):
                #print("R,C",i,j)
                axon=self.neurons[i,j].axon
                if(len(axon.synapses)==0):
                    continue

                for s in axon.synapses:
                    axonpolarity = (axon, s.polarity)
                    if (axonpolarity in dictaxonpolarity):
                        dictaxonpolarity[axonpolarity].append(s.dendritic)
                    else:
                        dictaxonpolarity[axonpolarity] = [s.dendritic]
                dictlen[str(set(dictaxonpolarity[axonpolarity]))]=len(dictaxonpolarity[axonpolarity])
                dictdendritic[str(set(dictaxonpolarity[axonpolarity]))]=set(dictaxonpolarity[axonpolarity])

        #print(dictlen.values())
        list1 = sorted(dictlen.items(), key=lambda x: x[1],reverse=True)
        #print(list1)
        for (k,v) in list1:
            if(v<=1):
                break
            #print("trees:",v)
            ds=dictdendritic[k]
            axonpolarity=[]
            n=neuron()
            for ap in dictaxonpolarity:
                if set(dictaxonpolarity[ap])==ds:
                    axonpolarity.append(ap)
                    n.dendritic.connectfrom(ap[0],ap[1])
                    for d in ds:
                        d.disconnectfrom(ap[0])

            #print(v,len(axonpolarity),axonpolarity)
            if len(axonpolarity)==1:
                for d in ds:
                    d.connectfrom(n.dendritic.synapses[0].axon, ap[1])
                del n
            else:
                self.pallium.append(n)
                for d in ds:
                    d.connectfrom(n.axon, ap[1])

    #       for nplm in self.pallium:
 #           isFirst = True
 #           for s in nplm.axon.synapses:
 #               ns = []
 #               for ds in s.dendritic.synapses:
 #                   ns.append(ds.axon)  # one dendritic
 #               if isFirst:
 #                   isFirst = False
 #                   intersection = set(ns)
 #               else:
 #                   intersection = intersection & set(ns)
#
#            if (isFirst == False and len(intersection) >= 2):
#                # print("intersction len:",len(intersection))
#                n = neuron()
#                self.pallium.append(n)
#                for axon in intersection:
#                    n.dendritic.connectfrom(axon, 1)
#
#                for s in nplm.axon.synapses:
#                    for ds in s.dendritic.synapses:  # every dendritic
#                        if ds.axon in intersection:
#                            s.dendritic.synapses.remove(ds)
#                    s.dendritic.connectfrom(n.axon, 1)

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
        print("synapses:",s,"pallium:",len(self.pallium))
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

    # print(on.knowledges)
    # print(on.neurons[0,0].axon.synapses)
    # print(len(on.pallium[0].dendritic.synapses))
    lb = on.predict(i2)
    print(lb)

    # import tensorflow.examples.tutorials.mnist.input_data as input_data

    # mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
    # batch_xs, batch_ys = mnist.train.next_batch(100)

