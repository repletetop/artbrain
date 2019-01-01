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



def pltshow(img,title):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title(title)
    ax.imshow(img, cmap='gray')
    plt.show()

class pallium:#局部回路神经元及抑制
    def __init__(self):
        self.neurons=[]
        self.maxvalue=0

class opticnerve:
    def __init__(self, ROWS, COLS):
        self.neurons = np.array([[neuron() for ii in range(COLS)]
                                 for jj in range(ROWS)])  # brains
        self.infrneurons=[] # inference neurons
        self.pallium=[] #for memory促发记忆主要由大脑皮层控制
        self.knowledges = {}
        self.dendritics = []
        self.COLS = COLS
        self.ROWS = ROWS
        self.NEURONSCOUNT = ROWS * COLS
        self.createfeellayers()


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

    def center(self, image):
        mn=image
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

    def sdr(self,img):
        self.input(img)
        R=1
        ROWS, COLS = img.shape
        for i in range(R,ROWS-R):
            for j in range(R,COLS-R):
                n=self.neurons[i, j]
                if(n.value==0):
                    continue
                actived = False
                for x in range(-R,R+1):
                    for y in range(-R,R+1):
                        if (x == 0 and y == 0):
                            continue
                        if self.neurons[i+x,j+y].actived:
                            actived = True
                            break
                n.actived = not actived
                img[i,j] = n.actived
        return  img
    def conv(self,img):
        self.input(img)
        ROWS, COLS = img.shape
        #SDR->COND3x3->SDR-COND3X3
        self.layer1 = np.array([[neuron() for ii in range(COLS//3)]
                                 for jj in range(ROWS//3)])  # brains
        cimg=np.zeros((ROWS//3,COLS//3))
        for i in range(ROWS//3):
            for j in range(COLS//3):
                n=self.layer1[i, j]
                #connect pre
                for x in range(3):
                    for y in range(3):
                        n.dendritic.connectfrom(self.neurons[3*i+y,3*j+x].axon,1)
                n.calcValue()
                print(n.value)
                if n.value>4:
                    cimg[i,j]=1
        return cimg

    def createfeellayers_org(self):
        ROWS, COLS = self.ROWS,self.COLS
        self.layers=[]
        lastlayer = self.neurons
        #sdr
        layer = np.array([[neuron() for ii in range(COLS)]
                            for jj in range(ROWS)])  # brains
        self.layers.append(layer)#orig layer
        #layer=layer.copy()
        #self.layers.append(layer)#sdr actived layer
        #sdr
        for i in range(1, ROWS - 1):
            for j in range(1, COLS - 1):
                n = layer[i, j]
                # connect pre
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        n.dendritic.connectfrom(lastlayer[i + x, j + y].axon, 1)
                        if (x == 0 and y == 0):  # max
                            continue
                        n.indendritics.append(layer[i + x, j + y].dendritic)
                        n.nagativeaxons.append(layer[i + x, j + y].axon)

        # conv2
        while ROWS>=4:
            ROWS = ROWS // 2
            COLS = COLS // 2
            lastlayer=layer
            layer = np.array([[neuron() for ii in range(COLS)]
                                    for jj in range(ROWS)])  # brains
            self.layers.append(layer)
            for i in range(1, ROWS - 1):
                for j in range(1, COLS - 1):
                    n = layer[i, j]
                    # connect pre
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            n.dendritic.connectfrom(lastlayer[2 * i + y, 2 * j + x].axon, 1)
                            if (x == 0 and y == 0):  # max
                                continue
                            n.indendritics.append(layer[i + x, j + y].dendritic)
                            n.nagativeaxons.append(layer[i + x, j + y].axon)

    def createfeellayers(self):
        ROWS, COLS = self.ROWS,self.COLS
        self.layers=[]
        #sdr
        layer = np.array([[neuron() for ii in range(COLS)]
                            for jj in range(ROWS)])  # brains
        self.layers.append(layer)#orig layer
        while True:
            # sdr connect
            lastlayer=layer
            layer = np.array([[neuron() for ii in range(COLS)]
                                    for jj in range(ROWS)])  # brains
            self.layers.append(layer)
            for i in range(ROWS):
                for j in range(COLS):
                    n = layer[i, j]
                    # connect neighbourhood
                    n.dendritic.connectfrom(lastlayer[i , j ].axon, 1)

                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if(i+x<0 or i+x>=ROWS or j+y<0 or j+y>=COLS):
                                continue
                            n.nbdendritic.connectfrom(lastlayer[i + x, j + y].axon, 1)
                            n.indendritics.append(layer[i + x, j + y].nbdendritic)
                            n.nagativeaxons.append(layer[i + x, j + y].axon)

            if ROWS<4:
                break;

            # conv2
            lastlayer = layer
            ROWS = ROWS // 2
            COLS = COLS // 2
            layer = np.array([[neuron() for ii in range(COLS)]
                                    for jj in range(ROWS)])  # brains
            self.layers.append(layer)
            for i in range(ROWS):
                for j in range(COLS):
                    n = layer[i, j]
                    # connect pre
                    for x in range(2):
                        for y in range(2):
                            n.dendritic.connectfrom(lastlayer[2 * i + y, 2 * j + x].axon, 1)

        self.infrneurons=[[] for i in range(len(self.layers))]


    def feeltimes(self,img):
        pass

    def feel(self, img):
        #for n in self.infrneurons:#
        #    n.dendritic.value=0

        r, c = img.shape
        for i in range(r):
            for j in range(c):
                # if img[i, j]!=0:
                self.layers[0][i, j].value = img[i, j]

        #pltshow(self.outputlayer(self.layers[0]))
        for ilayer in range(1,len(self.layers)):
            layer=self.layers[ilayer]
            ROWS,COLS=layer.shape
            for i in range(ROWS):
                for j in range(COLS):
                    layer[i, j].calcDendritic()
                    layer[i, j].calcNbDendritic()

            # hengxiangyizhi TOP3 n.value =1
            for i in range(ROWS):
                for j in range(COLS):
                    n = layer[i, j]
                    if n.dendritic.value == 0:
                        n.value = 0
                        continue
                    if n.indendritics != []:
                        list1 = sorted(n.indendritics, key=lambda v: v.value, reverse=True)
                        #if n.nbdendritic in  list1[0:3]:#6,6,6,6,5,5,5,5,2,1
                        if n.nbdendritic.value>=list1[3].value:
                            n.value=1
                        else:
                            n.value=0
                    else:
                        n.value = 1
                    #vmax = max(n.indendritics, key=lambda v: v.value)
                    #if n.dendritic.value >= vmax.value:
                    #    n.value = 1
                    #    #cimg[i, j] = 1
            #pltshow(self.outputlayer(layer))
            # sdr 3/9
            for i in range(ROWS):
                for j in range(COLS):
                    n = layer[i, j]
                    if n.nagativeaxons == [] or n.value == 0:
                        continue
                    #only one actived
                    #vmax = max(n.nagativeaxons, key=lambda v: v.connectedNeuron.actived)
                    #if vmax.connectedNeuron.actived:
                    #    continue
                    #reserve 3 point
                    cnt=0
                    for ax in n.nagativeaxons:
                        if ax.connectedNeuron.value:
                            cnt+=1
                    if cnt<3:#only reserve 3 point include self
                        n.value = 1
                    else:
                        n.value = 0
            #pltshow(self.outputlayer(layer))




    def feel_org(self, img):
        self.input(img)
        ROWS, COLS = img.shape
        self.layers=[]
        lastlayer = self.neurons
        #sdr
        layer = np.array([[neuron() for ii in range(COLS)]
                            for jj in range(ROWS)])  # brains
        self.layers.append(layer)

        for i in range(1, ROWS - 1):
            for j in range(1, COLS - 1):
                n = layer[i, j]
                # connect pre
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        n.dendritic.connectfrom(lastlayer[i + x, j + y].axon, 1)
                        if (x == 0 and y == 0):  # max
                            continue
                        n.indendritics.append(layer[i + x, j + y].dendritic)
                        n.nagativeaxons.append(layer[i + x, j + y].axon)

        cimg = np.zeros((ROWS, COLS))
        for i in range(ROWS):
            for j in range(COLS):
                n = layer[i, j]
                n.calcDendritic()
        # hengxiangyizhi n.value =1
        for i in range(ROWS):
            for j in range(COLS):
                n = layer[i, j]
                if n.indendritics == [] or n.dendritic.value == 0:
                    continue
                vmax = max(n.indendritics, key=lambda v: v.value)
                if n.dendritic.value >= vmax.value:
                    n.value = n.dendritic.value
                    cimg[i, j] = 1
        pltshow(cimg)
        # sdr
        cimg = np.zeros((ROWS, COLS))
        for i in range(ROWS):
            for j in range(COLS):
                n = layer[i, j]
                if n.nagativeaxons == [] or n.value == 0:
                    continue
                vmax = max(n.nagativeaxons, key=lambda v: v.connectedNeuron.actived)
                if vmax.connectedNeuron.actived:
                    continue
                n.actived = True
                cimg[i, j] = 1  # n.dendritic.value
        pltshow(cimg)
        # conv2
        while ROWS>=6:
            ROWS = ROWS // 3
            COLS = COLS // 3
            lastlayer=layer
            layer = np.array([[neuron() for ii in range(COLS)]
                                    for jj in range(ROWS)])  # brains
            self.layers.append(layer)
            cimg = np.zeros((ROWS, COLS))
            for i in range(1, ROWS - 1):
                for j in range(1, COLS - 1):
                    n = layer[i, j]
                    # connect pre
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            n.dendritic.connectfrom(lastlayer[3 * i + y, 3 * j + x].axon, 1)
                            if (x == 0 and y == 0):  # max
                                continue
                            n.indendritics.append(layer[i + x, j + y].dendritic)
                            n.nagativeaxons.append(layer[i + x, j + y].axon)

                    n.calcDendritic()
                    if n.dendritic.value > 0:
                        n.value = 1
                        cimg[i, j] = 1
            pltshow(cimg)
            # sdr
            cimg = np.zeros((ROWS, COLS))
            for i in range(ROWS):
                for j in range(COLS):
                    n = layer[i, j]
                    if n.nagativeaxons == [] or n.value == 0:
                        continue
                    vmax = max(n.nagativeaxons, key=lambda v: v.connectedNeuron.actived)
                    if vmax.connectedNeuron.actived:
                        continue
                    n.actived = True
                    cimg[i, j] = 1  # n.dendritic.value
            pltshow(cimg)

        return cimg





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
        if len(self.pallium)==0:
            self.actived=[]
            return None

        self.reset()
        self.input(img)
        results=[]
        for n in self.pallium:
            n.calcValue()
            if(n.axon.outneurons!=[]):
                results.append(n)


        list1 = sorted(results, key=lambda n: n.dendritic.value, reverse=True)
        nmax=list1[0]# may be mut, not top, how can i do? = .value=28*28

        return nmax

    @with_goto
    def train_1230(self,imgs,labels):
        label .trainlabel
        for i in range(imgs.shape[0]):
            #if labels[i] not in [4,9,7]:
            #    continue
            if(i==9):
                b=0
                pass
            #nu = self.predict(imgs[i])
            #lbs=[n.label for n in nu.axon.outneurons]

            alike = self.learn(imgs[i],labels[i]) #learn knowledge if confilict lost befor
            while len(alike)>0:
                ca=alike.copy()
                alike.clear()
                for n in ca:
                    self.clearneurons()
                    n.reappear()
                    img = self.output()
                    print(labels[i],"like", n.axon.outneurons[0].label,end=",  ")
                    sys.stdout.flush()
                    nalike = self.learn(img, n.axon.outneurons[0].label)
                    alike=alike+nalike

        label .testlabel
        ok =True
        for i in range(imgs.shape[0]):
            #if labels[i] not in [4,9,7]:
            #    continue
            if(i==9):
                b=0
            nu=self.predict(imgs[i])
            lbs=[n.label for n in nu.axon.outneurons]
            if(labels[i] not in lbs):
                print(i,"Error: %s predict %s ,learn again "%(labels[i],str(lbs)) )
                #print("May by 2 yi")
                #pltshow(imgs[i])

                self.learn(imgs[i],labels[i])
                lb = self.predict(imgs[i])
                ok = False

        if ok==False:
            goto .testlabel

    def getknow(self,label):
        if (self.knowledges.__contains__(label)):
            nlb = self.knowledges[label]
        else:
            nlb = neuron()
            nlb.label = label
            self.knowledges[label] = nlb
        return nlb

    @with_goto
    def train(self,imgs,labels):
        label .trainlabel
        for i in range(imgs.shape[0]):
            #if labels[i] not in [4,9,7]:
            #    continue
            if(i==9):
                b=0
                pass
            
            img=imgs[i]
            imglabel=labels[i]
            memory = self.remember(img,imglabel)
            print("i:",i)

            label .predictlabel
            nu,alike,ilayer = self.inference(img)
            know=self.getknow(imglabel)
            if(alike==None):#not found,create new
                nu.axon.outneurons.append(know)
                nu.memory = [memory]
                self.infrneurons[-ilayer].append(nu)
            else:
                lbs=[n.label for n in alike.axon.outneurons]
                if(imglabel not in lbs):#found but not this label, need create new and renew history 
                    print("####Error: %s predict %s ,need renew memory "%(imglabel,str(lbs)) )
                    his =  alike.memory.copy()
                    alike.memory.clear()
                    nu.axon.outneurons.append(know)
                    nu.memory=[memory]
                    self.infrneurons[-ilayer].append(nu)
                    for n in his:
                        #pltshow(img,imglabel)
                        self.clearneurons()
                        n.reappear()
                        img=self.output()
                        imglabel=n.axon.outneurons[0].label
                        memory=n
                        #pltshow(img,imglabel)
                        goto .predictlabel
                        #elf.learn(img,h.label) #use goto can instead diguisuanfa
        
                else:
                    if(len(lbs)>1):
                        print(i,"%s predict more than one: %s "%(labels[i],str(lbs)) )
                    else:
                        print(lbs[0],end=" ")

        label .testlabel
        ok =True
        for i in range(imgs.shape[0]):
            #if labels[i] not in [4,9,7]:
            #    continue
            if(i==9):
                b=0
            nu,alike,ilayer=self.inference(imgs[i])
            if alike!=None:
                lbs=[n.label for n in alike.axon.outneurons]
            else:
                lbs=[]
            if(labels[i] not in lbs):
                print(i,"Error: %s predict %s ,learn again "%(labels[i],str(lbs)) )
                #print("May by 2 yi")
                #pltshow(imgs[i])

                self.learn(imgs[i],labels[i])
                lb = self.predict(imgs[i])
                ok = False

        if ok==False:
            goto .testlabel



    def learn_org(self, img, label):#digui is slow
        self.feel(img)
        #inference
        self.actived=[]
        alike=[]

        newn = neuron()
        d = newn.dendritic
        if (self.knowledges.__contains__(label)):
            # print("Already in,create dendritic only", label)
            nlb = self.knowledges[label]
        else:
            nlb = neuron()
            self.knowledges[label] = nlb
            nlb.label = label
        lenactived=0
        for ilayer in range(1,len(self.layers)+1,1):#skip a lay ,only sdr
            layer=self.layers[-ilayer]
            #pltshow(self.outputlayer(layer))
            R, C = layer.shape
            for r in range(R):
                for c in range(C):
                    layer[r, c].conduct(self.actived)


            #if found actived label already in,return
            #else if found actived but not it history need renew and remember this label
            #else :no actived, add memory
            for r in range(R):
                for c in range(C):
                    n = layer[r, c]
                    d.connectfrom(n.axon, n.value)

            if(len(self.actived)==lenactived):
                self.infrneurons.append(newn)
                nlb.inaxon.append(newn)
                newn.axon.outneurons.append(nlb)

                #remember img,label , maybe already in memory
                #newn.memory = [self.remember(img, label)]
                newn.memory=[self.remember(img,label)]
                #
                #1.remember this
                #R,C=layer.shape
                break# out
            else:
                #print(len(self.actived))
                #actived new one
                if len(self.actived)==lenactived+1:
                    act = self.actived[lenactived]
                    #and it has one outneurons  and label is me
                    if len(act.axon.outneurons) == 1 \
                            and act.axon.outneurons[0].label == label:  # found
                        nhistory = self.remember(img, label)
                        if nhistory not in self.actived[0].memory:
                            self.actived[0].memory.append(nhistory)
                        del (newn)
                        # self.actived[0].memory.append(img)
                        break
                #other else renew memory
                for ia in range(lenactived,len(self.actived)):
                    act = self.actived[ia]
                    if len(act.axon.outneurons)==1 and len(act.memory)>0:
                        #memory need renew
                        alike = alike+act.memory
                        act.memory.clear()
                    if nlb not in act.axon.outneurons:
                        act.axon.outneurons.append(nlb)
                lenactived = len(self.actived)

        if len(alike)>1 and ilayer>len(self.layers):
            print("same img have two diffrent labels!",len(alike))

        #for a in alike:
        #    a.memory.clear()
        return alike

    def study(self,img,label):
        n=self.predict(img)
        if(n.label == label):
            return

    def learn(self, img, label):#digui is slow
        self.feel(img)

        self.actived=[]
        alike=[]

        newn = neuron()
        d = newn.dendritic
        if (self.knowledges.__contains__(label)):
            # print("Already in,create dendritic only", label)
            nlb = self.knowledges[label]
        else:
            nlb = neuron()
            self.knowledges[label] = nlb
            nlb.label = label
        lenactived=0
        for ilayer in range(1,len(self.layers)+1,1):#skip a lay ,only sdr
            layer=self.layers[-ilayer]
            #pltshow(self.outputlayer(layer))
            R, C = layer.shape
            for r in range(R):
                for c in range(C):
                    n = layer[r, c]
                    d.connectfrom(n.axon, n.value)

            #self.conductlayer(layer,self.actived)
            #for r in range(R):#too slowly
            #    for c in range(C):
            #        layer[r, c].conduct(self.actived)
            for n in self.infrneurons[-ilayer]:
                n.calcValue()
                if n.dendritic.value>=len(n.dendritic.synapses):#actived
                    if n.axon.outneurons != []:
                        self.actived.append(n)


            #if found actived label already in,return
            #else if found actived but not it history need renew and remember this label
            #else :no actived, add memory

            if(len(self.actived)==lenactived):
                self.infrneurons[-ilayer].append(newn)
                nlb.inaxon.append(newn)
                newn.axon.outneurons.append(nlb)

                #remember img,label , maybe already in memory
                #newn.memory = [self.remember(img, label)]
                newn.memory=[self.remember(img,label)]
                #
                #1.remember this
                #R,C=layer.shape
                break# out
            else:
                #print(len(self.actived))
                #actived new one
                if len(self.actived)==lenactived+1:
                    act = self.actived[lenactived]
                    #and it has one outneurons  and label is me
                    if len(act.axon.outneurons) == 1 \
                            and act.axon.outneurons[0].label == label:  # found
                        nhistory = self.remember(img, label)
                        if nhistory not in self.actived[0].memory:
                            self.actived[0].memory.append(nhistory)
                        del (newn)
                        # self.actived[0].memory.append(img)
                        break
                #other else renew memory
                for ia in range(lenactived,len(self.actived)):
                    act = self.actived[ia]
                    if len(act.axon.outneurons)==1 and len(act.memory)>0:
                        #memory need renew
                        alike = alike+act.memory
                        act.memory.clear()
                    if nlb not in act.axon.outneurons:
                        act.axon.outneurons.append(nlb)
                lenactived = len(self.actived)

        if len(alike)>1 and ilayer>len(self.layers):
            print("same img have two diffrent labels!",len(alike))

        #for a in alike:
        #    a.memory.clear()
        return alike

    def remember(self, img, label):
        #img=self.sdr(img)
        nmax = self.look(img)  # found mutipy?
        #remember every thing diffrence
        if (nmax!=None):
            lb=nmax.axon.outneurons[0].label
            if lb == label: #allow mistake ,train twice
                return nmax
            #if nmax.dendritic.value==self.ROWS*self.COLS:
            #    return nmax
            #else:
            #    print(lb,label,nmax.dendritic.value)

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

        r, c = img.shape
        for i in range(r):
            for j in range(c):
                if (img[i][j] > 0):#positive
                    d.connectfrom(self.neurons[i, j].axon, 1)
                else:#negative
                    d.connectfrom(self.neurons[i, j].axon, -1)
        return n

    def reform(self):
        dictaxonpolarity = {}
        dictlen = {}
        dictdendritic = {}
        dictset={}
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
        for n in self.pallium:
            axon = n.axon
            if (len(axon.synapses) == 0):
                continue
            # calc axon,polarity in set
            for s in axon.synapses:
                axonpolarity = (axon, s.polarity)
                if (axonpolarity in dictaxonpolarity):
                    dictaxonpolarity[axonpolarity].append(s.dendritic)
                else:
                    dictaxonpolarity[axonpolarity] = [s.dendritic]

        for axonpolarity in dictaxonpolarity:
            dictlen[str(set(dictaxonpolarity[axonpolarity]))] = len(set(dictaxonpolarity[axonpolarity]))
            dictdendritic[str(set(dictaxonpolarity[axonpolarity]))] = set(dictaxonpolarity[axonpolarity])
            if str(set(dictaxonpolarity[axonpolarity])) in dictset:
                dictset[str(set(dictaxonpolarity[axonpolarity]))].append(axonpolarity)
            else:
                dictset[str(set(dictaxonpolarity[axonpolarity]))] = [axonpolarity]

        list1 = sorted(dictlen.items(), key=lambda x: x[1], reverse=True)
        # print(list1)
        for (k, v) in list1: #
            #v=dendrtic cnt
            if (v < 2):# 4: 2*4s+4d => 2s+1d+4s+1d+1n
                break
            dset = dictset[k]
            ncnt=len(dset)
            if(ncnt<2):# dendritic must have up 2 synapses
                continue
            if(ncnt*v<8):
                continue

            ds = dictdendritic[k]

            n = neuron()
            self.pallium.append(n)
            for (axon,polarity) in dset:
                n.dendritic.connectfrom(axon,polarity)

            for d in ds:
                d.connectfrom(n.axon,1)
                for (axon, polarity) in dset:
                    d.disconnectfrom(axon,polarity)
        self.sortpallium()

    def sortpallium(self):
        v=[]
        while len(self.pallium)>0:
            for n in self.pallium:
                hasin=False
                for s in n.dendritic.synapses:
                    if s.axon.connectedNeuron in self.pallium:
                        hasin = True
                        break
                if hasin == False:
                    v.append(n)
                    self.pallium.remove(n)
        self.pallium=v


    def predict(self,img):
        nu,alike,ilayer = self.inference(img)
        if alike!=None:
            lbs=[n.label for n in alike.axon.outneurons]
        else:
            lbs=[]
        print(lbs)
        if lbs!=[]:
            return lbs[0]
        else:
            return None

    def predict_org(self, img):  # inference digui is slow
        self.feel(img)

        self.actived = []
        for i in range(1, len(self.layers) + 1, 1):
            layer = self.layers[-i]
            R, C = layer.shape
            for r in range(R):
                for c in range(C):
                    layer[r, c].conduct(self.actived)

            for a in self.actived:
                if len(a.axon.outneurons) == 1:
                    return a.axon.outneurons[0].label

            # if(len(self.actived)>0):
            #    print(len(self.actived))
            #    label=self.actived[0].axon.outneurons[0].label
            #    break
        return None

    def inference(self,img):#inference
        self.feel(img)

        #self.actived=[]
        nu=neuron()
        alike=[None]
        for ilayer in range(1,len(self.layers)+1,1):
            layer=self.layers[-ilayer]
            blAct = False
            for nrow in layer:
                for n in nrow:
                    nu.dendritic.connectfrom(n.axon,n.value)

            for n in self.infrneurons[-ilayer]:
                n.calcValue()
                if n.dendritic.value>=len(n.dendritic.synapses):#actived
                    #self.actived.append(n)
                    blAct = True
                    if len(n.axon.outneurons) == 1:
                        n.ilayer=ilayer
                        alike.append(n)
                        #return n# get active  but not the end,because someone can active more then one ,the last is we found


            if blAct==False:
                return nu,alike[-1],ilayer
            else:
                print("ilayer:",ilayer)

        #never goto this place
        print("#############all matched ,mayby 2yi,get last match",ilayer)
        #self.actived[-1].ilayer=ilayer
        return nu,alike[-1],ilayer


    def recall(self, img):#recall from history
        #img=self.sdr(img)
        nmax = self.look(img)
        if(nmax !=None):
            return nmax.axon.outneurons[0].label
        else:
            return None

    def reappear(self,label):
        n=self.knowledges[label]
        n.reappear()
        img=self.output()
        print(img)
    def output(self):
        img=np.zeros((self.ROWS,self.COLS),np.uint8)
        for i in range(self.ROWS):
            for j in range(self.COLS):
                img[i,j]=self.neurons[i,j].value
        return img
    def outputlayer(self,layer):
        R,C = layer.shape
        img=np.zeros((R,C),np.uint8)
        for i in range(R):
            for j in range(C):
                img[i,j]=layer[i,j].value
        return img

    def status(self):
        s=0
        for n in self.pallium:
            cnt=len(n.dendritic.synapses)
            s+= cnt
            if cnt==1:
                print ("Only one synapse",n)
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

