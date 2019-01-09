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
        self.palliumidx=[]
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
        self.layer1 = np.array([[neuron() for ii in range(COLS//2)]
                                 for jj in range(ROWS//2)])  # brains
        cimg=np.zeros((ROWS//2,COLS//2))
        for i in range(ROWS//2):
            for j in range(COLS//2):
                n=self.layer1[i, j]
                #connect pre
                for x in range(2):
                    for y in range(2):
                        n.dendritic.connectfrom(self.neurons[2*i+y,2*j+x].axon,1)
                n.calcValue()
                print(n.dendritic.value)
                if n.dendritic.value>0:
                    cimg[i,j]=1
        return cimg

    def diff(self,img):#3.56 52
        imgnew=np.zeros((28,28),np.int)
        ROWS, COLS = img.shape
        for i in range(1,ROWS-1):
            for j in range(1,COLS-1):
                imgnew[i,j]=img[i,j]*9-img[i-1:i+2,j-1:j+2].sum()
                #if abs(imgnew[i,j])<30*9:#20,28
                #    imgnew[i,j]=0
        return imgnew

    def diff_old(self,img):#5.93 61
        ROWS, COLS = img.shape
        self.layers = []
        layer = np.array([[neuron() for ii in range(COLS)]
                          for jj in range(ROWS)])  # brains
        self.layers.append(layer)
        lastlayer = layer
        layer = np.array([[neuron() for ii in range(COLS)]
                          for jj in range(ROWS)])  # brains
        for i in range(ROWS):
            for j in range(COLS):
                n = layer[i, j]
                lastlayer[i,j].value=img[i,j]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if (i + x < 0 or i + x >= ROWS or j + y < 0
                                or j + y >= COLS):
                            continue
                        n.dendritic.connectfrom(lastlayer[i + x, j + y].axon, 1)
                n.calcValue()
        for i in range(ROWS):
            for j in range(COLS):
                n = layer[i, j]
                n.value = (lastlayer[i,j].value-n.value/9)
        imgnew=self.outputlayer(layer)
        #pltshow(img,'')
        #pltshow(imgnew,'')
        #print(imgnew)
        return  imgnew


    def createfeellayers(self):
        ROWS, COLS = self.ROWS,self.COLS
        self.layers=[]
        #sdr
        layer = np.array([[neuron() for ii in range(COLS)]
                            for jj in range(ROWS)])  # brains
        self.layers.append(layer)#orig layer
        while True:
            #fine thin hengxiangyizhi
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
                            #n.nagativeaxons.append(layer[i + x, j + y].axon)

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
                            #n.nbdendritic.connectfrom(lastlayer[i + x, j + y].axon, 1)
                            #n.indendritics.append(layer[i + x, j + y].nbdendritic)
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

    def conv14(self,img):
        ROWS,COLS=28,28
        layer = np.array([[neuron() for ii in range(COLS)]
                          for jj in range(ROWS)])  # brains
        layer14=[np.array([[neuron() for ii in range(COLS)]
                          for jj in range(ROWS)])
                 for n in range(14)]
        toplayer14 = [np.array([[neuron() for ii in range(COLS)]
                          for jj in range(ROWS)])
                      for n in range(14)]

        for i in range(COLS):
            for j in range(ROWS):
                layer[i,j].value=img[i,j]

        for l in range(1,14):
            ly=layer14[l-1]
            toply=toplayer14[l-1]
            maxneu=neuron()
            for i in range(l,COLS-l):
                for j in range(l,ROWS-l):
                    n = ly[i, j]
                    # connect pre
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            n.dendritic.connectfrom(layer[ i + x, j + y].axon, 1)
                    n.axon.outneurons=[maxneu]
                    eqneu = neuron()
                    eqneu.dendritic.connectfrom(n.axon,1)
                    eqneu.dendritic.connectfrom(maxneu.axon, -1)
                    toply[i, j].dendritic.connectfrom(eqneu.axon,0)
                    n.calcValue()

        for l in range(1,14):
            toply=toplayer14[l-1]
            for i in range(l,COLS-l):
                for j in range(l,ROWS-l):
                    n = toply[i, j]
                    n.dendritic.synapses[0].axon.connectedNeuron.calcValue()
                    n.calcValue()
            pltshow(self.outputlayer(toply),str(l))






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
                    if cnt<=3:#only reserve 3 point include self
                        n.value = 1
                    else:
                        n.value = 0
            #pltshow(self.outputlayer(layer),"")




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

    def look_20190102(self, img):  # get max overlap
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
    def look20190208(self, img):  # get seem overlap for memory
        #if len(self.pallium)==0:
        #    self.actived=[]
        #    return None

        self.reset()
        self.input(img)
        results=[]
        for n in self.pallium:
            n.calcValue()
            #if(n.axon.outneurons!=[]):
            #    results.append(n)
            #if(n.axon.outneurons!=[] and n.dendritic.value>=len(n.dendritic.synapses)):
            #    #return n
            #    results.append(n)



        #list1 = sorted(results, key=lambda n: n.dendritic.value, reverse=True)
        #if len(list1)>0:
        #    nmax=list1[0]# may be mut, not top, how can i do? = .value=28*28
        #else:
        #    return None
        #return nmax
        lst = [v for k, v in self.knowledges.items()]
        list1 = sorted(lst,key = lambda n:n.value,reverse=True)
        #lst = sorted(list(self.knowledges.items()), key=lambda k: k[1].value, reverse=True)
        return list1

    def look(self, img):  # get seem overlap for memory

        self.reset()
        self.input(img)
        for i in self.palliumidx:
            self.pallium[i].calcValue()
            #print(i,self.pallium[i].value)
            #if(n.axon.outneurons!=[]):
            #    results.append(n)
            #if(n.axon.outneurons!=[] and n.dendritic.value>=len(n.dendritic.synapses)):
            #    #return n
            #    results.append(n)



        #list1 = sorted(results, key=lambda n: n.dendritic.value, reverse=True)
        #if len(list1)>0:
        #    nmax=list1[0]# may be mut, not top, how can i do? = .value=28*28
        #else:
        #    return None
        #return nmax
        lst = [v for k, v in self.knowledges.items()]
        list1 = sorted(lst,key = lambda n:n.value,reverse=True)
        #lst = sorted(list(self.knowledges.items()), key=lambda k: k[1].value, reverse=True)
        return list1



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
    def train_20190102(self,imgs,labels):
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
            #print("i:",i)

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
                    #print("####Error: %s predict %s ,need renew memory "%(imglabel,str(lbs)) )
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
                        #print("redo:",imglabel,end=" ")
                        goto .predictlabel
                        #elf.learn(img,h.label) #use goto can instead diguisuanfa
        
                else:
                    if(len(lbs)>1):
                        print(i,"%s predict more than one: %s "%(labels[i],str(lbs)) )
                    #else:
                    #    print(lbs[0],end=" ")

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
                #memory = self.remember(imgs[i],labels[i])
                self.learn(imgs[i],labels[i])
                #lb = self.predict(imgs[i])
                ok = False
                break

        if ok==False:
            goto .trainlabel

    @with_goto
    def train(self, imgs, labels):
        label .trainlabel
        for i in range(imgs.shape[0]):
            # if labels[i] not in [4,9,7]:
            #    continue
            if (i%100 == 0):
                print("i:",i)
                #maybe memory error
                b = 0
                pass
            if (i>=3):
                b=0
            self.learn(imgs[i],labels[i])
            if (i == 75):
                nu, alike, ilayer,nubest = self.inference(imgs[33])
                if(alike != None):
                    print (i,labels[33],alike.axon.outneurons[0].label)
                else:
                    print (i,None)

                b = 0

        #self.status()
        #self.reform()
        #self.status()
        label .testlabel
        ok = True
        for i in range(imgs.shape[0]):
            # if labels[i] not in [4,9,7]:
            #    continue
            if (i == 89):
                b = 0
            #print(i,end=" ")
            #sys.stdout.flush()
            #self.learn(imgs[i], labels[i])
            lb=self.predict(imgs[i])
            if (labels[i] != lb):
                print(i, "Error: %s predict %s " % (labels[i], lb))
                self.learn(imgs[i], labels[i])
                ok = False

        if ok == False:
            print("Learn again...")
            goto .testlabel
        print("End train.")

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

    @with_goto
    def learn(self,img,imglabel,memory=None):#recursion learn
        #if memory == None:
        #    memory = self.remember(img, imglabel)

        label.predictlabel
        ilayer,mostsimilar,lastact = self.inference(img)
        know = self.getknow(imglabel)
        if (mostsimilar == None):  # not found,create new
            nu=neuron()
            if(lastact!=None):
                nu.dendritic.connectfrom(lastact.axon,1)#connect to down layer
            for n in self.layers[-ilayer].flat:#use up layer
                nu.dendritic.connectfrom(n.axon, n.value)
            nu.axon.outneurons.append(know)
            self.infrneurons[-ilayer].append(nu)
        else:
            lbs = [n.label for n in mostsimilar.axon.outneurons]
            if (len(lbs) > 1):
                print(imglabel," predict more than one: ",lbs)
            if imglabel not in lbs:
                nu = neuron()
                if (lastact != None):
                    nu.dendritic.connectfrom(lastact.axon, 1)  # connect to down layer
                #print(ilayer)
                for n in self.layers[-ilayer].flat:  # use up layer
                    nu.dendritic.connectfrom(n.axon, n.value)
                nu.axon.outneurons.append(know)
                self.infrneurons[-ilayer].append(nu)
                return False
        return None

    @with_goto
    def learn_20190103(self,img,imglabel,memory=None):#recursion learn
        #if memory == None:
        #    memory = self.remember(img, imglabel)

        self.feel(img)

        label.predictlabel
        nu, alike, ilayer,mostsimilar = self.inference(img)
        know = self.getknow(imglabel)
        if (alike == None):  # not found,create new
            if mostsimilar!=None and mostsimilar.axon.outneurons[0].label==imglabel:
                mostsimilar.memory.append(imglabel)
            else:
                nu.axon.outneurons.append(know)
                nu.memory = [imglabel]
                self.infrneurons[-ilayer].append(nu)
        else:
            if len(alike.memory)>0:
                lbs = [n.label for n in alike.axon.outneurons]
                if (len(lbs) > 1):
                    print(i, "%s predict more than one: %s " % (labels[i], str(lbs)))
                if imglabel in lbs:
                    alike.memory.append(imglabel)
                    return
                else:
                    if mostsimilar != None \
                            and mostsimilar.axon.outneurons[0].label == imglabel:
                        mostsimilar.memory.append(imglabel)
                        return
                    alike.memory.clear()
                    nu.axon.outneurons.append(know)
                    nu.memory = [imglabel]
                    self.infrneurons[-ilayer].append(nu)
            else:
                if mostsimilar != None \
                        and mostsimilar.axon.outneurons[0].label == imglabel:
                    mostsimilar.memory.append(imglabel)
                    return
                alike.memory.clear()
                nu.axon.outneurons.append(know)
                nu.memory = [imglabel]
                self.infrneurons[-ilayer].append(nu)


    @with_goto
    def learn_withmemory(self,img,imglabel,memory=None):#recursion learn
        if memory == None:
            memory = self.remember(img, imglabel)

        self.feel(img)

        label.predictlabel
        nu, alike, ilayer = self.inference(img)
        know = self.getknow(imglabel)
        if (alike == None):  # not found,create new
            nu.axon.outneurons.append(know)
            nu.memory = [memory]
            self.infrneurons[-ilayer].append(nu)
        else:
            lbs = [n.label for n in alike.axon.outneurons]
            if (imglabel not in lbs):  # found but not this label, need create new, and renew history memory
                # print("####Error: %s predict %s ,need renew memory "%(imglabel,str(lbs)) )
                his = alike.memory.copy()
                alike.memory.clear()
                nu.axon.outneurons.append(know)
                nu.memory = [memory]
                self.infrneurons[-ilayer].append(nu)
                for n in his:
                    # pltshow(img,imglabel)
                    self.clearneurons()
                    n.reappear()
                    img = self.output()
                    imglabel = n.axon.outneurons[0].label
                    memory = n
                #    # pltshow(img,imglabel)
                #    # print("redo:",imglabel,end=" ")
                    self.learn(img,imglabel,memory)
                #    # elf.learn(img,h.label) #use goto can instead diguisuanfa

            else:
                alike.memory.append(memory)
                if (len(lbs) > 1):
                    print(i, "%s predict more than one: %s " % (labels[i], str(lbs)))
                # else:
                #    print(lbs[0],end=" ")


            if (imglabel not in lbs):  # found but not this label, need create new and renew history
                # print("####Error: %s predict %s ,need renew memory "%(imglabel,str(lbs)) )
                his = alike.memory.copy()


    def learn_20190102(self, img, label):#digui is slow
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
        nlist = self.look(img)  # found mutipy?
        #remember every thing diffrence
        if (nlist!=[]):
            lb=nlist[0].label
            if lb == label: #allow mistake ,train twice
                return nlist[0].actived
            #if nmax.dendritic.value==self.ROWS*self.COLS:
            #    return nmax
            #else:
            #    print(lb,label,nmax.dendritic.value)

        if (self.knowledges.__contains__(label)):
            # print("Already in,create dendritic only", label)
            nlb = self.knowledges[label]
            n=neuron()
            self.palliumidx.append(len(self.pallium))
            self.pallium.append(n)
            nlb.inaxon.append(n)
            n.axon.outneurons.append(nlb)
            d=n.dendritic
        else:
            nlb = neuron()
            self.knowledges[label]=nlb
            nlb.label=label
            n=neuron()
            self.palliumidx.append(len(self.pallium))
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
                    d.connectfrom(self.neurons[i, j].axon, 0)
        return n

    def reform_190208(self):
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
                for (axon, polarity) in dset:
                    d.disconnectfrom(axon,polarity)
                if d.synapses==[]:
                    print("No synapses,delete dendritic")
                    for ss in d.connectedNeuron.axon.synapses:
                        ss.axon=n.axon
                    self.pallium.remove(d.connectedNeuron)
                    del d.connectedNeuron
                    del d
                else:
                    d.connectfrom(n.axon,1)

        #
        listlen=len(list1)
        for i in range(listlen):
            for j in range(i,listlen):
                ka,va=list1[i]
                kb,vb=list1[j]



        self.sortpallium()

    #all img same dendritics count first ,imgs count first
    @with_goto
    def reform_A(self):
        label .lbtryform

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
        blDone=True
        for (k, v) in list1: #
            #v=dendrtic cnt
            if (v < 2):# 4: 2*4s+4d => 2s+1d+4s+1d+1n
                break
            dset = dictset[k]
            ncnt=len(dset)
            if(ncnt<2):# dendritic must have up 2 synapses
                continue
            #if(ncnt*v<8):
            #    continue

            blDone=False

            ds = dictdendritic[k]

            n = neuron()
            self.pallium.append(n)
            for (axon,polarity) in dset:
                n.dendritic.connectfrom(axon,polarity)

            for d in ds:
                for (axon, polarity) in dset:
                    d.disconnectfrom(axon,polarity)
                if d.synapses==[]:
                    print("No synapses,delete dendritic")
                    for ss in d.connectedNeuron.axon.synapses:
                        ss.axon=n.axon
                    self.pallium.remove(d.connectedNeuron)
                    del d.connectedNeuron
                    del d
                else:
                    d.connectfrom(n.axon,1)
        if blDone==False:#need resum
            print("Re count...")
            #goto .lbtryform
        #
        listlen=len(list1)
        i=0
        #blDone=True
        while i < listlen-1:
            for j in range(i+1,listlen):
                ka,va=list1[i]
                kb,vb=list1[j]
                seta=dictdendritic[ka]
                setb=dictdendritic[kb]
                daset = dictset[ka]
                dbset = dictset[kb]
                ab = seta & setb
                if len(ab)>3:
                    print(i,j,len(ab))
                    #if i==6:
                    #    i=9999
                    #    break
                    n=neuron()
                    self.pallium.append(n)
                    d=n.dendritic
                    for (axon, polarity) in daset:
                        d.connectfrom(axon, polarity)
                        for dab in ab:
                            dab.disconnectfrom(axon,polarity)
                    for (axon, polarity) in dbset:
                        d.connectfrom(axon, polarity)
                        for dab in ab:
                            dab.disconnectfrom(axon,polarity)

                    for dab in ab:
                        dab.connectfrom(n.axon,1)# must 1

                    blDone = False
                i=j+1
                break

        if blDone == False:
            print("Do again...")
            goto .lbtryform


        self.sortpallium()
    #2 img same count first, pix cout first

    @with_goto
    def reform(self):
        lastmaxlen=28*28

        #2 img pix count first
        label .lbtryform
        blDone = True
        dendritics = [n.dendritic for n in self.pallium]
        listlen=len(dendritics)
        maxlen = 0
        i=0
        while i < listlen-1:
            for j in range(i+1,listlen):
                apa=[(s.axon,s.polarity) for s in dendritics[i].synapses]
                apb = [(s.axon, s.polarity) for s in dendritics[j].synapses]
                seta=set(apa)
                setb=set(apb)
                ab = seta & setb
                if maxlen<len(ab):
                    maxlen=len(ab)
                    print(i,j,maxlen)
                    maxidx=(i,j)
                    if(maxlen>=lastmaxlen):
                        i=listlen
                        break
            i=i+1

        lastmaxlen=maxlen
        if(maxlen>3):
            i,j=maxidx
            apa = [(s.axon, s.polarity) for s in dendritics[i].synapses]
            apb = [(s.axon, s.polarity) for s in dendritics[j].synapses]
            seta = set(apa)
            setb = set(apb)
            ab = seta & setb
            print(i,j,len(ab))
            n=neuron()
            self.pallium.append(n)
            d=n.dendritic
            for (axon, polarity) in ab:
                d.connectfrom(axon, polarity)
                dendritics[i].disconnectfrom(axon,polarity)
                dendritics[j].disconnectfrom(axon, polarity)

            dendritics[i].connectfrom(n.axon,1)# must 1
            dendritics[j].connectfrom(n.axon, 1)  # must 1

            blDone = False


        if blDone == False:
            print("Do again...")
            goto .lbtryform

        #delete on snapse neuron
        lenp=len(self.pallium)
        #print(lenp)
        for i in range(lenp-1,-1,-1):
            n=self.pallium[i]
            if len(n.dendritic.synapses)==1:
                ns=n.dendritic.synapses[0]
                if ns.polarity==1:
                    print("one synapse, delete it.")
                    for s in n.axon.synapses:
                        d=s.dendritic
                        d.disconnect(s)
                        d.connectfrom(ns.axon,ns.polarity*s.polarity)
                    n.dendritic.disconnect(ns)
                    self.pallium.remove(n)
                    del n
                else:
                    print("??one synapse but polarity=",ns.polarity)
        #lenp=len(self.pallium)
        #print(lenp)

        self.sortpallium()

    def sortpallium(self):
        print("Sorting...")
        vidx=[]
        lenpallium=len(self.pallium)
        flag=np.zeros(lenpallium)#flag idxed
        while len(vidx)<lenpallium:
            for i in range(lenpallium):
                if(flag[i]==1):
                    continue

                n=self.pallium[i]
                hasin=False
                for s in n.dendritic.synapses:
                    if s.axon.connectedNeuron in self.pallium \
                            and flag[self.pallium.index(s.axon.connectedNeuron)]==0:
                        hasin = True
                        break
                if hasin == False:
                    vidx.append(i)
                    flag[i]=1
                    #self.pallium.remove(n)
        self.palliumidx=vidx
        print(vidx)
    def sortpallium_20190208(self):
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

    def reforminference(self):
        dictaxonpolarity = {}
        dictlen = {}
        dictdendritic = {}
        dictset={}
        for layer in self.layers:
            for nrow in layer:
                for n in nrow:
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
        for layer in self.infrneurons:
            for n in layer:
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

        #self.sortinference()

    def predict(self,img):
        ilayer, mostsimilar, lastact = self.inference(img)
        if mostsimilar != None:
            lbs = [n.label for n in mostsimilar.axon.outneurons]
            if len(lbs) > 0:
                if len(lbs) > 1:
                    print("Have two label:", lbs)
                return lbs[0]
        return None
        #
        # if (alike == None):  # not found,create new
        #     if mostsimilar!=None :
        #         lbs = [n.label for n in mostsimilar.axon.outneurons]
        #         if len(lbs)>0:
        #             if len(lbs)>1:
        #                 print("Have two label:",lbs)
        #             return lbs[0]
        #     return  None
        # else:
        #     if len(alike.memory)>0:
        #         lbs = [n.label for n in alike.axon.outneurons]
        #         if(len(lbs)>0):
        #             if (len(lbs) > 1):
        #                 print("predict more than one:" ,lbs)
        #             return lbs[0]
        #         else:
        #             print("Error,No outneurons.")
        #             return None
        #     else:
        #         if mostsimilar != None :
        #             lbs = [n.label for n in mostsimilar.axon.outneurons]
        #             if len(lbs) > 0:
        #                 if len(lbs) > 1:
        #                     print("Have two label:", lbs)
        #                 return lbs[0]
        #         return None


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

    #input img
    #output nu,lastactived,lastlayer,mostsimilar
    #nu,current layer new neuron connect form,lastactived:lastlayer actived neuron,ilayer:lastlayer
    #mostsimilar:current layer most similar neuron
    @with_goto
    def inference(self,img):#inference
        self.feel(img)

        lastact = None
        ilayer = 1
        lastinflayer=[n for n in self.infrneurons[-ilayer]]

        label .lbtry
        vmax = -1
        mostsimilar = None
        blAct = False
        for n in lastinflayer:
            n.calcValue()
            if n.dendritic.value > vmax:
                vmax = n.dendritic.value
                mostsimilar = n

            if n.dendritic.value >= len(n.dendritic.synapses):  # actived
                ilayer += 1
                lastact = n
                lastinflayer =[ s.dendritic.connectedNeuron for s in n.axon.synapses]
                if lastinflayer!=[]:
                    goto .lbtry
                else:
                    #last layer and not have same actived
                    blAct = True
                    mostsimilar = n
                    break

        return ilayer,mostsimilar,lastact #if mostsimilar is actived ilayer is up layer else ilayer is current layer



    def recall(self, img):#recall from history
        #img=self.sdr(img)
        nlist = self.look(img)
        return nlist[0].label
        #nlist[0].actived
        #if(nlist !=None):
        #             return nlist[0].label
        #else:
        #    return None

    def reappear(self,label):
        n=self.knowledges[label]
        n.reappear()
        img=self.output()
        print(img)
    def output(self):
        img=np.zeros((self.ROWS,self.COLS),np.int)
        for i in range(self.ROWS):
            for j in range(self.COLS):
                img[i,j]=self.neurons[i,j].value
        return img
    def outputlayer(self,layer):
        R,C = layer.shape
        img=np.zeros((R,C),np.int)
        for i in range(R):
            for j in range(C):
                img[i,j]=layer[i,j].value
        return img

    def status(self):
        #dds=[n.dendritic for n in self.pallium]

        s=0
        for n in self.pallium:
            cnt=len(n.dendritic.synapses)
            s+= cnt
            if cnt==1:
                print ("Only one synapse",n,n.dendritic.synapses[0].polarity)
            #else:
            #    for ds in n.dendritic.synapses:
            #        print(ds.axon.connectedNeuron.pos,ds.polarity)

        cinf=0
        for layer in self.infrneurons:
            for n in layer:
                cinf+=1
                cnt=len(n.dendritic.synapses)
                s+= cnt
                if cnt==1:
                    print ("Only one synapse",n)

        print("synapses:",s,"pallium:",len(self.pallium),"infnu:",cinf)
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

