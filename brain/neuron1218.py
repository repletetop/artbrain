#coding= utf-8
'''''''''''''''
@author:893802998@qq.com
'''

'''add htmdentritic

neurons ->axon->synape->dendritic->neurons->axon
     1     1 1  m        1     m    m    1   1
     
neurons connect to synape
                from dendritic
'''
class dendritic:
    def __init__(self):
        self.connectedNeurons=[] #can link to most neuron
        self.synapses=[]
        self.value=0

    def __init__xxx(self,x,y):
        self.__init__()
        self.x=x
        self.y=y

    def connectfrom(self,axon,polarity=1):
        s=synapse(axon,self,polarity)
        self.synapses.append(s)
        axon.synapses.append(s)
        return s


class axon:
    def __init__(self,neuron):
        self.connectedNeuron=neuron
        self.synapses=[]


class synaptosome:
    pass

class synapse:
    def __init__(self,axon,dendritic,polarity=1): #input:axon, output:dendritic
        self.polarity=polarity
        self.axon=axon
        self.dendritic=dendritic

class neuron:
    def __init__(self):
        self.value = 0
        self.dendritic=dendritic() #axon-dendritic
        self.inaxon=[] #axon->neuron
        self.axon=axon(self) #output axon
        return

    def calcValue(self):#only 0,1  actived deactived
        for axon in self.inaxon:
            if axon.connectedNeuron.value>0:
                self.value=1
                return
        #print(self.dendritic)
        if(self.dendritic.value==len(self.dendritic.synapses)):
            self.value=1
            #for axon in self.outaxon:
        else:
            self.value=0

    def conduct(self): #step 0
        #self.axon.conduct() #step 1
        #self.calcValue()
        #v=self.value
        #nv=int(not v)
        for s in self.axon.synapses:
            #if (s.polarity != 0):
                s.dendritic.value += self.value  #step 2 axon=>dendritic
            #else:
            #    s.dendritic.value += nv
        #back send
        if self.axon.synapses!=[]:
            self.value = 0
            #self.dendritic.value=0
        #reduce shengjintizhi





if __name__ =="__main__":
    print ("main")
