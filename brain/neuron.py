# coding= utf-8
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
    def __init__(self,neuron):
        self.connectedNeuron = neuron  # can link to most neuron
        self.synapses = []
        self.value = 0

    def __init__xxx(self, x, y):
        self.__init__()
        self.x = x
        self.y = y

    def connectfrom(self, axon, polarity=1):
        s = synapse(axon, self, polarity)
        self.synapses.append(s)
        axon.synapses.append(s)
        return s
    def disconnect(self,ss):
        self.synapses.remove(ss)
        ss.axon.synapses.remove(ss)
        del ss
    def disconnectfrom(self,axon,polarity):
        for s in self.synapses:
            if s.axon==axon and s.polarity==polarity:
                self.disconnect(s)
                return
        print("disconnect not found!")



class axon:
    def __init__(self, neuron):
        self.connectedNeuron = neuron
        self.outneurons=[]
        self.synapses = []


class synaptosome:
    pass


class synapse:
    def __init__(self, axon, dendritic, polarity=1):  # input:axon, output:dendritic
        self.polarity = polarity
        self.axon = axon
        self.dendritic = dendritic


class neuron:
    def __init__(self):
        self.value = 0
        self.dendritic = dendritic(self)  # axon-dendritic
        self.nbdendritic = dendritic(self)
        self.indendritics = [] #dendritic-dendritic ,get Max one actived
        self.nagativeaxons = [] # axon-axon hengxiang nagative
        self.inaxon = []  # axon->neuron
        self.axon = axon(self)  # output axon
        self.actived = False

        return
    def calcDendritic(self):
        v=0
        for s in self.dendritic.synapses:
            if(s.polarity>0):
                v=v+s.axon.connectedNeuron.value
            else:
                v=v+int(not s.axon.connectedNeuron.value)
        self.dendritic.value=v

    def calcNbDendritic(self):#Neighbourhood
        v=0
        for s in self.nbdendritic.synapses:
            if(s.polarity>0):
                v=v+s.axon.connectedNeuron.value
            else:
                v=v+int(not s.axon.connectedNeuron.value)
        self.nbdendritic.value=v


    def calcValue(self):  # zhenghe only 0,1  actived deactived
        #for n in self.inaxon:
        #    self.value = 1#len(self.dendritic.synapses)
        #    return
        v=0
        for s in self.dendritic.synapses:
            if(s.polarity != 0):
                v=v+s.axon.connectedNeuron.value*s.polarity
            else:
                v=v+int(not s.axon.connectedNeuron.value)
        self.dendritic.value=v
        self.value = self.dendritic.value
        #if (self.dendritic.value >= len(self.dendritic.synapses)):
        #    self.value =1
        #else:
        #    self.value = 0
        for n in self.axon.outneurons:
            if n.value < v:
                n.value=v
                n.actived = self


    def conduct(self,actived):  # step 0
        #for n in self.axon.outneurons:
        #    n.value=1
        #    #print (n.label)
        #    if not n in actived:
        #        actived.append(n)
        #    #n.conduct(actived)
        #if hasattr(self,'label'):
        #    if not self in actived:
        #        actived.append(self)
        if self.actived:
            if self.axon.outneurons!=[]:
                if not self in actived:
                    actived.append(self)
                    #if len(self.axon.outneurons)==1:
                    #    return self #founded

        v=self.value
        nv=int(not v)
        for s in self.axon.synapses:
            if (s.polarity > 0) :#+1,-1 postive nagative
                s.dendritic.value += (self.value*s.polarity)  # step 2 axon=>dendritic
            else:
                s.dendritic.value += nv#0=>1 1=>0
            if( s.dendritic.value>=len(s.dendritic.synapses)):
                s.dendritic.connectedNeuron.value=1
                s.dendritic.connectedNeuron.actived=True
                s.dendritic.connectedNeuron.conduct(actived)


    def reappear(self):
        for s in self.dendritic.synapses:
            if s.polarity > 0:
                s.axon.connectedNeuron.value = 1
                s.axon.connectedNeuron.reappear()
            else:
                s.axon.connectedNeuron.value = 0
        return


if __name__ == "__main__":
    print("main")
