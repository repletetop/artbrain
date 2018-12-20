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
    def __init__(self):
        self.connectedNeurons = []  # can link to most neuron
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
        self.dendritic = dendritic()  # axon-dendritic
        self.inaxon = []  # axon->neuron
        self.axon = axon(self)  # output axon
        return

    def calcValue(self):  # zhenghe only 0,1  actived deactived
        for axon in self.inaxon:
            self.value = 1
            return

        self.value = self.dendritic.value


        # print(self.dendritic)
        #if (self.dendritic.value>0 and self.dendritic.value == len(self.dendritic.synapses)):#all synapses activte
        #    self.value += 1
            # for axon in self.outaxon:
        #else:#part active ,use max one#############
        #    self.value = 0

    def conduct(self):  # step 0
        # self.axon.conduct() #step 1
        # self.calcValue()
        v=self.dendritic.value
        #nv=int(not v)
        for s in self.axon.synapses:
            if (s.polarity > 0) :#+1,-1 postive nagative
                s.dendritic.value += (self.value*s.polarity)  # step 2 axon=>dendritic
            else:
                s.dendritic.value += (self.value*s.polarity+1)#0=>1 1=>0
        # back send
        if self.axon.synapses != []:
            self.value = 0
            # self.dendritic.value=0
        # reduce shengjindizhi
    def recall(self):
        self.value=1
        ss=self.dendritic.synapses
        if(ss!=[]):
            for s in self.dendritic.synapses:
                s.axon.connectedNeuron.recall()
            return
        if self.inaxon != []:
            self.inaxon[0].connectedNeuron.recall()
        return


if __name__ == "__main__":
    print("main")
