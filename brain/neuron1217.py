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

    def connectfrom(self, axon, polarity=1):
        s = synapse(axon, self, polarity)
        self.synapses.append(s)
        axon.synapses.append(s)
        return s

    def connectto(self, n, polarity=1):
        self.connectedNeurons.append(n)
        n.dendritics.append(self)


class axon:
    def __init__(self, neuron):
        self.connectedNeuron = neuron
        self.synapses = []
        self.criticalvalue = 1  # >=1...1

    def getValue(self):  # getMax,Mean,fazhi,And,Or,Not....
        if hasattr(self, "activation"):
            return self.activation(self.connectedNeuron)
        # logic else
        return self.connectedNeuron.value
        # return int(self.connectedNeuron.value>=self.criticalvalue)

    def conduct(self):  # step 1 neuron=>axon
        v = self.getValue()
        #        if(v):
        #            self.connectedNeuron.value-=1
        for s in self.synapses:
            if (s.polarity != 0):
                s.dendritic.value += v * s.polarity  # step 2 axon=>dendritic
            else:
                s.dendritic.value += int(not v)
            # if(s.dendritic.value/len(s.dendritic.synapses)>s.dendritic.threshold):# setp 3 htmdendritic=>next neuron
            #    for n in s.dendritic.connectedNeurons:
            #        n.value = 1 #actived

    def actvcritcalvalue(n):
        return int(n.value >= n.axon.criticalvalue)

    def actvand(n):
        for s in n.dendritic.synapses:
            if s.value == 0:
                return 0
        return 1

    def actvandnot(n):
        v = 0
        for s in n.dendritic.synapses:
            if s.value != 0:
                v = v + 1
        if v > 0 and v < len(n.dendritic.synapses):
            return 1
        return 0

    def actvnot(n):  #
        for s in n.dendritic.synapses:
            if s.value == 0:
                return 1
        return 0

    def actvxnot(n):  # all 1 =0 all 0=0 , other=1::::notmean
        for s in n.dendritic.synapses:
            if s.value == 0:
                return 1
        return 0

    def actvor(n):
        for s in n.dendritic.synapses:
            if s.axon.connectedNeuron.value == 1:
                return 1
        return 0

    def actvxor(n):
        v = n.dendritic.synapses[0].value
        for i in range(1, len(n.dendritic.synapses)):
            if (v != n.dendritic.synapses[i].value):
                return 1
        return 0

    def actvmean(n):
        v = n.dendritic.synapses[0].value
        for i in range(1, len(n.dendritic.synapses)):
            v += n.dendritic.synapses[i].value
        return v / len(n.dendritic.synapses)

    def actvpart(n):
        v = n.dendritic.synapses[0].value
        for i in range(1, len(n.dendritic.synapses)):
            v += n.dendritic.synapses[i].value
        if v == 0:
            return 0
        if v != len(n.dendritic.synapses):
            return 1

    def actvsum(n):
        v = 0
        for i in range(len(n.dendritic.synapses)):
            v += n.dendritic.synapses[i].value
        return v

    def actvmax(n):
        v = -999999
        for i in range(len(n.dendritic.synapses)):
            if v < n.dendritic.synapses[i].value:
                v = n.dendritic.synapses[i].value
        return v

    def actvmin(n):
        v = 999999
        for i in range(len(n.dendritic.synapses)):
            if v > n.dendritic.synapses[i].value:
                v = n.dendritic.synapses[i].value
        return v

    def actvmaxidx(n):
        idx = 0
        v = -99999
        for i in range(len(n.dendritic.synapses)):
            if v < n.dendritic.synapses[i].value:
                v = n.dendritic.synapses[i].value
                idx = i
        return idx

    def actvminidx(n):
        idx = 0
        v = 9999999  # max
        for i in range(0, len(n.dendritic.synapses)):
            if v > n.dendritic.synapses[i].value:
                v = n.dendritic.synapses[i].value
                idx = i
        return idx

    def actvselect(n):
        return n.dendritic.synapses[n.dendritic.selected.getValue()].value

    def actvcount(n):
        v = 0
        for i in range(0, len(n.dendritic.synapses)):
            if n.dendritic.synapses[i].value != 0:
                v += 1
        return v

    def actvmutply(n):
        v = 1
        for i in range(len(n.dendritic.synapses)):
            if n.dendritic.synapses[i].value != 0:
                v *= n.dendritic.synapses[i].value
        return v


class synaptosome:
    pass


class synapse:
    def __init__(self, axon, dendritic, polarity=1):  # input:axon, output:dendritic
        self.polarity = polarity
        self.axon = axon
        self.dendritic = dendritic

    def conduct(self):
        v = self.axon.getValue()
        if (self.polarity != 0):
            self.dendritic.connectedNeuron.value += v * self.polarity
        else:
            self.dendritic.connectedNeuron.value += int(not v)


class neuron:
    def __init__(self):
        self.value = 0
        self.dendritics = []
        self.axon = axon(self)
        return
        # init when connect
        # self.dendritic=dendritic(self)
        # self.axon=axon(self)#must have one

    def getValue(self):  # only 0,1  actived deactived
        return self.value

    def conduct(self):  # step 0
        self.axon.conduct()  # step 1


'''
    def connectfrom(self,n,polarity=1):
        if(not hasattr(self,"dendritic")):
            self.dendritic=dendritic(self)
        if(not hasattr(n,"axon")):
            n.axon=axon(n)
        s=synapse(n.axon,self.dendritic,polarity)
        self.dendritic.synapses.append(s)
        n.axon.synapses.append(s)
        return s

    def connectto(self,n,polarity=1):
        if(not hasattr(n,"dendritic")):
            n.dendritic=dendritic(n)
        if(not hasattr(self,"axon")):
            self.axon=axon(self)
        s=synapse(self.axon,n.dendritic,polarity)
        n.dendritic.synapses.append(s)
        self.axon.synapses.append(s)
        return s
    def connecthtmdendritic(self,dendritic):
        self.dendritics.append(dendritic)
        dendritic.connectedNeurons.append(self)
        pass
'''

if __name__ == "__main__":
    print("main")
