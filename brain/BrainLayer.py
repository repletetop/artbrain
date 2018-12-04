#-*-coding:utf-8-*-
from neuron import neuron,axon
import numpy as np

class BrainLayer:
    def __init__(self, rows, cols):
        self.neurons = np.array([[neuron() for ii in range(cols)]
                                 for jj in range(rows)])

    def input(self, img, row=0, col=0):
        rows = len(img)
        cols = len(img[0])
        for i in range(rows):
            for j in range(cols):
                n = self.neurons[row + i][col + j]
                n.value = img[i][j]
                n.conduct()

    def reset(self):
        for i in range(self.height):
            for j in range(self.width):
                self.neurons[i][j].value = 0

    # angle 0-360,x0 r,y0 c center
    def rotate(self, lb, angle, x0, y0):
        la = self
        a = np.pi * angle / 180
        for i in range(la.neurons.shape[0]):
            for j in range(la.neurons.shape[1]):
                x = int(round((i - x0) * np.cos(a) - (j - y0) * np.sin(a)) + x0)
                y = int(round((i - x0) * np.sin(a) + (j - y0) * np.cos(a)) + y0)
                #print(x, y)
                if (x >= lb.neurons.shape[0] or y >= lb.neurons.shape[1]):
                    continue

                ##print i-x0,j-y0,x-x0,y-y0
                nla = la.neurons[i][j]
                nlb = lb.neurons[x][y]
                nla.connectto(nlb)
                nlb.axon.activation = axon.actvmutply

    @property
    def width(self):
        return len(self.neurons[0])

    @property
    def height(self):
        return len(self.neurons)

    def getAxonMat(self):
        w = len(self.neurons[0])
        h = len(self.neurons)
        img = np.array([[self.neurons[row, col].axon.getValue() for col in range(w)]
                        for row in range(h)])
        return img

    def getMat(self):
        w = len(self.neurons[0])
        h = len(self.neurons)
        img = np.array([[self.neurons[row, col].value for col in range(w)]
                        for row in range(h)])
        return img

    def getMatFilter(self, func):
        return brain.getMatFilter(self, self, 0, 0, self.height, self.width, func)

    def convolutionfun(self, lb,cmat,axonactivation):
        la=self
        ha=la.neurons.shape[0]
        wa=la.neurons.shape[1]
        ra=0
        ca=0
        rb=0
        cb=0
        wc = len(cmat[0])
        hc = len(cmat)
        for j in range(int(wc/2),wa-int(wc/2)):
            for k in range(int(hc/2),ha-int(hc/2)):
                ntmp = lb.neurons[rb+k,cb+j]
                for a in range(wc):
                    for b in range(hc):
                        if cmat[b][a]:
                            nla =la.neurons[ ra+k + b - int(int(hc/2)), ca+j + a - int(wc/2)]
                            nla.connectto(ntmp,cmat[b][a])
                ntmp.axon.activation = axonactivation
