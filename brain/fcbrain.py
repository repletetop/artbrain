#-*-coding:utf-8-*-
import numpy as np

class brain:
    def __init__(self):
        self.neurons=np.array((28,28,14),np.uint8)
        self.t=0

    def compute(self,t):
        for i in range(28):
            for j in range(28):
                vt-1
                self.neurons[i,j,t]=v
        pass

    def input(self,img):
        self.neurons[:,:,0]=img
        self.t=0
