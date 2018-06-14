#coding= utf-8
'''''''''''''''
@author:893802998@qq.com
'''

from brain.neuron import *


#huidu 255 zhuanhuawei kongjian weidu 0-255
class FCompare2D:
    def __init__(self):
        self.la=brain.BrainLayer(512,512)
        self.lb=brain.BrainLayer(512,512)
        cmat=[[1,1,1],[1,1,1],[1,1,1]]
        brain.convolution(self.la,self.lb,cmat)
        self.likeresult=self.lb.getMat

        return

    def equal(self,a,b):
        return np.nonzero(a-b)[0].size==0

    def like(self,a,b):
        self.la.input(a)
        self.la.input(-b)
        return self.likeresult()

    #duiqi zhongxingdian bu xuyao trans
    def translike(self,a,b):
        return

    def roatelike(self,a,b):
        la=brain.BrainLayer(a.shape)
        r0=int(a.shape[0]/2)
        c0=int(a.shape[1]/2)
        lb=brain.BrainLayer(a.shape)
        la.rotate(lb,45,r0,c0)

        #brain.convolution(lb,self.la,[1])
        la.input(a)#45
        la.input(lb.getMat())#90
        self.lb.input(lb.getMat())
        self.la.input(-b)
        return self.likeresult()

    def scalelike(self,a,b):
        return




