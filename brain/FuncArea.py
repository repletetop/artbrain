#coding= utf-8
'''''''''''''''
@author:893802998@qq.com
'''

from neuron import *
from BrainLayer import BrainLayer
from brain import brain


#huidu 255 zhuanhuawei kongjian weidu 0-255
class FCompare2D:
    def __init__(self):
        self.la=BrainLayer(16,16)
        self.lb=BrainLayer(16,16)
        cmat=[[1,1,1],[1,1,1],[1,1,1]]
        self.la.convolutionfun(self.lb,cmat,axon.actvor)
        self.likeresult=self.lb.getAxonMat

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




