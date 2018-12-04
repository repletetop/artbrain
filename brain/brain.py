#-*-coding:utf-8-*-
import neuron

class brain:
    def __init__(self):
        self.knowledges = []

    def __init__(self,layers=0,rows=0,cols=0):
        self.knowledges = []




    def diff(self,la,ra,ca,wa,ha, lb,rb,cb, lt,rt,ct):
        for r in range(ha):
            for c in range(wa):
                nt=self.getNeuron(lt,rt+r,ct+c)
                na=self.getNeuron(la,ra+r,ca+c)
                nb=self.getNeuron(lb,rb+r,cb+c)
                na.connectto(nt)
                nb.connectto(nt,-1)


    def Extraction(self,la,ra,ca,wa,ha,lb,rb,cb):
        for c in range(1,wa-1):
            for r in range(1,ha-1):
                nla = self.getNeuron(la, ra + r , ca + c )
                nlb = self.getNeuron(lb, rb + r , cb + c )

                n4=neuron(-1,-1,-1)
                n4.connectfrom(self.getNeuron(la,ra+r,ca+c))
                n4.connectfrom(self.getNeuron(la,ra+r-1,ca+c))
                n4.connectfrom(self.getNeuron(la,ra+r+1,ca+c))
                n4.connectfrom(self.getNeuron(la,ra+r,ca+c-1))
                n4.connectfrom(self.getNeuron(la,ra+r,ca+c+1))
                n4.axon.criticalvalue=5

                m1=neuron(-1,-1,-1) #0/1/1/0
                m2=neuron(-1,-1,-1) #0-1-1-0
                m11=neuron(-1,-1,-1) #0/1/1/0
                m22=neuron(-1,-1,-1) #0-1-1-0
                m3=neuron(-1,-1,-1) #0/1/0
                m4=neuron(-1,-1,-1) #0-1-0
                self.connectMat(m1,la,ra+r,ca+c,[[0],[1],[1],[0]])#?????
                self.connectMat(m11,la,ra+r+1,ca+c,[[0],[1],[1],[0]])
                self.connectMat(m2,la,ra+r,ca+c,[[0,1,1,0]])
                self.connectMat(m22,la,ra+r,ca+c+1,[[0,1,1,0]])
                self.connectMat(m3,la,ra+r,ca+c,[[0],[1],[0]])
                self.connectMat(m4,la,ra+r,ca+c,[[0,1,0]])
                m=neuron(-1,-1,-1)
                m.connectfrom(m1)
                m.connectfrom(m11)
                m.connectfrom(m2)
                m.connectfrom(m22)
                m.connectfrom(m3)
                m.connectfrom(m4)
                m.axon.criticalvalue=1
                #  0 1  1 0
                #  1 0  0 1
                m5=neuron(-1,-1,-5) #0/1/0
                m6=neuron(-1,-1,-6) #0-1-0
                m55=neuron(-1,-1,-55) #0/1/0
                m66=neuron(-1,-1,-66) #0-1-0
                self.connectMat(m5,la,ra+r,ca+c,[[1,0],[0,1]])
                self.connectMat(m55,la,ra+r+1,ca+c+1,[[1,0],[0,1]])
                self.connectMat(m6,la,ra+r,ca+c+1,[[0,1],[1,0]])
                self.connectMat(m66,la,ra+r+1,ca+c+1,[[0,1],[1,0]])
                m.connectfrom(m5)
                m.connectfrom(m55)
                m.connectfrom(m6)
                m.connectfrom(m66)
                #    0 1 x
                #    1 1 0
                m7=neuron(-1,-1,-7)
                m77=neuron(-1,-1,-77)
                self.connectMat(m7,la,ra+r,ca+c,[[0,1,1],[1,1,0]])
                self.connectMat(m77,la,ra+r,ca+c,[[0,1,0],[1,1,0]])
                m.connectfrom(m7)
                m.connectfrom(m77)
                #    - - -
                #    0 1 1
                #    x 1 0
                m8=neuron(-1,-1,-7)
                m88=neuron(-1,-1,-77)
                self.connectMat(m8,la,ra+r+1,ca+c,[[0,1,1],[1,1,0]])
                self.connectMat(m88,la,ra+r+1,ca+c,[[0,1,1],[0,1,0]])
                m.connectfrom(m8)
                m.connectfrom(m88)





                #(n4>0 zxd ) or m>0
                ### Double xiangsu

                n4.connectto(nlb)
                m.connectto(nlb)
                nlb.axon.criticalvalue=1


    def adj(self, la,ra,ca,wa,ha, lb,rb,cb):
        for c in range(1,wa-1):
            for r in range(1,ha-1):
                nla0=self.getNeuron(la,ra+r-1,ca+c-1)
                nla1=self.getNeuron(la,ra+r-1,ca+c)
                nla2=self.getNeuron(la,ra+r-1,ca+c+1)
                nla3=self.getNeuron(la,ra+r,ca+c-1)
                nla4=self.getNeuron(la,ra+r,ca+c)
                nla5=self.getNeuron(la,ra+r,ca+c+1)
                nla6=self.getNeuron(la,ra+r+1,ca+c-1)
                nla7=self.getNeuron(la,ra+r+1,ca+c)
                nla8=self.getNeuron(la,ra+r+1,ca+c+1)

                nlb0=self.getNeuron(lb,rb+r*3,cb+c*3)
                nlb1=self.getNeuron(lb,rb+r*3,cb+c*3+1)
                nlb2=self.getNeuron(lb,rb+r*3,cb+c*3+2)
                nlb3=self.getNeuron(lb,rb+r*3+1,cb+c*3)
                nlb4=self.getNeuron(lb,rb+r*3+1,cb+c*3+1)
                nlb5=self.getNeuron(lb,rb+r*3+1,cb+c*3+2)
                nlb6=self.getNeuron(lb,rb+r*3+2,cb+c*3)
                nlb7=self.getNeuron(lb,rb+r*3+2,cb+c*3+1)
                nlb8=self.getNeuron(lb,rb+r*3+2,cb+c*3+2)

                nla0.connectto(nlb0)
                nla1.connectto(nlb1)
                nla2.connectto(nlb2)
                nla3.connectto(nlb3)
                nla4.connectto(nlb4)
                nla5.connectto(nlb5)
                nla6.connectto(nlb6)
                nla7.connectto(nlb7)
                nla8.connectto(nlb8)

                nla4.connectto(nlb0)
                nla4.connectto(nlb1)
                nla4.connectto(nlb2)
                nla4.connectto(nlb3)
                nla4.connectto(nlb4)
                nla4.connectto(nlb5)
                nla4.connectto(nlb6)
                nla4.connectto(nlb7)
                nla4.connectto(nlb8)
                nlb0.axon.criticalvalue=2
                nlb1.axon.criticalvalue=2
                nlb2.axon.criticalvalue=2
                nlb3.axon.criticalvalue=2
                nlb4.axon.criticalvalue=2
                nlb5.axon.criticalvalue=2
                nlb6.axon.criticalvalue=2
                nlb7.axon.criticalvalue=2
                nlb8.axon.criticalvalue=2




    def convolutionfun(self, la,ra,ca,wa,ha, lb,rb,cb, cmat,fun):
        wc = len(cmat[0]);
        hc = len(cmat)
        for j in range(int(wc/2),wa-int(wc/2)):
            for k in range(int(hc/2),ha-int(hc/2)):
                ntmp = self.getNeuron(lb,rb+k,cb+j)
                for a in range(wc):
                    for b in range(hc):
                        if cmat[b][a]:
                            nla =self.getNeuron(la, ra+k + b - int(int(hc/2)), ca+j + a - int(wc/2))
                            nla.connectto(ntmp,cmat[b][a])
                ntmp.axon.critcalvalue=4
                #ntmp.axon.activation = fun

    @staticmethod
    def filter( la,ra,ca,wa,ha, lb,rb,cb, cmat):
        wc = len(cmat[0]);
        hc = len(cmat)
        ss=[]
        for j in range(int(wc/2),wa-int(wc/2)):
            for k in range(int(hc/2),ha-int(hc/2)):
                ntmp = brain.getNeuron(lb,rb+k,cb+j)
                for a in range(wc):
                    for b in range(hc):
                        nla =brain.getNeuron(la, ra+k + b - int(int(hc/2)), ca+j + a - int(wc/2))
                        s=nla.connectto(ntmp,cmat[b][a])
                        ss.append(s)
                ntmp.axon.criticalvalue = wc*hc
        return ss

    def defilter(self, la,ra,ca,wa,ha, lb,rb,cb, cmat):
        wc = len(cmat[0]);
        hc = len(cmat)
        for j in range(int(wc/2),wa-int(wc/2)):
            for k in range(int(hc/2),ha-int(hc/2)):
                nla = self.getNeuron(la,ra+k,ca+j)
                for a in range(wc):
                    for b in range(hc):
                        if(cmat[b][a]):
                            nlb =self.getNeuron(lb, rb+k + b - int(int(hc/2)), cb+j + a - int(wc/2))
                            nla.connectto(nlb,cmat[b][a])
                nlb.axon.criticalvalue = 1



        for j in range(int(wc/2),wa-int(wc/2)):
            for k in range(int(hc/2),ha-int(hc/2)):
                nla = self.getNeuron(la,ra+k,ca+j)
                for a in range(wc):
                    for b in range(hc):
                        if cmat[b][a]:
                            nlb =self.getNeuron(lb, rb+k + b - int(int(hc/2)), cb+j + a - int(wc/2))
                            nla.connectto(nlb,cmat[b][a])
                #nlb.axon.activation=actvmean


    def conduct(self):
        for s in brain.synapses:
            #if(s.axon.getValue()!=0):
                ##print "===="
                ##print s.axon.connectedNeuron.xyz,s.axon.getValue(),
            s.conduct()
            ##print "=>",s.dendritic.connectedNeuron.xyz,s.dendritic.connectedNeuron.value

    def neuronsconduct(self,la,row,col,w,h):
        for i in range(col,col+w):
            for j in range (row,row+h):
                self.getNeuron(la, j, i).conduct()

    def getValue(self,la,w,h):
        #img =np.array([[self.getNeuron(la, row, col).value for col in range(16) ]for row in range(16)])
        ##print img
        img =np.array([[self.getNeuron(la, row, col).axon.getValue() for col in range(w) ]for row in range(h)])
        return img

    @staticmethod
    def getAxonMat(la,top,left,h,w):
        #img =np.array([[self.getNeuron(la, row, col).value for col in range(16) ]for row in range(16)])
        ##print img
        img =np.array([[brain.getNeuron(la, row, col).axon.getValue() for col in range(left,left+w) ]
                       for row in range(top,top+h)])
        return img

    def getMat(self,la,top,left,h,w):
        #img =np.array([[self.getNeuron(la, row, col).value for col in range(16) ]for row in range(16)])
        ##print img
        img =np.array([[self.getNeuron(la, row, col).getValue() for col in range(left,left+w) ]
                       for row in range(top,top+h)])
        return img

    def getMatFilter(self,la,top,left,h,w,func):
        #img =np.array([[self.getNeuron(la, row, col).value for col in range(16) ]for row in range(16)])
        ##print img
        img =np.array([[ func( self.getNeuron(la, row, col).getValue()) for col in range(left,left+w) ]
                       for row in range(top,top+h)])
        return img

    def compare(self,la,left,top,w,h,lb,leftb,topb):
        fda,disa=self.compareto(la,left,top,w,h,lb,leftb,topb)
        fdb,disb=self.compareto(lb,leftb,topb,w,h,la,left,top)
        return fda+fdb,disa+disb

    def compareto(self,la,left,top,w,h,lb,leftb,topb):
        maxw=w/2
        founds=[]
        dist=[]
        for j in range(w):
            for k in range(h):
                nla=self.getNeuron(la, top+k, left+j)
                if(nla.axon.getValue()==0):
                    continue
                ifound = []
                for i in range(maxw):
                    for jj in range(-i/2,1+i/2):
                        for kk in range(-i/2,1+i/2):
                            try:
                                nlb=self.getNeuron(lb,topb+k+kk,leftb+j+jj)
                                if nlb.axon.getValue()!=0:
                                    ifound =[kk,jj]
                                    break
                            except :
                                continue
                                ##print e
                    if (len(ifound)>0):break
                if (len(ifound)>0):
                    founds.append(ifound)
                    dist.append(jj*jj+kk*kk)
                else:
                    founds.append([maxw,maxw])

        ##print  "founds:",founds
        ##print "mean:%d,std:%d" %(np.mean(dist),np.std(dist))

        return founds,dist


    def pooling(self, la,row,col,wa,ha, lb,rb,cb,wp,hp):#err
        for j in range(wa):
            for k in range(ha):
                nla = self.getNeuron(la, row+k,col+j )
                nlb = self.getNeuron(lb,  rb+k/hp,cb+j /wp  )
                nla.connectto(nlb,1)
                nlb.axon.criticalvalue=(wp*hp+1)/2


    def maxpooling(self,  la,row,col,wa,ha, lb,rb,cb,wp,hp):
        for j in range(wa):
            for k in range(ha):
                nla = self.getNeuron(la, row+k,col+j )
                nlb = self.getNeuron(lb,  rb+k/hp,cb+j /wp  )
                nla.connectto(nlb,1)
                #nlb.axon.criticalvalue=(wp*hp+1)/2


    def overlappingmaxpooling(self,la,wa,ha,lb,wc,hc):
        for w in range(wa):
            for h in range(ha):
                m=-999
                for iw in range(wc):
                    for ih in range(hc):
                        if(self.getNeuron(la,w+iw,h+ih)>m):
                            m=self.getNeuron(la,w+iw,h+ih)
                self.setNeuron(lb,w,h,m)

    def meanpooling(self, la,wa,ha, lb,wc,hc):
        for w in range(wa/wc):
            for h in range(ha/hc):
                nlb=self.getNeuron(lb,h,w)
                nlb.axon.activation = actvmean
                for iw in range(wc):
                    for ih in range(hc):
                        nla=self.getNeuron(la,w*wc+iw,h*hc+ih)
                        nla.connectto(nlb, 1)

    def getHeng(self,la,w,h,lb):
        wc = 8;
        for j in range(1,w-1):
            for k in range(h):
                ntmp = self.getNeuron(lb,k,j)
                for a in range(wc):
                    nla = self.getNeuron(la, k, j + a - int(wc/2))
                    nla.connectto(ntmp, 1)
                ntmp.axon.activation=actvmean

    def getCircle(self,la,w,h,lb):
        wc = 8;
        for j in range(int(wc/2),w-int(wc/2)):
            for k in range(int(wc/2),h-int(wc/2)):
                ntmp = self.getNeuron(lb,k,j)
                for a in range(wc):
                    for b in range(wc):
                        if((a-4)*(a-4)+(b-4)*(b-4)==4*4):
                            nla = self.getNeuron(la, k, j + a - int(wc/2))
                            nla.connectto(ntmp, 1)
                ntmp.axon.activation=actvmean

    #x0,y0 yuandian
    def rotate(self,la,angle,x0,y0,lb):

        a=np.pi*angle/180
        for i in range(la.neurons.shape[0]):
            for j in range(la.neurons.shape[1]):
                x=int(round((i-x0)*np.cos(a)-(j-y0)*np.sin(a))+x0)
                y=int(round((i-x0)*np.sin(a)+(j-y0)*np.cos(a))+y0)
                print(x,y)
                if(x>=lb.neurons.shape[0] or y>=lb.neurons.shape[1]):
                    continue

                ##print i-x0,j-y0,x-x0,y-y0
                nla = self.getNeuron(la,i,j)
                nlb = self.getNeuron(lb,x,y)
                nla.connectto(nlb)
                nlb.axon.activation=actvmutply

    def getArea(self,la):
        nsumr=self.getNeuron(la,26,26)
        nsumr.axon.activation=actvcount
        for i in range(16):
            nr=self.getNeuron(la,i,26)
            for j in range(16):
                nla=self.getNeuron(la,i,j)
                nla.connectto(nr)
            nr.connectto(nsumr)

        nsumc=self.getNeuron(la,26,27)
        nsumc.axon.activation=actvcount
        for i in range(16):
            nc=self.getNeuron(la,26,i)
            for j in range(16):
                nla=self.getNeuron(la,j,i)
                nla.connectto(nc)
            nc.connectto(nsumc)

        narea=self.getNeuron(la,16,16)
        narea.axon.activation=actvmutply
        nsumc.connectto(narea)
        nsumr.connectto(narea)

    @staticmethod
    def GetFeatureBezierO(img):
        h = len(img)
        wd = len(img[0])
        feature = []
        layerImg = brain.BrainLayer(h, wd)
        layerImg.input(img, 0, 0)
        bzLay=brain.BrainLayer(h,wd)
        found=False
        for n in range(wd-1, int(int(wd/3)), -1):
            for b in range(n, wd + 1):
                a = n
                p = [[0, 0], [0, b], [a, b], [a, 0], [0, 0], [0, b], [a, b], [a, 0]]
                cmat = brain.getBezier(p)
                bzLay.reset()
                ss=brain.convolution(layerImg, 0, 0, 16, 16, bzLay, 0, 0, cmat)
                for s in ss:
                    s.conduct()

                r = brain.getAxonMat(bzLay, 0, 0, wd, wd)
                if (r.sum() > 0):
                    found=True
                    break


                p = [[0, 0], [0, a], [b, a], [b, 0], [0, 0], [0, a], [b, a], [b, 0]]
                cmat = brain.getBezier(p)
                ##print cmat
                bzLay.reset()
                ss=brain.convolution(layerImg, 0, 0, 16, 16, bzLay, 0,  0, cmat)
                for s in ss:
                    s.conduct()
                r = brain.getAxonMat(bzLay, 0, 0, wd, wd)
                if (r.sum() > 0):
                    ##print r
                    found=True
                    break
            else:
                continue
            break
            #if(blExitFor):
            #    break

        if(not found):
            return img,found

        bzMat = brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzLay, 0, 0, 16, 16, bzMat, 0, 0, cmat)
        for s in ss:
            s.conduct()

        bzTmp=brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzMat, 0, 0, 16, 16, bzTmp, 0, 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        for s in ss:
            s.conduct()

        c = bzTmp.getAxonMat()
        img = img - c
        layerImg.input(img, 0, 0)
        img = layerImg.getAxonMat()

        return img, found

    @staticmethod
    def getFeatures(img):
        feature=[]

        img, f=brain.GetFeatureBezierO(img)
        if(f):
            feature.append(1)
        else:
            feature.append(0)
            if(np.sum(img)==0):return feature
        img, f=brain.GetFeatureBezier2(img)
        if(f):
            feature.append(1)
        else:
            feature.append(0)
            if (np.sum(img) == 0): return feature

        img,f=brain.GetFeatureLineRow(img)
        if(f):
            feature.append(1)
        else:
            feature.append(0)
            if (np.sum(img) == 0): return feature
        img,f=brain.GetFeatureLineCol(img)
        if(f==True):
            feature.append(1)
        else:
            feature.append(0)
            if (np.sum(img) == 0): return feature

        return feature




    class Knowledge:
        def __init__(self,img,label):
            self.label=label
            rows=len(img)
            cols=len(img[0])
            self.layer=brain.BrainLayer(rows,cols)
            self.layer.input(img,0,0)
            self.feature=brain.getFeatures(img)


        def compareto(self, la, lb):
            left=0; top=0; w=len(la.neurons[0]); h=len(la.neurons);
            leftb=0; topb=0;
            maxw = w / 2
            founds = []
            dist = [0]
            for j in range(w):
                for k in range(h):
                    nla =la.neurons[ top + k][ left + j]
                    if (nla.axon.getValue() == 0):
                        continue
                    ifound = []
                    for i in range(maxw):
                        for jj in range(-i / 2, 1 + i / 2):
                            for kk in range(-i / 2, 1 + i / 2):
                                try:
                                    nlb = lb.neurons[ topb + k + kk][ leftb + j + jj]
                                    if nlb.axon.getValue() != 0:
                                        ifound = [kk, jj]
                                        break
                                except :
                                    continue
                                    # #print e
                        if (len(ifound) > 0): break
                    if (len(ifound) > 0):
                        founds.append(ifound)
                        dist.append(jj * jj + kk * kk)
                    else:
                        founds.append([maxw, maxw])
                        dist.append(maxw*maxw+maxw*maxw)
            # #print  "founds:",founds
            # #print "mean:%d,std:%d" %(np.mean(dist),np.std(dist))
            return founds,dist

        def match(self,layerImg):
            fda, disa = self.compareto(self.layer,layerImg)
            fdb, disb = self.compareto(layerImg,self.layer)
            ##print disa,disb
            ##print(np.std(disa),np.std(disb))
            w=len(self.layer.neurons)
            d=w*w*2/16 # (w/4)^2 + (w/4)^2 = w*w/16+w*w/16  +-w/4point
            ##print np.mean(disa),np.mean(disb) #Max dist 2*2
            return 100-100*np.mean(disa+disb)/d #bianyixishu std/mean 0*1, 0+1/2
            #return 100 #100%

    @staticmethod
    def center(mn):
        w= len(mn[0])
        h= len(mn)
        left = -1;
        right = w+1;
        top = -1;
        bottom = h+1
        for n in range(h):
            if (mn[n, :].max() > 0 and top == -1):
                top = n
            if (mn[-n-1, :].max() > 0 and bottom == h+1):
                bottom = h - n

        for n in range(w):
            if (mn[:, n].max() > 0 and left == -1):
                left = n
            if (mn[:, -n-1].max() > 0 and right == w+1):
                right = w - n

        new = mn[top :bottom  , left :right ]
        return new

    @staticmethod
    def Bezier2(x, yt, t):
        xt = x[0] * yt * yt + x[1] * 2 * yt * t + x[2] * t * t
        return xt

    @staticmethod
    def getBezier2(points):
        x, y = zip(*points)
        wd=max(x)*2
        h=max(y)*2
        hz = np.array([[0] * wd] * h)
        #hz[y[0]][x[0]] = 1
        rate = wd +h
        for tt in range(rate):
            t = 1.0 * tt / rate
            yt = 1 - t
            xt = brain.Bezier2(x, yt, t)
            yt = brain.Bezier2(y, yt, t)
            # #print t,xt,yt
            try:
                hz[int(round(yt))][int(round(xt))] = 1
                ##print int(yt),int(xt)
            except:
                print (wd, yt, xt)
                #print hz
        hz = brain.center(hz)
        return hz

    @staticmethod
    def GetFeatureBezier2(img):
        h = len(img)
        wd = len(img[0])
        feature = []
        layerImg = brain.BrainLayer(h, wd)
        layerImg.input(img, 0, 0)

        bzLay=brain.BrainLayer(h,wd)
        found=False
        for n in range(wd-1, int(wd/3), -1):
            for b in range(n , wd ):
                a = n
                # p = [[0, 0], [0, b], [a, b], [a, 0], [0, 0], [0, b], [a, b], [a, 0]]
                p = [[0, b], [a / 2, 0], [a, b]]
                cmat = brain.getBezier2(p)
                ##print cmat
                bzLay.reset()
                ss=brain.convolution(layerImg, 0, 0, 16, 16, bzLay, 0, 0, cmat)
                for s in ss:
                    s.conduct()
                r = brain.getAxonMat(bzLay, 0, 0, wd, wd)
                if (r.sum() > 0):
                    found=True
                    break;


                p = [[0, a], [b / 2, 0], [b, a]]
                cmat = brain.getBezier2(p)
                ##print cmat
                bzLay.reset()
                brain.convolution(layerImg, 0, 0, 16, 16, bzLay, 0,  0, cmat)
                for s in ss:
                    s.conduct()
                r = brain.getAxonMat(bzLay, 0, 0, wd, wd)
                if (r.sum() > 0):
                    found=True
                    break;
            else:
                continue
            break

        if(not found):
            return img,found

        bzMat = brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzLay, 0, 0, 16, 16, bzMat, 0, 0, cmat)
        for s in ss:
            s.conduct()

        bzTmp=brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzMat, 0, 0, 16, 16, bzTmp, 0, 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        for s in ss:
            s.conduct()

        c = bzTmp.getAxonMat()
        img = img - c
        layerImg.input(img, 0, 0)
        img = layerImg.getAxonMat()

        return img, found

    @staticmethod
    def getLine(len,a):
        h = abs(int(len * np.sin(np.pi * a / 180)))+1
        w = abs(int(len * np.cos(np.pi * a / 180)))+1
        mat = np.array([[0] * w * 2] * h * 2)
        if(h<w):
            for x in range(w):
                y=int(x*np.sin(np.pi*a/180))
                mat[h+y,w+x]=1
        else:
            for y in range(h):
                x=int(y*np.cos(np.pi*a/180))
                mat[h+y,w+x]=1
        mat=brain.center(mat)
        return mat


    @staticmethod
    def GetFeatureLineRow(img):
        rows = len(img)
        cols = len(img[0])
        feature = []
        layerImg = brain.BrainLayer(rows, cols)
        layerImg.input(img, 0, 0)

        wd = len(layerImg.neurons[0])
        h=len(layerImg.neurons)
        bzLay=brain.BrainLayer(h,wd)
        found=False
        for n in range(int(wd/2), int(wd/2)-1,-1):
            for a in range(-10,10,2):
                cmat = brain.getLine(n,a)
                ##print cmat
                bzLay.reset()
                ss=brain.convolution(layerImg, 0, 0, 16, 16, bzLay, 0, 0, cmat)
                for s in ss:
                    s.conduct()
                r = brain.getAxonMat(bzLay, 0, 0, wd, wd)
                if (r.sum() > 0):
                    ##print r
                    found=True
                    break
            else:
                continue
            break


        if(not found):
            return img,found

        bzMat = brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzLay, 0, 0, 16, 16, bzMat, 0, 0, cmat)
        for s in ss:
            s.conduct()

        bzTmp=brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzMat, 0, 0, 16, 16, bzTmp, 0, 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        for s in ss:
            s.conduct()

        c = bzTmp.getAxonMat()
        img = img - c
        layerImg.input(img, 0, 0)
        img = layerImg.getAxonMat()

        return img, found

    @staticmethod
    def GetFeatureLineCol(img):
        h = len(img)
        wd = len(img[0])
        feature = []
        layerImg = brain.BrainLayer(h, wd)
        layerImg.input(img, 0, 0)

        bzLay=brain.BrainLayer(h,wd)

        found=False
        for n in range(int(wd/2), int(wd/2)-1,-1):
            for a in range(85,95,2):
                cmat = brain.getLine(n,a)
                ##print cmat
                bzLay.reset()
                ss=brain.convolution(layerImg, 0, 0, 16, 16, bzLay, 0, 0, cmat)
                for s in ss:
                    s.conduct()
                r = brain.getAxonMat(bzLay, 0, 0, wd, wd)
                if (r.sum() > 0):
                    ##print r
                    found=True
                    break
            else:
                continue
            break

        if(not found):
            return img,found

        bzMat = brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzLay, 0, 0, 16, 16, bzMat, 0, 0, cmat)
        for s in ss:
            s.conduct()

        bzTmp=brain.BrainLayer(h, wd)
        ss = brain.deconvolution(bzMat, 0, 0, 16, 16, bzTmp, 0, 0, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        for s in ss:
            s.conduct()

        c = bzTmp.getAxonMat()
        img = img - c
        layerImg.input(img, 0, 0)
        img = layerImg.getAxonMat()

        return img, found


    @staticmethod
    def Bezier(x, t):  # 3次 Bezier
        xt = 1.0 / 6 * (
            (-x[0] + 3 * x[1] - 3 * x[2] + x[3]) * t * t * t + \
            (3 * x[0] - 6 * x[1] + 3 * x[2]) * t * t + \
            (-3 * x[0] + 3 * x[2]) * t + \
            (x[0] + 4 * x[1] + x[2])
        )
        return xt

    @staticmethod
    def getBezier(points):
        wd = max(max(points))
        hz = np.array([[0] * wd*2] * wd*2)
        segs = len(points) - 3
        for m in range(segs):
            n = wd
            x, y = zip(*points[m:m + 4])
            # hz[x[0]][y[0]]=1
            for tt in range(n):
                t = 1.0 * tt / n
                xt = brain.Bezier(x, t)
                yt = brain.Bezier(y, t)
                # #print t,xt,yt
                #hz[int(round(yt))][int(round(xt))] = 1
                hz[int((yt))][int((xt))] = 1

        hz=brain.center(hz)
        return hz



    def remember(self,hz,label):
        know=brain.Knowledge(hz,label)
        self.knowledges.append(know)

    def recognize(self,img):
        rows = len(img)
        cols = len(img[0])
        layerImg = brain.BrainLayer(rows, cols)
        layerImg.input(img)

        know="Unknow"
        for n in self.knowledges: #100% match 完全匹配
            m=n.match(layerImg)
            ##print m
            if m>=100:
                know = n.label
                break
        if(know=="Unknow"):#模糊匹配
            #Fuzzy match
            f=brain.getFeatures(img)
            for kn in self.knowledges:
                if kn.feature==f:
                    know = kn.label



        return know
