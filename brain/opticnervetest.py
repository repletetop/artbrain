#-*-coding:utf-8-*-

from opticnerve import *
from hzk import *
from mnistdatabin import *
from goto import with_goto
import shelve
import os
import time


@with_goto
def test28x28():
    TCNT = 3000#1000 75 4  2000 85 7 5000 84 13.5

    sys.setrecursionlimit(1000000000)  # for shelve

    if os.path.exists('on7000.sav.dat'):
        print("Loading  .sav not run on.__init__  !!!")
        sv=shelve.open('on3000formedA5.sav')#3000 88%
        on=sv['on']
        sv.close()
        on.status()
    else:#train
        on = opticnerve(28,28)

    #test
    ROWS=28
    COLS=28
    for i in range(ROWS):
        for j in range(COLS):
            on.neurons[i,j].pos=(i,j)
            on.neurons[i, j].id=-1
    #on.verilog()
    for i in range(len(on.pallium)):
        on.pallium[i].id=i
    tmp=""
    fd = open('./pallum.v', 'w')#28*28=784
    strmodule='''
module pallum(
    input[783:0] img,
    output reg[15:0] num0,
    output reg[15:0] num1,
    output reg[15:0] num2,
    output reg[15:0] num3,
    output reg[15:0] num4,
    output reg[15:0] num5,
    output reg[15:0] num6,
    output reg[15:0] num7,
    output reg[15:0] num8,
    output reg[15:0] num9
    );
    '''

    fneuron = open('./neuron.v','w')
    nset=[]

    print(strmodule,file=fd)
    for i in on.palliumidx[0:]:
        n=on.pallium[i]
        nlen=len(n.dendritic.synapses)
        if nlen not in nset:
            nset.append(nlen)
            modneu="module neuron%d("%(nlen)
            for x in range(nlen):
                modneu +="input i%d,"%(x)
            modneu+="output o);\n assign o=i0"
            for x in range(1,nlen):
                modneu+="+ai%d"%(x)
            modneu+=";\nendmodule\n"
            print(modneu,file=fneuron)


        tmp="wire o%d;\nneuron%d nu%d("%(i,nlen,i)
        for s in n.dendritic.synapses:
            npre=s.axon.connectedNeuron
            if npre.id == -1:
                tmp = tmp+"img[%d*28+%d],"%(npre.pos[0], \
                                            npre.pos[1])
            else:
                tmp = tmp + "o%d," % (npre.id)

        tmp=tmp+"o%d);"%(i)
        #print(tmp)
        print(tmp,file=fd)
        if(n.axon.outneurons!=[]):
            label=n.axon.outneurons[0].label
'''
always@(oid)
begin 
  if (oid>numlabel)
    numlabel=oid
'''
            tmp = '''
always@(o%d)
begin
  if (o%d>num%s)
    num%s=o%d
            '''%(i,i,label,label,i)

    print("endmodule",file=fd)
    fd.close()
    fneuron.close()

    exit(1)

    #on.reform()#85 befor
    #on.status()
    #for i in on.palliumidx[1200:1300]:
    #for i in range(100,200):
    #    n=on.pallium[i]
    #    on.clearneurons()
    #    n.reappear()
    #    pltshow(on.output(),'0')
    #exit(1)

    trlabels = load_train_labels()
    if os.path.exists("trainimgcenter2.npy"):
        trimagesT=np.load('trainimgcenter2.npy')
    else:
        trimages = load_train_images()
        sp=trimages.shape
        trimagesT=trimages.T.reshape(sp[1],28,28)
        for i in range(len(trimagesT)):
            trimagesT[i]=on.center(trimagesT[i])
        np.save('trainimgcenter2.npy',trimagesT)


    labels = load_test_labels()
    if os.path.exists("testimgcenter.npy"):
        imagesT=np.load('testimgcenter.npy')
    else:
        images = load_test_images()
        sp=images.shape
        imagesT=images.T.reshape(sp[1],28,28)
        for i in range(len(imagesT)):
            imagesT[i]=on.center(imagesT[i])
        np.save('testimgcenter.npy',imagesT)


    #pltshow(trimagesT[0],'')
    #on.conv14(trimagesT[3])
    #exit(1)


    print(trimagesT.shape)
    a = time.time()
    for _ in range(-1):
        for i in range(6000,TCNT):
            img=trimagesT[i]
            lb=trlabels[i]
            #img = on.diff(img)
            #print(img)
            #pltshow(img,labels[i])
            #on.feel(img)
            #img=on.outputlayer(on.layers[1])#200: 0,56 1,51 2,55 3,13
            #pltshow(img,labels[i])         #400: 0,39 1,53 2,50 3,12
            #on.remember(img, trlabels[i])     #5000 0,69 1,42
            #img=on.outputlayer(on.layers[3])
            on.remember(img, lb)

            if((i+1)%1000== -1):
                print (i+1)
                fn = "on%d.sav" % (i+1)
                print("Save file %s..." % (fn))
                sv = shelve.open(fn)
                sv['on'] = on
                sv.close()
                #on.status()
                #on.reform()
                #on.status()
                fn = "on%d.sav" % (i+1)
                print("Save file %s..." % (fn))
                sv = shelve.open(fn)
                sv['on'] = on
                sv.close()


    b = time.time()
    print("Train cost:", b - a)

    on.status()
    on.reform()#85 befor
    on.status()
    fn = "on%dformedA5.sav" % (TCNT)
    print("Save file %s..." % (fn))
    sv = shelve.open(fn)
    sv['on'] = on
    sv.close()


    ok=0
    fail=0
    print("Predict...")
    c=time.time()
    for i in range(9900+0,10000):
    #for i in [9009,9011,9012]:
        #lb=on.predict(imagesT[i])
        img = imagesT[i]
        #img = on.diff(img)
        #on.feel(img)
        #img = on.outputlayer(on.layers[2])
        if(i==-10):#5
            xx=10*44
            nlist = on.look(img)
            ovalue = nlist[0].value
            on.reform()
            nlist = on.look(img)
            print( ovalue,nlist[0].value)
            #exit(1)


        nlist = on.look(img)
        lb=nlist[0].label
        #print(lb)
        if(lb==labels[i]):
            ok=ok+1
        else:
            fail+=1
            print(i,":",labels[i]," predict ",lb)
            #nlist = on.look(img)
            #pltshow(img,lb)
    d = time.time()
    print(" Right:", ok," total:",ok+fail,"%.2f%%"%(ok*100/(ok+fail)), d-c)


if __name__ == "__main__":
    test28x28()
   # test3x3()
   #testfeel()