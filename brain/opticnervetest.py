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
    TCNT = 30#1000 75 4  2000 85 7 5000 84 13.5

    sys.setrecursionlimit(1000000000)  # for shelve

    if os.path.exists('sav/on7000.sav.dat'):
        print("Loading  .sav not run on.__init__  !!!")
        on=loadnerve('sav/on5000formed.sav')
        on.status()
    else:#train
        on = opticnerve(28,28)

    #on.genverilog()
    #on.gencpp()
    #exit(1)

    #on.reform()#85 befor
    #fn = "on%dformed.sav" % (i + 1)
    #savenerve(fn)
    #on.status()
    if True:
        #on.clip()
        on.status()
    #exit(1)
    #ttl=len(on.pallium)
    #for i in range(7500,7703):
    #    #print(i)
    #    n=on.pallium[on.palliumidx[i]]
    #    on.clearneurons()
    #    n.reappear()
    #    img=on.output()[:,5:22]
    #    if(img.sum()>20 ):
    #        #print(img.sum,img)
    #        pltshow(img,'0')
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
    if os.path.exists("-testimgcenter.npy"):
        imagesT=np.load('testimgcenter.npy')
    else:
        images = load_test_images()
        sp=images.shape
        imagesT=images.T.reshape(sp[1],28,28)
        #for i in range(len(imagesT)):
        #    imagesT[i]=on.center(imagesT[i])
        #np.save('testimgcenter.npy',imagesT)


    #pltshow(trimagesT[0],'')
    #on.conv14(trimagesT[3])
    #exit(1)


    print(trimagesT.shape)
    a = time.time()
    for _ in range(-1):
        for i in range(0,TCNT):
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
            #if(lb != 0):
            #    continue
            if i==56:
                xxx=555
                #continue
            on.remember(img, lb)

            if((i+1)%1000== -1):
                print (i+1)
                fn = "sav\on%d.sav" % (i+1)
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


    #on.status()
    #fn = "sav/on%d.sav" % (TCNT)
    #savenerve(fn)
    #on.reform()#85 befor
    #on.status()
    fn = "sav/son%dformed.sav" % (TCNT)
    #savenerve(fn)

    #on.gencpp()


    ok=0
    fail=0
    print("Predict...")
    c=time.time()
    for i in range(0,100):
    #for i in [9009,9011,9012]:
        #lb=on.predict(imagesT[i])
        label=labels[i]
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
            print(i,":",label," predict ",lb,nlist[0].actor.value)
            #nlist = on.look(img)
            #pltshow(img,lb)
    d = time.time()
    print(" Right:", ok," total:",ok+fail,"%.2f%%"%(ok*100/(ok+fail)), d-c)

def test3x3():
    on = opticnerve(3,3)
    img1=np.array([[0,1,0],
                   [0,1,0],
                   [0,1,0]])
    img2=np.array([[0,0,0],
                   [1,1,1],
                   [0,0,0]])
    img3=np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    img4=np.array([[0,1,0],
                   [1,0,1],
                   [0,1,0]])
    img5=np.array([[0,1,0],
                   [0,1,0],
                   [0,0,0]])
    img6=np.array([[0,0,0],
                   [0,1,0],
                   [0,1,0]])

    img7=np.array([[0,0,0],
                   [1,1,0],
                   [0,0,0]])
    img8=np.array([[0,0,0],
                   [0,1,1],
                   [0,0,0]])


    on.remember(img1,"|")
    on.remember(img2,'-')
    on.remember(img3,'+')
    #on.think()  # too slow
    lbs = on.predict(img4)
    #lbs = on.predict(img8)
    print(lbs)
    #on.recall('-')
    #on.think()
    #lbs = on.predict(img4)
    #print(lbs)
pass

if __name__ == "__main__":
    test28x28()
   #test3x3()
   #testfeel()