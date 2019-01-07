#-*-coding:utf-8-*-

from opticnerve import *
from hzk import *
from mnistdata import *
from goto import with_goto
import shelve
import os
import time


@with_goto
def test28x28():


    if os.path.exists('on1001.sav.dat'):
        print("Loading on100.sav not run on.__init__  !!!")
        sv=shelve.open('on100.sav')
        on=sv['on']
        sv.close()
        on.status()
    else:#train
        on = opticnerve(28,28)

    #timages = load_train_images()
    #tlabels = load_train_labels()
    labels = load_test_labels()
    if os.path.exists("testimgcenter256.npy"):
        imagesT=np.load('testimgcenter256.npy')
    else:
        images = load_test_images()
        sp=images.shape
        imagesT=images.T.reshape(sp[1],28,28)
        for i in range(len(imagesT)):
            imagesT[i]=on.center(imagesT[i])
        np.save('testimgcenter256.npy',imagesT)





    TCNT = 10
    a = time.time()
    for i in range(TCNT):
        img=imagesT[i]
        on.diff(img)

        #on.feel(img)
        #img=on.outputlayer(on.layers[0])#200: 0,56 1,51 2,55 3,13
        #pltshow(img,labels[i])         #400: 0,39 1,53 2,50 3,12
        on.remember(img, labels[i])     #5000 0,69 1,42
        #img=on.outputlayer(on.layers[3])
        #on.remember(img, labels[i])

    b = time.time()
    print("Train cost:", b - a)
    on.status()
    #on.reform()
    #on.status()

    #sys.setrecursionlimit(1000000)  # for shelve
    #fn = "on%d.sav" % (TCNT)
    #print("Save file %s..." % (fn))
    #sv = shelve.open(fn)
    #sv['on'] = on
    #sv.close()




    ok=0
    print("Predict...")
    c=time.time()
    for i in range(9000+0,9000+100):
    #for i in [9009,9011,9012]:
        #lb=on.predict(imagesT[i])
        img = imagesT[i]
        #on.feel(img)
        #img = on.outputlayer(on.layers[2])
        nmax = on.look(img)
        lb=nmax.axon.outneurons[0].label
        #print(lb)
        if(lb==labels[i]):
            ok=ok+1
        else:
            print(i,":",labels[i]," predict ",lb)
            #lb = on.predict(imagesT[i])
    d = time.time()
    print(" Right:", ok, '%',d-c)

    print (ok)
    exit(1)





    for n in range(len(batchs)):
        on.train(imagesT[0:TCNT],labels[0:TCNT])

        on.status()
        #on.reform()  # too slow
        #print("After reform:",end=" ")
        #on.status()

        fn="on%d.sav"%(TCNT)
        print("Save file %s..."%(fn))
        sv=shelve.open(fn)
        sv['on']=on
        sv.close()

        total=0
        ok=0
        b = time.time()
        for i in range(9901,10000):
            img = imagesT[i]
            #pltshow(img)
            if i==9:
                a=5
            lb=on.predict(img)
            #print(labels[i],lb)
            total = total+1
            if(labels[i]==lb):
                ok=ok+1
            else:
                #print("")
                print(i," label:" ,labels[i]," pre:", lb)
                #for act in on.actived:
                #    print(act.axon.outneurons[0].label,end=" ")
                #pltshow(img)
                #print("====")
                pass

            #import matplotlib.pyplot as plt
            #plt.imshow(img, cmap='gray')
            #plt.show()

        c = time.time()
        print(c - b)

        print(" Right:",ok,total,int(ok*100/total),'%')
        if ok!=total:
            print("####need train again###")
            #goto .train


#result
#train 500
#42.6 after think 13.5
#train test
#500,  1000  84.4%
#500   100   74  pass already fast 0.7
#500   100   74  pass already fast 0.8
#500   100   76  pass already fast 0.9
#500   100   80  pass already fast 0.95
#500   100   81  no pass slow 1.0
#==========================
#100   1000-1100  62  threshold 0.7
#200   1000-1100  71  threshold 0.7
#300   1000-1100  74  threshold 0.7
#400   1000-1100  76  threshold 0.7
#500   1000-1100  76  threshold 0.7
#1000   1000-1100  81  threshold 0.7
#==========================
#      5000-5100 threshold 0.7
#100   58  threshold 0.7
#200   67  threshold 0.7
#300   73  threshold 0.7
#400   79  threshold 0.7
#500   81  threshold 0.7
#=============================
#Train 5000-5100 threshold 0.8
#100  58
#200  67
#300  73
#400  79
#500  81
#=============================
#Train 5000-5100 threshold 1.0
#100  0
#200  0
#300  0
#==============================
#Train 5000-5100 threshold 0.99
#100  0
#200  0
#300  0
#========================================
#Train 5000-5100 threshold 0.50,0.10,0,01
#100  58
#200  67
#300  73
#===================================threshold not need
#Train 5000-5100 threshold 0.50,0.10,0.00001
#100  58
#200  67
#300  73
#5000 95%

#after center:
#200 73%
#1000 86%
#2000 82%


#speed 100,6.47
#calc by neuron 100,0.93
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
def testfeel():
    on = opticnerve(28,28)
    images = load_test_images()
    labels = load_test_labels()

    img=images[:,0].reshape(28,28)

    import time
    a=time.time()
    batchs=[500]#''',400,500'''

    xx=0
    for n in range(len(batchs)):
        for i in range(batchs[n]):
            if(i==9):
                b=0
                pass
            on.feel(images[:,i].reshape(28,28))
            xx=xx+images[:,i].sum()+1
            print(i,images[:,i].sum(),xx)
            on.status()

        for n in on.pallium:
            on.clearneurons()
            n.recall()
            img=on.output()
            if(img.sum()>20):
                pltshow(img)

        on.status()

if __name__ == "__main__":
    test28x28()
   # test3x3()
   #testfeel()