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
    TCNT = 10000#1000 75 4  2000 85 7 5000 84 13.5

    sys.setrecursionlimit(10000000)  # for shelve

    if os.path.exists('on7000formed.sav.dat'):
        print("Loading on7000.sav not run on.__init__  !!!")
        sv=shelve.open('on7000.sav')
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



    on.clearneurons()
    for n in on.pallium[50:100]:
        n.reappear()
        #pltshow(on.output(),'0')
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




    print(trimagesT.shape)
    a = time.time()
    for _ in range(-1):
        for i in range(7000,TCNT):
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

            if((i+1)%1000==0):
                print (i+1)
                on.status()
                on.reform()
                on.status()
                fn = "on%dformed.sav" % (i+1)
                print("Save file %s..." % (fn))
                sv = shelve.open(fn)
                sv['on'] = on
                sv.close()


    b = time.time()
    print("Train cost:", b - a)

    #on.status()
    on.reform()#85 befor
    on.status()
    #fn = "on%dformed.sav" % (TCNT)
    #print("Save file %s..." % (fn))
    #sv = shelve.open(fn)
    #sv['on'] = on
    #sv.close()


    ok=0
    fail=0
    print("Predict...")
    c=time.time()
    for i in range(000+0,0000+100):
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
            nlist = on.look(img)
    d = time.time()
    print(" Right:", ok," total:",ok+fail, d-c)

#>0 58 7.55
#45 7.3
#!=0 54
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