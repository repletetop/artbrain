#-*-coding:utf-8-*-

from opticnerve import *
from hzk import *
from mnistdatabin import *

def test28x28():
    #allhz=hzk()
    #i0=hz2img(allhz[0])


    on = opticnerve(28,28)
    #on.load()
    #on.remember(i0, 0)



    #timages = load_train_images()
    #tlabels = load_train_labels()
    images = load_test_images()
    labels = load_test_labels()

    img=images[:,0].reshape(28,28)

    import time
    a=time.time()
    TCNT=500#17
    batchs=[TCNT]#''',400,500'''
    for n in range(len(batchs)):
        for i in range(batchs[n]):
            if(i==16):
                b=0
                pass
            on.remember(images[:,i].reshape(28,28), labels[i])

        #on.save()
        #print("Before think:")
        on.status()
        on.think() #too slow
        print("After think:")
        on.status()

        #b=time.time()
        #lb=on.predict(img)
        #c=time.time()
        #print(b-a,c-b)
        #print(labels[0],lb)
        total=0
        ok=0
        b = time.time()
        for i in range(000,000+TCNT):
            img = images[:, i].reshape(28, 28)
            #pltshow(img)
            lb=on.predict(img)
            #print(labels[i],lb)
            total = total+1
            if(labels[i]==lb):
                ok=ok+1
            else:
                print(i," label:" ,labels[i]," pre:", lb)
                for act in on.actived:
                    print(act.axon.outneurons[0].label,)
                #pltshow(img)
                #print("====")
                pass

            #import matplotlib.pyplot as plt
            #plt.imshow(img, cmap='gray')
            #plt.show()

        c = time.time()
        print(c - b)

        print("train ",batchs[n]," Right:",ok,total,int(ok*100/total),'%')
#result
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