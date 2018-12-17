#-*-coding:utf-8-*-

from opticnerve import *


if __name__ == "__main__":
    from hzk import *
    allhz=hzk()
    i0=hz2img(allhz[0])
    i1=hz2img(allhz[1])
    i2=hz2img(allhz[2])
    i3=hz2img(allhz[3])
    i4=hz2img(allhz[4])
    i5=hz2img(allhz[5])
    i6=hz2img(allhz[6])
    i7=hz2img(allhz[7])
    i8=hz2img(allhz[8])
    i9=hz2img(allhz[9])
    #print(i1)


    on = opticnerve()
    #on.remember(i0, 0)
    #on.remember(i1, 1)
    #on.remember(i2, 2)
    #on.remember(i3, 3)
    #on.remember(i4, 4)
    #on.remember(i5, 5)
    #on.remember(i6, 6)
    #on.remember(i7, 7)
    #on.remember(i8, 8)
    #on.remember(i9, 9)

    #print(on.knowledges)
    #print(on.neurons[0,0].axon.synapses)
    #print(len(on.pallium[0].dendritic.synapses))
    #lb=on.predict(i2)
    #print(lb)

    from mnistdatabin import *
    #timages = load_train_images()
    #tlabels = load_train_labels()
    images = load_test_images()
    labels = load_test_labels()

    img=images[:,0].reshape(28,28)

    import time
    a=time.time()
    batchs=[5000]#''',400,500'''
    for n in range(len(batchs)):
        for i in range(batchs[n]):
                on.remember(images[:,i].reshape(28,28), labels[i])
        #b=time.time()
        #lb=on.predict(img)
        #c=time.time()
        #print(b-a,c-b)
        #print(labels[0],lb)
        total=0
        ok=0
        for i in range(5000,5100):
            img = images[:, i].reshape(28, 28)
            lbs=on.predict(img)
            #print(labels[i],lb)
            total = total+1
            if(len(lbs)==1 and labels[i]==lbs[0]):
                ok=ok+1
            else:
                print(labels[i]," pre:", lbs)
                pltshow(img)
                #print("====")
                pass

            #import matplotlib.pyplot as plt
            #plt.imshow(img, cmap='gray')
            #plt.show()
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