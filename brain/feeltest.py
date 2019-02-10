#-*-coding:utf-8-*-

from opticnerve import *
from hzk import *
from mnistdatabin import *
from goto import with_goto
import shelve
import os
import time

def testfeel():
    on = opticnerve(28,28)

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

    pltshow(imagesT[0],'')
    pltshow(on.rotate(imagesT[0],15),',')
    pltshow(on.rotate(imagesT[0],-15),',')
    exit(1)
    for j in range(3):
        for i in range(100):
            on.feel(imagesT[i])
            img = on.outputlayer(on.layers[2])
            #pltshow(img,labels[i])
            #img=imagesT[i]
            on.remember(img, labels[i])
            exit(1)
    ok=0
    on.reform()
    a=time.time()
    for i in range(9000,9100):
        on.feel(imagesT[i])
        img = on.outputlayer(on.layers[2])
        #img = imagesT[i]
        #pltshow(img, labels[i])
        nmax=on.look(img)
        lbs=[n.label for n in nmax.axon.outneurons]
        if(labels[i] in lbs):
            ok=ok+1
    b=time.time()
    print (ok,b-a)

    exit(1)
    for i in range(3):
        for layer in on.layers:
            on.feel(imagesT[i])
            img=on.outputlayer(layer)
            pltshow(img,labels[i])


if __name__ == "__main__":
   testfeel()