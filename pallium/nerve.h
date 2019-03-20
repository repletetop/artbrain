//
// Created by tiansheng on 3/8/19.
//

#ifndef TESTBENCH_NERVE_H
#define TESTBENCH_NERVE_H

#include <opencv2/opencv.hpp>

#define _MYDEBUG_
//for focus
class cell{
    int r,c,h;
};
class nerve {
    nerve(int r,int c,int h){
        cells=new cell(r*c*h);
        data=new int(r*c*h);
        for(int i=0;i<r;i++)
            for(int j=0;j<c;j++)
                for(int k=0;k<h;k++){
            int idx=i*c*h+j*h+k;
            cell[idx].r=i,cell[idx].c=j,cell[idx].h=k;
        }
    }
    conv(cv::Mat &img){
        for(int i=0;i<r;i++)
            for(int j=0;j<c;j++)
                for(int k=0;k<h;k++){
            int idx=i*c*h+j*h+k;
            cell[idx].pVa
            cell[idx].r=i,cell[idx].c=j,cell[idx].h=k;
        }
        for()
        for i in range(ROWS):
            for j in range(COLS):
                dist=int(math.sqrt((i-ROWS/2)*(i-ROWS/2)+(j-COLS/2)*(j-COLS/2)))
                r=dist//(ROWS//4)
                #print(i,j,dist,r )
                n=self.layer1[i, j]
                #connect pre
                if(r==0):
                    xxx=1
                #r=3
                for x in range(-r,r+1):
                    for y in range(-r,r+1):
                        if(i+x>=0 and j+y>=0 and i+x<ROWS and j+y<COLS):
                            n.dendritic.connectfrom(self.neurons[i+x,j+y].axon,1)
                n.calcValue()
                n.value=n.value//len(n.dendritic.synapses)
                if(max<n.value):
                    max=n.value
                #cimg[i,j]=1 if(n.value>=60) else 0
                cimg[i,j]=n.value
        #print(max)
        #cimg=(cimg*255/max).astype(np.uint8)
        return cimg
    }
    feel();
    cell* cells;
    int* data;
    int r,c,h;
};


#endif //TESTBENCH_NERVE_H
