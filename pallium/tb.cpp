#include <iostream>
#include <iomanip>
#include <time.h>

#define USE_MNIST_LOADER
//#define MNIST_DOUBLE
#include "mnist.h"
#include "topone.h"
#include "opticnerve.h"
#include "neuron.hpp"
#include "tb.h"
using namespace std;

void showimg(unsigned char *img) {
	for (int r = 0; r<28; r++)
	{
		for (int c = 0; c<28; c++)
			printf("%d,",img[r*28+c]>0 ? 1 : 0);
		printf("\n");
	}	
}
int main(){

    mnist_data *mnist;
	unsigned char* img;
    unsigned int cnt;
    int ret;
    int i, j;
	
    if (ret = mnist_load("../../Mnist_data/train-images.idx3-ubyte", "../../Mnist_data/train-labels.idx1-ubyte", &mnist, &cnt))
    {
        printf("An error occured: %d\n", ret);
		free(mnist);
		getchar();
		return 1;
	}
    printf("image count: %d, %d\n", cnt, sizeof(mnist->data));
	opticnerve on;

    double dur;
    clock_t start,ca,end;
    start = clock();

	for (i = 0; i < 1000; i++) {
		img = (unsigned char*)((mnist+i)->data);
		//showimg(img);
		//printf("%d\n",i);
		string lb = to_string((mnist+i)->label);
		on.remember(img, lb);
/*
		KNOWLEDGES::iterator k = on.look(img);
		neuron *nu = on.neurons+k->second;
		printf("__%s predict %s ,%d:%d\n",lb.c_str(), k->first.c_str(),nu->actor,on.neuronsdata[nu->actor]);
*/
	}
	//return 1;

	//img = (unsigned char*)((mnist+1)->data);
	//showimg(img);
	//KNOWLEDGES::iterator k =  on.look(img);
	//printf("%s \n", k->first.c_str());
    ca=clock();

    int ok = 0,ttl=0;
	for (i = 59000; i < 60000; i++) {
		ttl++;
		img = (unsigned char*)((mnist + i)->data);
		string lb = to_string((mnist + i)->label);
		KNOWLEDGES::iterator k = on.look(img);
		neuron *nu = on.neurons+k->second;
		if (lb == k->first)
			ok++;
		//else
		//	printf("%s predict %s ,%d\n",lb.c_str(), k->first.c_str(),on.neuronsdata[nu->actor]);
	}
	end = clock();
    dur = (double)(end - ca);
    printf("Use Time train:%f, test:%f\n",(double)(ca-start)/CLOCKS_PER_SEC,(dur/CLOCKS_PER_SEC));

	printf("OK:%d, %0.3f\n", ok, ok*1.0 / ttl);
	printf("lays:%d,neurons:%d\n",on.palliumlayers.size(),on.neuronscnt);
	


	free(mnist);
	//getchar();
	return 0;

}
