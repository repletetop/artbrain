#include <iostream>
#include <iomanip>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <fstream>

#define USE_MNIST_LOADER
//#define MNIST_DOUBLE

#include "mnist.h"
#include "topone.h"
#include "opticnerve.h"
#include "neuron.hpp"
#include "tb.h"
#include "celeb.h"

using namespace std;
using namespace cv;

void showimg(unsigned char *img) {
	for (int r = 0; r<28; r++)
	{
		for (int c = 0; c<28; c++)
			printf("%d,",img[r*28+c]>0 ? 1 : 0);
		printf("\n");
	}	
}



//test /home/tiansheng/aidata/CelebA
void feel(opticnerve &on,cv::Mat image){
	on.clearfeel();
    int nl = image.rows;
    int nc = image.cols * image.channels();

    //遍历图像的每个像素
    for(int j=0; j<nl ;++j)
    {
        uchar *data = image.ptr<uchar>(j);
        for(int i=0; i<nc; ++i)
        {
            //减少图像中颜色总数的关键算法：if div = 64,
			// then the total number of colors is 4x4x4;整数除法时，是向下取整。
		    int div=128;
            data[i] = data[i]/div*div+div/2;
			int v=data[i]/div;
            //printf("%d,%d ",data[i],v);
			//int v=(data[i] +256-20)/256;
			on.neuronsdata[i +nc * j+v*nl*nc] = 1;
        }
    }
}

void unfeel(opticnerve &on,cv::Mat &image){
    int nl = image.rows;
    int nc = image.cols * image.channels();

    //遍历图像的每个像素
    for(int j=0; j<nl ;++j)
    {
        uchar *data = image.ptr<uchar>(j);
        for(int i=0; i<nc; ++i)
        {
        	if(on.neuronsdata[i +nc * j+0*nl*nc])
            	data[i] = 64;
			else
				data[i] = 200;
        }
    }
}
void unfeela(opticnerve &on,cv::Mat &image){
    int nl = image.rows;
    int nc = image.cols * image.channels();

    //遍历图像的每个像素
    for(int j=0; j<nl ;++j)
    {
        uchar *data = image.ptr<uchar>(j);
        for(int i=0; i<nc; ++i)
        {
        	if(on.neuronsdata[i +nc * j+0*nl*nc])
            	data[i] = 0;
			else
				data[i] = 255;
        }
    }
}
void unfeelb(opticnerve &on,cv::Mat &image){
    int nl = image.rows;
    int nc = image.cols * image.channels();

    //遍历图像的每个像素
    for(int j=0; j<nl ;++j)
    {
        uchar *data = image.ptr<uchar>(j);
        for(int i=0; i<nc; ++i)
        {
        	if(on.neuronsdata[i +nc * j+0*nl*nc])
            	data[i] = 255;
			else
				data[i] = 0;
        }
    }
}

void testreappear(opticnerve &on){
	for(KNOWLEDGES::iterator it=on.knowledgs.begin();it!=on.knowledgs.end();it++){
		neuron know =on.neurons[ it->second];
		cv::Mat img(54,44,CV_8U);
		for(int i=0;i<know.inneurons.size();i++){
			on.clearfeel();
			on.reappear(know.inneurons[i]);
			unfeel(on,img);
			imshow("",img);
			waitKey(1000);
		}
		waitKey(1000);
	}
}
int main_celebA(){
	celebdata data;
	cv::Mat img,result;
	data.loadmat(1,img);
	resize(img,img,Size(img.cols>>3,img.rows>>3),0,0,INTER_LINEAR);//div 8
	//data.saveto("/home/tiansheng/aidata/CelebA/Img/img8d/",1,img);
	//return 1;

	opticnerve on(100000000);// s7000.on2: 6071331  12000.on2:10172301

    double dur;
    clock_t start,ca,end;
    start = clock();

	on.load("celeb8d91000.on2");//6071331
	//on.think();
	//on.save("think8d91000.on2");

	printf("knows:%d\n",on.knowledgs.size());
	vector<string> knows;
	for(int i=on.palliumlayers.size();i>0;i--){
		printf("layer:%d\n",i);
		for(int j= on.palliumlayers[i-1]->size();j>0;j--){
			int n=(*on.palliumlayers[i-1])[j-1];
			//knows.clear();
			//on.statusto(n,knows);
			//if(knows.size()>=10){
				on.clearfeel();
				on.reappear(n);
				//unfeel(on,img);
				unfeela(on,img);
				imshow("",img);
				//printf("\n%dsize(%d): ",n,knows.size());
				for(int j=0;j<knows.size();j++){
					printf("%s ",knows[j].data());
				}
				waitKey(100);
			//}
			//printf("%d:%d ",n,on.neurons[n].tosynapses.size())
		}
		waitKey(2000);
	}

//	on.reduce();
	ca=clock();
	int ok=0,fail=0;
	int beginid=000;//data.imgs.size()=202599
	for(int i=beginid;i<beginid+200;i++){
		//printf("i=%d \n",i);
	    data.loadmat(i,img);
	    resize(img,img,Size(img.cols>>3,img.rows>>3),0,0,INTER_LINEAR);//div 8
	    //imshow("",img);
		//cv::waitKey(500);

		vector<KNOWLEDGES::iterator> allmax;
		feel(on, img);
		on.calculate(allmax);
		if (data.labels[i] == allmax[0]->first) {
			ok++;
			printf("%d ok  : predict %s ok.\n",i,data.labels[i].data());
			//imshow(data.labels[i],img);
			//waitKey(500);
		}
		else{
			printf("%d fail: %s predict %s\n",i, data.labels[i].c_str(),allmax[0]->first.c_str());
			fail++;
		}
		if (allmax.size() > 1) {
			for (int j = 0; j < allmax.size(); j++)
				printf("%s ", allmax[j]->first.c_str());
			printf(" mult predict!\n");
		}
	}
	end = clock();
    dur = (double)(end - ca);
	on.status();
    printf("Use Time test(%d):%10.3fs\n",(ok+fail),(dur/CLOCKS_PER_SEC));
	printf("OK:%d,Fail:%d, score: %0.3f\%\n", ok,fail, ok*100.0 / (ok+fail));


	return 0;

}
void reduce(opticnerve &on){

	vector<string> knows;
	cv::Mat img(28,28,CV_8UC4);
	img.data=(unsigned char*)(on.neuronsdata)+28*28*4;
	for(int i=on.palliumlayers.size();i>0;i--){
		printf("process layer:%d\n",i);
		for(int j=0;j<on.palliumlayers[i-1]->size();){
			int n=(*on.palliumlayers[i-1])[j];
//			int cnt=on.neuthreshold[n];
//			if(cnt>60 && cnt<600){
//    			printf("%d to:%d\n",n,cnt);
//			    on.clearfeel();
//			    on.reappear(n);
//			    imshow("",img);
//			    waitKey(500);
			//}
			//j++;
            //continue;

			knows.clear();
			on.statusto(n,knows);
			//vector<string>::iterator loc=find(knows.begin(),knows.end(),"1");
			//if(loc!=knows.end()){
			if(knows.size()==1 && knows[0]=="1"){
			    //printf("lay:%d id:%d,know:%s\n",i,n,loc->data());
                printf("lay:%d id:%d,know:%s\n",i,n,knows[0].data());
			    on.clearfeel();
			    on.reappear(n);
			    imshow("",img);
			    waitKey(200);

			    j++;
			} else{
			    //printf("lay:%d %d:%d\n",i,n,knows.size());
			    j++;
				//std::vector<int>::iterator it = (*palliumlayers[i-1]).begin()+j;
				//(*palliumlayers[i-1]).erase(it);
				//removeneuron(n);
			}
		}
	}

}

int main_mnist(){

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
	opticnerve on(1000000);//94795
	//on.load("mnist.nev");
	//reduce(on);

	double dur;
    clock_t start,ca,end;
    start = clock();

//goto lbload;


	for (i = 0; i < 10000; i++) {
		img = (unsigned char*)((mnist+i)->data);
		//showimg(img);
		//printf("%d\n",i);
		string lb = to_string((mnist+i)->label);
		on.learn(img, lb);

/*
		vector<KNOWLEDGES::iterator> allmax;
		on.look(img,allmax);
		if(allmax.size()>1) {
			for (int i = 0; i < allmax.size(); i++)
				printf("%s ", allmax[i]->first.c_str());
			printf(" mult predict %s.\n",lb.c_str());
		}
		printf("%s predict %s, %d \n",lb.c_str(), allmax[0]->first.c_str(),
               on.neuronsdata[on.neurons[allmax[0]->second].actor]);

*/
	}
    //return 1;
	//img = (unsigned char*)((mnist+1)->data);
	//showimg(img);
	//KNOWLEDGES::iterator k =  on.look(img);
	//printf("%s \n", k->first.c_str());
	//printf("lays:%d, neurons:%d, synapes:%d\n",on.palliumlayers.size(),on.neuronscnt,on.countsynapse());
	//on.reform();
	//printf("after reform lays:%d, neurons:%d, synapes:%d\n",on.palliumlayers.size(),on.neuronscnt,on.countsynapse());

//	return 1;
	on.status();
	//on.save("mnist.nev");
lbload:

    ca=clock();

    int ok = 0,ttl=0;
	for (i = 10000; i < 10100; i++) {
		ttl++;
		img = (unsigned char*)((mnist + i)->data);
		string lb = to_string((mnist + i)->label);
		vector<KNOWLEDGES::iterator> allmax;
		on.input(img);
		int nuidx=on.predict();
		printf("get nuidx:%d ",nuidx);
		map<int,string>::iterator it= on.knowidx.find(nuidx);
		if(it!=on.knowidx.end()) {
			printf(" %s predict %s\n",lb.data(),it->second.data());
			if(lb==it->second)
				ok++;
		}

		continue;
		//on.look(img,allmax);
		/*
		if(allmax.size()>1) {
			for (int i = 0; i < allmax.size(); i++)
				printf("%s ", allmax[i]->first.c_str());
			printf(" mult predict %s.\n",lb.c_str());
		}
		*/

		KNOWLEDGES::iterator k=allmax[0];
		neuron *nu = on.neurons+k->second;
		if (lb == k->first)
			ok++;
		//else
		//  printf("%s predict %s ,%d\n",lb.c_str(), k->first.c_str(),on.neuronsdata[nu->actor]);
	}
	end = clock();
    dur = (double)(end - ca);
    printf("Use Time train:%f, test:%f\n",(double)(ca-start)/CLOCKS_PER_SEC,(dur/CLOCKS_PER_SEC));
	printf("OK:%d, %0.3f\%\n", ok, ok*100.0 / ttl);
	on.status();
	


	free(mnist);
	//getchar();
	return 0;

}
int main() {
	main_mnist();
	//main_celebA();
}