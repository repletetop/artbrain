#include <iostream>
#include <iomanip>
using namespace std;

#include "topone.h"

unsigned int topone(unsigned int *nm,unsigned int cnt){
#pragma HLS INLINE off
	int max=*nm;
	int ret=0;
	for(int i=1;i<cnt;i++){
		if (*(nm+i)>max)
		{
			max=*(nm+i);
			ret=i;
		}
	}
	return ret;
}
