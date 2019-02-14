/*
@author:893802998@qq.com
neurons->axon->synape->dendritic->neurons->axon
1     1 1  m        1     m    m    1   1

neurons connect to synape
from dendritic
'''
*/
#include <list>
#include <vector>
using namespace std;

struct synapse {
	synapse(int vneufrom, int vneuto,int vpolarity = 1) {
		this->neufrom = vneufrom;
		this->polarity = vpolarity;
		this->neuto=vneuto;
	}
	int neufrom;
	int neuto;
	int polarity;
};

struct neuron {
public:
	vector<int> outneurons;//point to knowledgs
	vector<int> inneurons;//point to pallium index;
	int actor = 0;
	vector<synapse*> tosynapses;//Dendritic synapses
    vector<synapse*> fromsynapses;//parent synapses
    list<list<int>*>::iterator layer;
    int *pVal = NULL;

};

