#ifndef _NEURON_HPP_
#define _NEURON_HPP_
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
#include <set>
#include <iostream>
using namespace std;

typedef struct synapse {
    synapse(int from,int to){
        neufrom=from;
        neuto=to;
    }
	int neufrom;
	int neuto;
	int threshold;
} SYNAPSE;
/*“仿函数"。指定排序准则*/
class synapseSortCriterion {
    public:
        bool operator() (const SYNAPSE &a, const SYNAPSE &b) const {
            if(a.neuto < b.neuto)
                return true;
            else if(a.neuto == b.neuto) {
                if(a.neufrom < b.neufrom)
                    return true;
                else
                    return false;
            } else
                return false;
        }
};

typedef vector<int> PALLIUMLAY;
struct neuron {
	vector<int> outneurons;//point to knowledgs
	vector<int> inneurons;//point to pallium index;
	int actor = 0;
	set<int> tosynapses;//Dendritic synapses after
    vector<int> fromsynapses;//when connect add element ={0,0};//pre synapses
	//vector<int> frompolarity;//all data is =1;
    int layer;
    int *pVal = NULL;

};

#endif