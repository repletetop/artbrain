#include<string.h>
#include<algorithm>
#include "opticnerve.h"
#include "neuron.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

opticnerve::opticnerve()
{
	this->neurons = new neuron[28*28*2+100000];
	this->neuronsdata = new int[28 * 28*2 + 100000];

	this->neuthreshold = new int[28 * 28*2 + 100000];
	for(int i=0;i<28*28*2;i++)neuthreshold[i]=1;

	this->neuronscnt = 28*28*2;
	list<int> *layer0=new list<int>();
	this->palliumlayers.push_back(layer0);

}

KNOWLEDGES::iterator opticnerve::calculate()
{
    vector<int> layer1;
    layer1.assign( (*palliumlayers.begin())->begin(),(*palliumlayers.begin())->end());
    //#pragma omp parallel for //gen man more slower
    for(int i=0;i<layer1.size();i++){
    	int idx=layer1[i];
        neuron *nu = &(this->neurons[idx]);
        //#pragma omp for reduction(+:this->neuronsdata[idx])
	    for(int ii=0;ii<nu->fromsynapses.size();ii++){
	    	//#pragma omp critical
        	this->neuronsdata[idx] += this->neuronsdata[(nu->fromsynapses[ii])->neufrom];
        }

        //printf("%d ",i);
    }
/*
	//calc pallium
	for (list<list<int>*>::iterator it=this->palliumlayers.begin();it!=palliumlayers.end();it++){
		for(list<int>::iterator itp=(*it)->begin();itp!=(*it)->end();itp++){
			int idx=*itp;
			vector<synapse*> ::iterator iter;
			neuron *nu = &(this->neurons[idx]);

			for (iter =nu->fromsynapses.begin(); iter != nu->fromsynapses.end(); iter++) {
				this->neuronsdata[idx] = this->neuronsdata[idx] + this->neuronsdata[(*iter)->neufrom];
			}
			//printf("%d:%d \n",idx,this->neuronsdata[idx]);
		}
	}
*/
//	for (int i = 0; i < this->palliumcnt; ++i) {
//		int idx = this->palliumidx[i];
//		//(this->pallium+idx)->calcValue();
//		vector<synapse*> ::iterator iter;
//		neuron *nu = &(this->neurons[idx]);
//
//		for (iter =nu->fromsynapses.begin(); iter != nu->fromsynapses.end(); iter++) {
//			/*
//			if ((*iter)->polarity != 0) {//polarity !=1
//				(*iter)->value = this->neuronsdata[(*iter)->neufrom] * (*iter)->polarity;
//			}
//			else{
//				(*iter)->value = int(!this->neuronsdata[(*iter)->neufrom]);
//			}
//			this->neuronsdata[idx] = this->neuronsdata[idx] + (*iter)->value;
//			*/
//			//connect =1
//			this->neuronsdata[idx] = this->neuronsdata[idx] + this->neuronsdata[(*iter)->neufrom];
//		}
//	}

	//calc knowledgs,get max neuron and get max knowledge
	KNOWLEDGES::iterator itknow,kmaxit;
	int kmax = 0;
	itknow = this->knowledgs.begin();
	kmaxit = itknow;
	while (itknow != this->knowledgs.end()) {
		vector<int>::iterator itneu = this->neurons[ itknow->second].inneurons.begin();
		int vmax = neuronsdata[*itneu];
		int vidx = *itneu;
		itneu++;
		for (;itneu != this->neurons[itknow->second].inneurons.end(); itneu++) {
			if (vmax < this->neuronsdata[*itneu]) {
				vmax = this->neuronsdata[*itneu];
				vidx = *itneu;				
			}
		}
		this->neurons[itknow->second].actor = vidx;
		if (kmax < vmax) {
			kmax = vmax;
			kmaxit = itknow;
		}
		itknow++;
	}
	return kmaxit;
}

void opticnerve::input(unsigned char * img)
{
	//clear pallium
	memset(this->neuronsdata, 0, this->neuronscnt*sizeof(int));
	//input feel
	for (int i = 0; i < 28 * 28; i++) {
		this->neuronsdata[i] = *(img + i)>0?1:0;
		this->neuronsdata[i+28*28] = int(!*(img + i));
	}
}
KNOWLEDGES::iterator opticnerve::look(unsigned char*img) {
	this->input(img);
	return this->calculate();

}

void opticnerve::remember(unsigned char * img, string label)
{
	this->input(img);
	vector<int> actived;
	int act=0,deact=0;
	if(this->knowledgs.size()>0){
		KNOWLEDGES::iterator know=this->calculate();
		string lb = know->first;
		if (lb == label)
			return;
		//split tree imed or reform on sleep ,select reform when free or new pallium>100

		//神经元bu neng分裂,first proces one layer
		int nuact =neurons[know->second].actor;
		if(neuthreshold[nuact]==neuronsdata[nuact]){
			printf("same img has another label,lost it,write to log\n");
			return;
		}
		getactived(nuact,&actived);
	}

	//remember;
	int nu = -1;
	KNOWLEDGES::iterator iter = this->knowledgs.find(label);
	if (iter != this->knowledgs.end()) {
		nu =  iter->second;
	}
	else {
		nu = this->neuronscnt++;
		this->knowledgs.insert(KNOWLEDGE(label, nu));
	}
	int n = this->neuronscnt++;
	neurons[nu].inneurons.push_back(n);
	for (int i = 0; i < 28 * 28*2; i++) {
		if(this->neuronsdata[i]>0)
			connect(i,n,1);
	}
	for(int i=0;i<actived.size();i++)
	{
		connect(actived[i],n,1);
		//neuthreshold[n]+=neuthreshold[actived[i]];
	}
	neuthreshold[n]=28*28;
	list<list<int>*>::iterator layer = --(this->palliumlayers.end());
	(*layer)->push_back(n);
	neurons[n].layer=layer;

}
//neucommon parents nuid move down for calc after neucommon
void opticnerve::layerdown(list<list<int>*>::iterator currentlayer,int nuid){
    currentlayer++;
    if(currentlayer==palliumlayers.end()){
        list<int>* newlay=new list<int>();
        newlay->push_back(nuid);
        this->palliumlayers.push_back(newlay);
        currentlayer--;
        neurons[nuid].layer=currentlayer;
        return;
    }else{
        (*currentlayer)->push_back(nuid);
        neurons[nuid].layer=currentlayer;
        for(vector<synapse*>::iterator it=neurons[nuid].tosynapses.begin();
            it!=neurons[nuid].tosynapses.end();it++){
        	(*currentlayer)->remove((*it)->neuto);
            layerdown(currentlayer,(*it)->neuto);
        }
    }
}
void opticnerve::setzero(int nuid){
	neuronsdata[nuid]=0;
    for(vector<synapse*>::iterator it=neurons[nuid].fromsynapses.begin();
	it!=neurons[nuid].fromsynapses.end();it++){
    	setzero((*it)->neufrom);
    }
}
//actived only containe root
void opticnerve::getactived(int nuid,vector<int>* actived) {
    vector<int> actalone,actreserve;
    for (vector<synapse*>::iterator it=neurons[nuid].fromsynapses.begin();
		it!=neurons[nuid].fromsynapses.end();it++){
    	//printf("%d,%d\n",(*it)->neufrom,(*it)->parentneuidx);
        if(neuronsdata[(*it)->neufrom]>=neuthreshold[(*it)->neufrom]){
            if(neurons[(*it)->neufrom].tosynapses.size()==1)
                actalone.push_back((*it)->neufrom);
            else
                actreserve.push_back((*it)->neufrom);

			setzero((*it)->neufrom);
	   }else{
        	getactived((*it)->neufrom,actived);
        }
    }
    if(actalone.size()>1){//create new one for not reserve,and disconnect old synapse
        int newact=neuronscnt++;
        for(int i=0;i<actalone.size();i++){
            disconnect(neurons[actalone[i]].tosynapses[0]);
            connect(actalone[i],newact,1);
        }
        (*neurons[nuid].layer)->push_back(newact);//current layer calc
        connect(newact,nuid,1);
        (*neurons[nuid].layer)->remove(nuid);//nuid move down
        layerdown(neurons[nuid].layer,nuid);//nuid laydown
        actreserve.push_back(newact);//
    }else if(actalone.size()==1){//only one ,append to reserve
        actreserve.push_back(actalone[0]);
    }
    if(actreserve.size()>1) {
        //create new
        int newact = neuronscnt++;
        for (vector<int>::iterator it = actreserve.begin(); it != actreserve.end(); it++) {
            connect(*it, newact, 1);
            neuthreshold[newact] += neuthreshold[*it];
        }
        actived->push_back(newact);
        layerdown(neurons[nuid].layer,newact);
    }else if(actreserve.size()==1){
        actived->push_back(actreserve[0]);
    }
}

void opticnerve::connect(int neufrom,int neuto, int polarity = 1) {
    synapse *s = new synapse(neufrom,neuto, polarity);
    neurons[neufrom].tosynapses.push_back(s);
    neurons[neuto].fromsynapses.push_back(s);
}
void opticnerve::disconnect(synapse *s) {
	for(vector<synapse*>::iterator it=neurons[s->neufrom].tosynapses.begin();
			it!=neurons[s->neufrom].tosynapses.end();it++)
		if(*it==s){
		neurons[s->neufrom].tosynapses.erase(it);
		break;
	}
	for(vector<synapse*>::iterator it=neurons[s->neuto].fromsynapses.begin();
			it!=neurons[s->neuto].fromsynapses.end();it++)
		if(*it==s){
		neurons[s->neuto].fromsynapses.erase(it);
		break;
	}
    //neurons[s->neufrom].tosynapses.remove(s);
    //neurons[s->neuto].fromsynapses.remove(s);
    delete s;
}
void opticnerve::disconnectfrom(int neuid,int neufrom, int polarity) {
    vector<synapse*> ::iterator iter;
    for (iter = neurons[neuid].fromsynapses.begin(); iter != neurons[neuid].fromsynapses.end(); iter++)
    {
        if ((*iter)->neufrom == neufrom && (*iter)->polarity == polarity) {
            disconnect((*iter));
        }
        else {
            printf("Not found fromsynapses");
        }
    }
}
