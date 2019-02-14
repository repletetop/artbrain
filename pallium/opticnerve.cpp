#include<string.h>
#include<algorithm>
#include "opticnerve.h"
#include "neuron.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#define FEELEN (28*28*4)
#define NEURONBUFFERLEN    (FEELEN+1000000)

opticnerve::opticnerve()
{
	this->neurons = new neuron[NEURONBUFFERLEN];
	this->neuronsdata = new int[NEURONBUFFERLEN];

	for(int i=0;i<NEURONBUFFERLEN;i++)neurons[i].pVal=neuronsdata+i;

	this->neuthreshold = new int[NEURONBUFFERLEN];
	for(int i=0;i<FEELEN;i++)neuthreshold[i]=1;

	this->neuronscnt = FEELEN;
	list<int> *layer0=new list<int>();
	this->palliumlayers.push_back(layer0);

}

void opticnerve::calculate(vector<KNOWLEDGES::iterator> &allmax)
{
    //*/
    vector<int> layer1;
    for (list<list<int>*>::iterator it=this->palliumlayers.begin();it!=palliumlayers.end();it++) {
        layer1.assign((*it)->begin(), (*it)->end());
        //#pragma omp parallel for //gen man more slower
        for (int i = 0; i < layer1.size(); i++) {
            int idx = layer1[i];
            neuron *nu = &(this->neurons[idx]);
            this->neuronsdata[idx] =0;
            //#pragma omp for reduction(+:this->neuronsdata[idx])
            for (int ii = 0; ii < nu->fromsynapses.size(); ii++) {
                //#pragma omp critical
                this->neuronsdata[idx] += this->neuronsdata[(nu->fromsynapses[ii])->neufrom];
            }
        }
    }


	//calc knowledgs,get max neuron and get max knowledge
	KNOWLEDGES::iterator itknow,kmaxit;
	int kmax = 0;
	itknow = this->knowledgs.begin();
	kmaxit = itknow;
	bool newmax=true;
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

			newmax= true;
			allmax.clear();
			allmax.push_back(itknow);
		}else if(kmax==vmax){
		    newmax=false;
		    allmax.push_back(itknow);
		}
		itknow++;
	}
	/*
	if(allmax.size()>1){
        for(int i=0;i<allmax.size();i++)
            printf("%s ",allmax[i]->first.c_str());
        printf(" mult predict.\n");
	}
	*/
	//return kmaxit;
}

void opticnerve::input(unsigned char * img)
{
    //have same count img but pos not equ ,so must all pos img same
	//clear pallium memset(this->neuronsdata, 0, this->neuronscnt*sizeof(int));
	memset(this->neuronsdata, 0, FEELEN*sizeof(int));
	//input feel
	for (int i = 0; i < 28 * 28; i++) {
	    int v=(*(img + i)+256-20)/256;
	    //printf("%d,%d  ",*(img + i),v);
		this->neuronsdata[i+28*28*v] =1;
	}
	//printf("\n");
}
void opticnerve::look(unsigned char*img,vector<KNOWLEDGES::iterator> &allmax) {
	this->input(img);
	this->calculate(allmax);

}

void opticnerve::remember(unsigned char * img, string label)
{
	this->input(img);
	vector<int> actived;
	int act=0,deact=0;
	if(this->knowledgs.size()>0){
	    vector<KNOWLEDGES::iterator> allmax;
		this->calculate(allmax);
		KNOWLEDGES::iterator know=allmax[0];
		string lb = know->first;
		if (lb == label)
			return;
		//split tree imed or reform on sleep ,select reform when free or new pallium>100

		int nuact =neurons[know->second].actor;
		if(neuthreshold[nuact]==neuronsdata[nuact]){
			printf("same img has another label,lost it,write to log\n");
			return;
		}
		//神经元bu neng分裂,first proces one layer
		//getactived(nuact,&actived);
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

	for (int i = 0; i < FEELEN; i++) {
		if(this->neuronsdata[i]>0)
			connect(i,n,1);
	}
	for(int i=0;i<actived.size();i++)
	{
		connect(actived[i],n,1);
		//neuthreshold[n]+=neuthreshold[actived[i]];
	}
	//neuthreshold[n]=28*28;
    neuthreshold[n]=FEELEN;
	list<list<int>*>::iterator layer = --(this->palliumlayers.end());
	(*layer)->push_back(n);
	neurons[n].layer=layer;
	/*
	vector<int> acts;
	for (int i = 0; i < 28 * 28*2; i++) {
		if(this->neuronsdata[i]>0)
			acts.push_back(i);
	}
	acts+=actived;
	for(int i=0;i<acts.size();i++){

	}*/


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
	   }else if (neuronsdata[(*it)->neufrom]>0){
        	getactived((*it)->neufrom,actived);
        }
    }
    if(actalone.size()>1){//create new one for not reserve,and disconnect old synapse
        int newact=neuronscnt++;
        for(int i=0;i<actalone.size();i++){
            disconnect(neurons[actalone[i]].tosynapses[0]);
            connect(actalone[i],newact,1);
            neuthreshold[newact]+=neuthreshold[actalone[i]];
        }
        (*neurons[nuid].layer)->push_back(newact);//current layer calc
        neurons[newact].layer=neurons[nuid].layer;
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

int opticnerve::countsynapse() {
    int cnt=0;
    for(int i=0;i<neuronscnt;i++)
        cnt+=neurons[i].fromsynapses.size();
    return cnt;
}
/*
void opticnerve::merge(int a,int b){
    neuron *nu = &(this->neurons[a]);
    this->neuronsdata[b] =0;
    //#pragma omp for reduction(+:this->neuronsdata[idx])
    for (int ii = 0; ii < nu->fromsynapses.size(); ii++) {
        //#pragma omp critical
        this->neuronsdata[idx] += this->neuronsdata[(nu->fromsynapses[ii])->neufrom];
    }
}
*/
void opticnerve::reform() {
    //平衡二叉搜索树（Balanced Binary Tree）具有以下性质：
    // 它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。
    vector<int> layer1;
    for (list<list<int>*>::iterator it=this->palliumlayers.begin();it!=palliumlayers.end();it++) {
        layer1.assign((*it)->begin(), (*it)->end());
        //#pragma omp parallel for //gen man more slower
        for (int i = 0; i < layer1.size()-1; i++)
          for (int j = i+1; j < layer1.size(); i++) {
            int a = layer1[i];
            int b = layer1[j];
            if(neurons[a].fromsynapses==neurons[b].fromsynapses){
            //mergefrom(idx1,idx2);

            }
            if(neurons[a].tosynapses==neurons[b].tosynapses){
                //mergeto
            }

        }
    }
}
