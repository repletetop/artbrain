#include<string.h>
#include<algorithm>
#include "opticnerve.h"
#include "neuron.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <fstream>
#include <set>
#include <assert.h>
#include <map>
#include <vector>
#include <cxcore.h>
#include <highgui.h>

using namespace std;

opticnerve::opticnerve(int maxneucnt)
{
	maxneucnt+=FEELEN;
	this->neurons = new neuron[maxneucnt];
	this->neuronsdata = new unsigned char[maxneucnt];
	//neutimes=new int[maxneucnt];
	//curtimes=0;

	for(int i=0;i<maxneucnt;i++) {
		neurons[i].pVal = neuronsdata + i;
		neurons[i].layer=-1;
	}

	this->neuthreshold = new int[maxneucnt];
	//memset(neuronsdata,FEELEN,sizeof(int));
	for(int i=0;i<FEELEN;i++)neuthreshold[i]=1;

	this->neuronscnt = FEELEN;
	vector<int> *layer0=new vector<int>();
	this->palliumlayers.push_back(layer0);
}

void opticnerve::calculate(vector<int> &allmax)
{
	list<int> maybe;
    for (vector<vector<int>*>::iterator it=this->palliumlayers.begin();it!=palliumlayers.end();it++) {
        //#pragma omp parallel for //gen man more slower
        for (int i = 0; i < (*it)->size(); i++) {
            int idx = (*(*it))[i];
			neuron *nu = &(this->neurons[idx]);
			//#pragma omp for reduction(+:this->neuronsdata[idx])
			int tmp = 0;
			for(int j=0;j<nu->fromsynapses.size();j++){
				//tmp += abs(this->neuronsdata[nu->fromsynapses[j]]-nu->fromthreshold[j]);
				tmp++;
			}
			//if(tmp==0)
			//	this->neuronsdata[idx]=255;
			//else{
				this->neuronsdata[idx]=255-tmp/nu->fromsynapses.size();
			//}

        }
    }



	//calc knowledgs
	KNOWLEDGES::iterator itknow;
	itknow = this->knowledgs.begin();
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
		itknow++;
	}

	//get max and second knowledge
	KNOWLEDGES::iterator kmaxit,ksecondit,itcur,itnext;
	itcur = this->knowledgs.begin();
	itnext=itcur;
	itnext++;
	if(itnext==knowledgs.end()){
		allmax.push_back((*itcur).second);
		return;
	}
	if(neuronsdata[neurons[itcur->second].actor]>neuronsdata[neurons[itnext->second].actor]){
		kmaxit=itcur;ksecondit=itnext;
	}else{
		kmaxit=itnext;ksecondit=itcur;
	}
	itknow = itnext++;
	while (itknow != knowledgs.end()) {
		if (neuronsdata[neurons[itknow->second].actor] > neuronsdata[neurons[kmaxit->second].actor]) {
			ksecondit=kmaxit;
			kmaxit=itknow;
		}
		else if(neuronsdata[neurons[itknow->second].actor] > neuronsdata[neurons[ksecondit->second].actor]) {
			ksecondit=itknow;
		}
		itknow++;
	}
	allmax.push_back((*kmaxit).second);
	allmax.push_back((*ksecondit).second);
}
void opticnerve::clearfeel(){
	//clear pallium memset(this->neuronsdata, 0, this->neuronscnt*sizeof(int));
	//memset(this->neuronsdata, 0, neuronscnt*sizeof(int));
	memset(this->neuronsdata, 0, FEELEN*sizeof(int));
}
void opticnerve::input(unsigned char * img)
{
    //have same count img but pos not equ ,so must all pos img same
	clearfeel();
	//input feel
	for (int i = 0; i < 28 * 28; i++) {
	    int v=(*(img + i)+256-20)/256;
	    //printf("%d,%d  ",*(img + i),v);
		this->neuronsdata[i+28*28*v] =1;
	}
	//printf("\n");
}
void opticnerve::look(unsigned char*img,vector<int> &allmax) {
	this->input(img);
	this->calculate(allmax);
}
//befor remember must give it feel or input
//void opticnerve::remember(string label){
//	//curtimes++;
//	vector<int> actived;
//	int act=0,deact=0;
//	int rootact;
//	if(this->knowledgs.size()>0){
//	    vector<int> allmax;
//		this->calculate(allmax);
//		int know=allmax[0];
//
//		string lb = (knowidx.find(know))->second;
//		if (lb == label)
//			return;
//
//		int nuact =neurons[know].actor;
//		//if(neuthreshold[nuact]==neuronsdata[nuact]){
//		//	printf("same img has another label [%s],lost it,write to log\n",label.data());
//		//	//return;
//		//}
//		//神经元bu neng分裂,first proces one layer
//		//getactived(nuact,&actived);
//		if(actived.size()>0) {
//			//rootact = createbtree(actived);
//			rootact = createtree(actived);
//			neurons[rootact].outneurons.push_back(know);
//		}
//
//
//	}
//
//
///*
//	vector<int> tmp ;
//	for (int i = 0; i < FEELEN; i++) {
//		if(this->neuronsdata[i]>0){
//			tmp.push_back(i);}
//	}
//	int rootnew=0;
//	if(tmp.size()>0)
//	{
//		rootnew=createbtree(tmp);
//		neurons[rootnew].outneurons.push_back(nu);
//	}
//
//	if(actived.size()>0){
//		neurons[rootact].outneurons.push_back(nu);
//
//		int n = getneuron();
//		connect(rootnew, n);
//		connect(rootact, n);
//		neuthreshold[n]+=(neuthreshold[rootact] + neuthreshold[rootnew]);
//		int ilay = neurons[rootact].layer >= neurons[rootnew].layer ? neurons[rootact].layer : neurons[rootnew].layer;
//		ilay++;
//		if(palliumlayers.size() < ilay + 1){
//			vector<int>* lay=new vector<int>();
//			palliumlayers.push_back(lay);
//		}
//		palliumlayers[ilay]->push_back(n);
//		neurons[n].layer=ilay;
//		neurons[nu].inneurons.push_back(n);
//		neurons[n].outneurons.push_back(nu);
//	}else{
//		neurons[nu].inneurons.push_back(rootnew);
//	}
//*/
//////////////
//
//
//	int rootnew = this->getneuron();
//	for(int l=0;l<FEELLAYS;l++){
//		//connected neurons
//		for(int r=0;r<FEELWIDTH;r++){
//			for(int c=0;c<FEELWIDTH;c++){
//				if(this->neuronsdata[l*FEELWIDTH*FEELWIDTH+r*FEELWIDTH+c]>0) {
//
//					//way second lixing
//					vector<pair<int,int>> pts,zone;
//					pts.push_back(pair<int,int>(r,c));
//					int area=getneuron();
//					while(pts.size()>0){
//						pair<int,int> pt=pts.back();
//						pts.pop_back();
//						if (this->neuronsdata[l*FEELWIDTH*FEELWIDTH+pt.first*FEELWIDTH+pt.second]!=0){
//							//way first zhijue
//							connectvalue(l*FEELWIDTH*FEELWIDTH+pt.first*FEELWIDTH+pt.second, rootnew,this->neuronsdata[l*FEELWIDTH*FEELWIDTH+pt.first*FEELWIDTH+pt.second]);
//                            zone.push_back(pt);
//                            this->neuronsdata[l*FEELWIDTH*FEELWIDTH+pt.first*FEELWIDTH+pt.second]=0;
//						}
//						if(this->neuronsdata[l*FEELWIDTH*FEELWIDTH+pt.first*FEELWIDTH+pt.second+1]>0)pts.push_back(pair<int,int>(pt.first,pt.second+1));
//						if(this->neuronsdata[l*FEELWIDTH*FEELWIDTH+(pt.first+1)*FEELWIDTH+pt.second-1]>0)pts.push_back(pair<int,int>(pt.first+1,pt.second-1));
//						if(this->neuronsdata[l*FEELWIDTH*FEELWIDTH+(pt.first+1)*FEELWIDTH+pt.second]>0)pts.push_back(pair<int,int>(pt.first+1,pt.second));
//						if(this->neuronsdata[l*FEELWIDTH*FEELWIDTH+(pt.first+1)*FEELWIDTH+pt.second+1]>0)pts.push_back(pair<int,int>(pt.first+1,pt.second+1));
//					}
//					if(zone.size()>300){
//						//get center
//						int minr=9999,maxr=0,minc=9999,maxc=0;
//						//int area=getneuron();
//						for(int idx=0;idx<zone.size();idx++){
//							if(minr>zone[idx].first)minr=zone[idx].first;
//							if(maxr<zone[idx].first)maxr=zone[idx].first;
//							if(minc>zone[idx].second)minc=zone[idx].second;
//							if(maxc<zone[idx].second)maxc=zone[idx].second;
//						}
//						printf("lay:%d, size %d,rect:%d,%d  %d,%d\n",l,zone.size(),minr,minc,maxr,maxc);
//						//connectvalue(area,rootnew,255);
//						//rememberarea();//remember feature;
//						cv::Mat areamat(maxr-minr+1,maxc-minc+1,CV_8U);
//						for(int idx=0;idx<zone.size();idx++){
//							areamat.at<uchar>(zone[idx].first-minr,zone[idx].second-minc)=255;
//						}
//						//cv::imshow("",areamat);
//						//cv::waitKey(0);
//						neuthreshold[rootnew]+=neuthreshold[area];
//						//way second, remember area shape and position
//					}else{
//						//printf("zone.size=%d not connect!\n",zone.size());
//					}
//				}
//
//			}
//		}
//	}
//
//	for (int i = 0; i < FEELEN; i++) {
//		if(this->neuronsdata[i]>0) {
//			connect(i, rootnew);
//			neuthreshold[rootnew]+=neuthreshold[i];
//		}
//	}
//
//	if(neurons[rootnew].fromsynapses.size()==0){
//		unused.push_back(rootnew);
//		return;
//	}
//
//	//remember;
//	int nu = -1;
//	KNOWLEDGES::iterator iter = this->knowledgs.find(label);
//	if (iter != this->knowledgs.end()) {
//		nu =  iter->second;
//	}
//	else {
//		nu = getneuron();
//		this->knowledgs.insert(KNOWLEDGE(label, nu));
//		this->knowidx.insert(pair<int,string>(nu,label));
//	}
////////////
//
//	neurons[rootnew].outneurons.push_back(nu);
//	neurons[rootnew].layer=0;
//	palliumlayers[0]->push_back(rootnew);
//	if(actived.size()>0){
//		neurons[rootact].outneurons.push_back(nu);
//
//		int n = getneuron();
//		connect(rootnew, n);
//		connect(rootact, n);
//		neuthreshold[n]+=(neuthreshold[rootact] + neuthreshold[rootnew]);
//		int ilay = neurons[rootact].layer >= neurons[rootnew].layer ? neurons[rootact].layer : neurons[rootnew].layer;
//		ilay++;
//		if(palliumlayers.size() < ilay + 1){
//			vector<int>* lay=new vector<int>();
//			palliumlayers.push_back(lay);
//		}
//		palliumlayers[ilay]->push_back(n);
//		neurons[n].layer=ilay;
//		neurons[nu].inneurons.push_back(n);
//		neurons[n].outneurons.push_back(nu);
//	}else{
//		neurons[nu].inneurons.push_back(rootnew);
//	}
//}
void opticnerve::remember(string label){
	int rootnew = this->getneuron();
	for(int x=255;x>1;x--){
		for(int r=1; r<FEELWIDTH-1 ;++r)
		for(int c=1; c<FEELWIDTH-1 ;++c)
		{
			uchar *data0 = &neuronsdata[(r-1)*FEELWIDTH];
			uchar *data1 = &neuronsdata[(r)*FEELWIDTH];
			uchar *data2 = &neuronsdata[(r+1)*FEELWIDTH];
			int v=data0[c-1]+data1[c-1]+data2[c-1]+
			   data0[c]+data1[c]+data2[c]+
			   data0[c+1]+data1[c+1]+data2[c+1];
			if(v==9*x){
				connect(r*FEELWIDTH+c, rootnew);
				neuthreshold[rootnew]+=neuthreshold[r*FEELWIDTH+c];
			}
		}
	}

	//remember;
	int nu = -1;
	KNOWLEDGES::iterator iter = this->knowledgs.find(label);
	if (iter != this->knowledgs.end()) {
		nu =  iter->second;
	}
	else {
		nu = getneuron();
		this->knowledgs.insert(KNOWLEDGE(label, nu));
		this->knowidx.insert(pair<int,string>(nu,label));
	}

	neurons[rootnew].outneurons.push_back(nu);
	neurons[rootnew].layer=0;
	palliumlayers[0]->push_back(rootnew);
	neurons[nu].inneurons.push_back(rootnew);
}

int opticnerve::createbtree(vector<int> &tmp) {
	int cnt=tmp.size();
	int ilay=0;

	while(cnt>=2){
		for(int i=0;i<cnt-1;i+=2){
			int n = getneuron();
			connect(tmp[i], n);
			connect(tmp[i + 1], n);
			neuthreshold[n]+=(neuthreshold[tmp[i]] + neuthreshold[tmp[i + 1]]);
			ilay = neurons[tmp[i]].layer >= neurons[tmp[i + 1]].layer ? neurons[tmp[i]].layer : neurons[tmp[i + 1]].layer;
			ilay++;

			if(palliumlayers.size() < ilay + 1){
				vector<int>* lay=new vector<int>();
				palliumlayers.push_back(lay);
			}
			palliumlayers[ilay]->push_back(n);
			neurons[n].layer=ilay;
			tmp[i/2]=n;
		}
		if(cnt%2!=0){//calc next lay
			tmp[cnt/2]=tmp[cnt-1];
			cnt = (cnt>>1)+1;
		} else {
			cnt = cnt >> 1;
		}
	}
	return tmp[0];
}

void opticnerve::learn(unsigned char * img, string label)
{
	this->input(img);
	this->remember(label);
}
//neucommon parents nuid move down for calc after neucommon
void opticnerve::layerafter(int currentlayer,int nuid){
    currentlayer++;
    if(currentlayer==palliumlayers.size()){
        vector<int>* newlay=new vector<int>();
        newlay->push_back(nuid);
        this->palliumlayers.push_back(newlay);
        neurons[nuid].layer=this->palliumlayers.size()-1;
        return;
    }else{
    	vector<int>* lay=palliumlayers[currentlayer];
        lay->push_back(nuid);
        neurons[nuid].layer=currentlayer;
        for(set<int>::iterator it=neurons[nuid].tosynapses.begin();
            it!=neurons[nuid].tosynapses.end();it++){
        	auto mv=find(lay->begin(),lay->end(),(*it));
        	if(mv!=lay->end()){
        		lay->erase(mv);
            	layerafter(currentlayer,(*it));
        	}
        }
    }
}
void opticnerve::setzero(int nuid){
	neuronsdata[nuid]=0;
    for(vector<int>::iterator it=neurons[nuid].fromsynapses.begin();
	it!=neurons[nuid].fromsynapses.end();it++){
    	setzero((*it));
    }
}
//actived only containe root
void opticnerve::getactived(int nuid,vector<int>* actived) {
    vector<int> actalone,actreserve;
    for (vector<int>::iterator it=neurons[nuid].fromsynapses.begin();
		it!=neurons[nuid].fromsynapses.end();it++){
    	//printf("%d,%d\n",(*it)->neufrom,(*it)->parentneuidx);
        if(neuronsdata[(*it)]>=neuthreshold[(*it)]){
            if(neurons[(*it)].tosynapses.size()==1)
                actalone.push_back((*it));
            else
                actreserve.push_back((*it));

			setzero((*it));
	   }else if (neuronsdata[(*it)]>0){
        	getactived((*it),actived);
        }
    }
    if(actalone.size()>1){//create new one for not reserve,and disconnect old synapse
        int newact=getneuron();
        for(int i=0;i<actalone.size();i++){
            disconnect(actalone[i],nuid);
            //printf("disconnect from %d to %d \n",actalone[i],nuid);
            connect(actalone[i],newact);
            neuthreshold[newact]+=neuthreshold[actalone[i]];
        }
        palliumlayers[neurons[nuid].layer]->push_back(newact);//current layer calc
        neurons[newact].layer=neurons[nuid].layer;
        connect(newact,nuid);
        vector<int>::iterator itnuid=find(palliumlayers[neurons[nuid].layer]->begin(),
													   palliumlayers[neurons[nuid].layer]->end(),nuid);
        if(itnuid !=palliumlayers[neurons[nuid].layer]->end()){
        	palliumlayers[neurons[nuid].layer]->erase(itnuid);
        } else{
        	printf("%d not found in lay %d\n",nuid,neurons[nuid].layer);
        }
        layerafter(neurons[nuid].layer,nuid);//nuid laydown
        actreserve.push_back(newact);//
    }else if(actalone.size()==1){//only one ,append to reserve
        actreserve.push_back(actalone[0]);
    }
    if(actreserve.size()>1) {
        //create new
        int newact = getneuron();
        for (vector<int>::iterator it = actreserve.begin(); it != actreserve.end(); it++) {
            connect(*it, newact);
            neuthreshold[newact] += neuthreshold[*it];
        }
        actived->push_back(newact);
        layerafter(neurons[nuid].layer,newact);
    }else if(actreserve.size()==1){
        actived->push_back(actreserve[0]);
    }
}

void opticnerve::connect(int neufrom,int neuto) {
	//synapses.insert(SYNAPSE{neufrom,neuto});

    neurons[neufrom].tosynapses.insert(neuto);
    neurons[neuto].fromsynapses.push_back(neufrom);
}
void opticnerve::connectvalue(int neufrom,int neuto,int value) {
	//synapses.insert(SYNAPSE{neufrom,neuto});
    neurons[neufrom].tosynapses.insert(neuto);
    neurons[neuto].fromsynapses.push_back(neufrom);
    neurons[neuto].fromthreshold.push_back(value);
}

void opticnerve::disconnect(int neufrom,int neuto) {
	//SYNAPSE snp = SYNAPSE{neufrom,neuto};
	//set<SYNAPSE>::iterator it=synapses.find(snp);
	//if(it!=synapses.end()){
	//	synapses.erase(it);
	//}

	for(set<int>::iterator it=neurons[neufrom].tosynapses.begin();
			it!=neurons[neufrom].tosynapses.end();it++)
		if(*it==neuto){
		neurons[neufrom].tosynapses.erase(it);
		break;
	}
	for(vector<int>::iterator it=neurons[neuto].fromsynapses.begin();
			it!=neurons[neuto].fromsynapses.end();it++)
		if(*it==neufrom){
		neurons[neuto].fromsynapses.erase(it);
		break;
	}
}

int opticnerve::countsynapse() {
    int cnt=0;
    for(int i=0;i<neuronscnt;i++) {
    	//printf("%d ",neurons[i].fromsynapses.size());
		cnt += neurons[i].fromsynapses.size();
	}
	//printf("\n");
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
    for (int ilay=0;ilay<palliumlayers.size();ilay++) {
    	vector<int> *curlay = palliumlayers[ilay];
    	vector<int>::iterator ite=curlay->end();
        for (vector<int>::iterator ita=curlay->begin();ita!=curlay->end();ita++)
        {
        	int a= *ita;
        	vector<int>::iterator itb=ita;
        	itb++;
        	for(;itb!=curlay->end();itb++) {
				int b = *itb;
				vector<int> ab;
				ab.resize(neurons[a].fromsynapses.size());
				auto abend = set_intersection(begin(neurons[a].fromsynapses), end(neurons[a].fromsynapses),
											  begin(neurons[b].fromsynapses), end(neurons[b].fromsynapses),
											  ab.begin());
				int cnt = abend - ab.begin();
				if (abend - ab.begin() > 2) {
					int n = getneuron();
					connect(n, a);
					connect(n, b);
					printf("%d,%d have %d same.\n", a, b, cnt);
					for (int k=0;k<cnt;k++) {
						int x = ab[k];
						disconnect(x, a);
						disconnect(x, b);
						connect(x, n);
					}
					curlay->erase(itb);
					curlay->erase(ita);
					curlay->push_back(n);
					layerafter(ilay, a);
					layerafter(ilay, b);
					break;
				}
			}
        }
    }
}

void opticnerve::save(const char* filename)
{
	std::ofstream outfs;
	int data;
	outfs.open(filename,ios::binary|ios::out);//,std::ios::binary
	outfs.write((char*)&neuronscnt, sizeof(int));
	//outfs.write((char*)neutimes, sizeof(int)*neuronscnt);
	//outfs.write((char*)&curtimes, sizeof(int));

	int cnt=(palliumlayers.size());
	outfs.write((char*)&cnt, sizeof(int));
	for(vector<vector<int>*>::iterator it=palliumlayers.begin();it!=palliumlayers.end();it++){
		cnt=(*it)->size();
		outfs.write((char*)&cnt, sizeof(int));
		for(vector<int>::iterator layer=(*it)->begin();layer!=(*it)->end();layer++){
			data=(*layer);
			outfs.write((char*)&data, sizeof(data));
			cnt=neurons[data].fromsynapses.size();
			outfs.write((char*)&cnt, sizeof(cnt));
			for(vector<int>::iterator itf=neurons[data].fromsynapses.begin();itf!=neurons[data].fromsynapses.end();itf++){
				cnt=(*itf);
				outfs.write((char*)&cnt, sizeof(cnt));
			}
		}
	}
	cnt=(knowledgs.size());
	outfs.write((char*)&cnt, sizeof(int));
	for(KNOWLEDGES::iterator it=this->knowledgs.begin();it!=knowledgs.end();it++){
		int knowid=(*it).second;
		outfs.write((char*)&knowid, sizeof(int));
		cnt=neurons[knowid].inneurons.size();
		outfs.write((char*)&cnt, sizeof(int));
		for(int i=0;i<neurons[knowid].inneurons.size();i++){
			outfs.write((char*)&neurons[knowid].inneurons[i], sizeof(int));
		}
		cnt =it->first.length();
		outfs.write((char*)&cnt, sizeof(int));
		outfs.write(it->first.data(),it->first.length());
	}

	outfs.close();
	printf("%s ok!\n",filename);
}

void opticnerve::save1(const char* filename)
{
	std::ofstream outfs;
	int data;
	outfs.open(filename,ios::binary|ios::out);//,std::ios::binary
	outfs.write((char*)&neuronscnt, sizeof(int));
	//outfs.write((char*)neutimes, sizeof(int)*neuronscnt);
	//outfs.write((char*)&curtimes, sizeof(int));

	int cnt=(palliumlayers.size());
	outfs.write((char*)&cnt, sizeof(int));
	for(vector<vector<int>*>::iterator it=palliumlayers.begin();it!=palliumlayers.end();it++){
		cnt=(*it)->size();
		outfs.write((char*)&cnt, sizeof(int));
		for(vector<int>::iterator layer=(*it)->begin();layer!=(*it)->end();layer++){
			data=(*layer);
			outfs.write((char*)&data, sizeof(data));
			cnt=neurons[data].fromsynapses.size();
			outfs.write((char*)&cnt, sizeof(cnt));
			for(vector<int>::iterator itf=neurons[data].fromsynapses.begin();itf!=neurons[data].fromsynapses.end();itf++){
				cnt=(*itf);
				outfs.write((char*)&cnt, sizeof(cnt));
			}
			cnt=neurons[data].outneurons.size();
			outfs.write((char*)&cnt, sizeof(cnt));
			for(vector<int>::iterator ito=neurons[data].outneurons.begin();ito!=neurons[data].outneurons.end();ito++){
				cnt=(*ito);
				outfs.write((char*)&cnt, sizeof(cnt));
			}

		}
	}
	cnt=(knowledgs.size());
	outfs.write((char*)&cnt, sizeof(int));
	for(KNOWLEDGES::iterator it=this->knowledgs.begin();it!=knowledgs.end();it++){
		int knowid=(*it).second;
		outfs.write((char*)&knowid, sizeof(int));
		cnt=neurons[knowid].inneurons.size();
		outfs.write((char*)&cnt, sizeof(int));
		for(int i=0;i<neurons[knowid].inneurons.size();i++){
			outfs.write((char*)&neurons[knowid].inneurons[i], sizeof(int));
		}
		cnt =it->first.length();
		outfs.write((char*)&cnt, sizeof(int));
		outfs.write(it->first.data(),it->first.length());
	}

	outfs.close();
	printf("%s ok!\n",filename);
}

void opticnerve::save2(const char* filename)
{
	std::ofstream outfs;
	int data,cnt;
	outfs.open(filename,ios::binary|ios::out);//,std::ios::binary
	outfs.write((char*)&neuronscnt, sizeof(int));

	cnt=(knowledgs.size());
	outfs.write((char*)&cnt, sizeof(int));
	for(KNOWLEDGES::iterator it=this->knowledgs.begin();it!=knowledgs.end();it++){
		int knowid=(*it).second;
		outfs.write((char*)&knowid, sizeof(int));
		cnt=neurons[knowid].inneurons.size();
		outfs.write((char*)&cnt, sizeof(int));
		for(int i=0;i<neurons[knowid].inneurons.size();i++){
			outfs.write((char*)&neurons[knowid].inneurons[i], sizeof(int));
		}
		cnt =it->first.length();
		outfs.write((char*)&cnt, sizeof(int));
		outfs.write(it->first.data(),it->first.length());
	}


	//outfs.write((char*)&cnt, sizeof(int));
	for(vector<vector<int>*>::iterator it=palliumlayers.begin();it!=palliumlayers.end();it++){
		cnt=(*it)->size();
		//outfs.write((char*)&cnt, sizeof(int));
		for(vector<int>::iterator layer=(*it)->begin();layer!=(*it)->end();layer++){
			data=(*layer);
			outfs.write((char*)&data, sizeof(data));
			//cnt=neurons[data].fromsynapses.size();
			//outfs.write((char*)&cnt, sizeof(cnt));
			for(vector<int>::iterator itf=neurons[data].fromsynapses.begin();itf!=neurons[data].fromsynapses.end();itf++){
				cnt=(*itf);
				outfs.write((char*)&cnt, sizeof(cnt));
			}
		}
	}
	outfs.close();
	printf("%s ok!\n",filename);
}

void opticnerve::load(const char* filename ) {
	std::ifstream infs;
	printf("load from %s.\n",filename);
	infs.open(filename,ios::binary|ios::in);//
	infs.read((char*)&neuronscnt,sizeof(int));
	//infs.read((char*)neutimes,sizeof(int)*neuronscnt);
	//infs.read((char*)&curtimes,sizeof(int));

	int cntlay;
	int t;
	int cnt;
	infs.read((char*)&cntlay,sizeof(int));
	//is>>cnt;
	//int curlay=0;
	while(cntlay--){
		vector<int>* lay=new vector<int>();
		palliumlayers.push_back(lay);
		infs.read((char*)&cnt,sizeof(int));
		while(cnt--){
			infs.read((char*)&t,sizeof(int));
			lay->push_back(t);
			neurons[t].layer=palliumlayers.size()-1;
			int fcnt=0;
			infs.read((char*)&fcnt,sizeof(fcnt));
			while(fcnt--){
				int ft=0;
				infs.read((char*)&ft,sizeof(int));
				connect(ft,t);
				neuthreshold[t]+=neuthreshold[ft];
			}
		}
		//curlay++;
	}

	infs.read((char*)&cnt,sizeof(int));
	while(cnt--){
		int knowid;
		infs.read((char*)&knowid,sizeof(int));
		neuron &me=neurons[knowid];
		int incnt;
		infs.read((char*)&incnt,sizeof(int));
		while(incnt--) {
			infs.read((char*)&t,sizeof(int));
			me.inneurons.push_back(t);
			neurons[t].outneurons.push_back(knowid);
		}
		char buf[1024];
		infs.read((char*)&t,sizeof(int));
		infs.read(buf,t);
		buf[t]='\0';
		string label=buf;
		this->knowledgs.insert(KNOWLEDGE(label,knowid));
		this->knowidx.insert(pair<int,string>(knowid,label));
	}


	infs.close();
	status();
}

void opticnerve::load1(const char* filename ) {
	std::ifstream infs;
	printf("load from %s.\n",filename);
	infs.open(filename,ios::binary|ios::in);//
	infs.read((char*)&neuronscnt,sizeof(int));
	//infs.read((char*)neutimes,sizeof(int)*neuronscnt);
	//infs.read((char*)&curtimes,sizeof(int));

	int cntlay;
	int t;
	int cnt;
	infs.read((char*)&cntlay,sizeof(int));
	//is>>cnt;
	//int curlay=0;
	while(cntlay--){
		vector<int>* lay=new vector<int>();
		palliumlayers.push_back(lay);
		infs.read((char*)&cnt,sizeof(int));
		while(cnt--){
			infs.read((char*)&t,sizeof(int));
			lay->push_back(t);
			neurons[t].layer=palliumlayers.size()-1;
			int fcnt=0;
			infs.read((char*)&fcnt,sizeof(fcnt));
			while(fcnt--){
				int ft=0;
				infs.read((char*)&ft,sizeof(int));
				connect(ft,t);
				neuthreshold[t]+=neuthreshold[ft];
			}
			infs.read((char*)&fcnt,sizeof(fcnt));
			while(fcnt--){
				int ft=0;
				infs.read((char*)&ft,sizeof(int));
				neurons[t].outneurons.push_back(ft);
			}
		}
		//curlay++;
	}

	infs.read((char*)&cnt,sizeof(int));
	while(cnt--){
		int knowid;
		infs.read((char*)&knowid,sizeof(int));
		neuron &me=neurons[knowid];
		int incnt;
		infs.read((char*)&incnt,sizeof(int));
		while(incnt--) {
			infs.read((char*)&t,sizeof(int));
			me.inneurons.push_back(t);
			neurons[t].outneurons.push_back(knowid);
		}
		char buf[1024];
		infs.read((char*)&t,sizeof(int));
		infs.read(buf,t);
		buf[t]='\0';
		string label=buf;
		this->knowledgs.insert(KNOWLEDGE(label,knowid));
		this->knowidx.insert(pair<int,string>(knowid,label));
	}


	infs.close();
	status();
}
void opticnerve::load2(const char* filename ) {
	std::ifstream infs;
	int cnt,t;
	printf("load from %s.\n",filename);
	infs.open(filename,ios::binary|ios::in);//
	infs.read((char*)&neuronscnt,sizeof(int));

	infs.read((char*)&cnt,sizeof(int));
	while(cnt--){
		int knowid;
		infs.read((char*)&knowid,sizeof(int));
		neuron &me=neurons[knowid];
		int incnt;
		infs.read((char*)&incnt,sizeof(int));
		while(incnt--) {
			infs.read((char*)&t,sizeof(int));
			me.inneurons.push_back(t);
		}
		char buf[1024];
		infs.read((char*)&t,sizeof(int));
		infs.read(buf,t);
		buf[t]='\0';
		string label=buf;
		this->knowledgs.insert(KNOWLEDGE(label,knowid));
	}

	while(!infs.end){
		int data[3];
		infs.read((char*)&data,sizeof(int)*3);
		connect(data[1],data[0]);
		connect(data[2],data[0]);
	}
	createlayer();

	infs.close();
	status();
}
void opticnerve::createlayer(){
	int cnt=0;
	char *flags=new char[neuronscnt];
	memset(flags,0,sizeof(char)*neuronscnt);
	while(cnt!=neuronscnt){
		vector<int>* lay=new vector<int>();
		palliumlayers.push_back(lay);
		for(int i=FEELEN;i<neuronscnt;i++){
			bool haschild=false;
			for( vector<int>::iterator it=neurons[i].fromsynapses.begin();it!=neurons[i].fromsynapses.end();it++){
				if(*(flags+*it)!='\0')
					haschild=true;
			}
			if(haschild==false) {
				*(flags + i) = '1';
				lay->push_back(i);
				cnt++;
			}
		}
	}
}

void opticnerve::status() {
	printf("knows:%d, lays:%d, neurons:%d, synapes:%d\n",knowledgs.size(),palliumlayers.size(),neuronscnt-unused.size(),countsynapse());
}

//void opticnerve::statusto(int n,vector<int> &knows) {
//	if(neurons[n].outneurons.size()>0){
//		map<int,string>::iterator it= knowidx.find(neurons[n].outneurons[0]);
//		if(it!=knowidx.end()){
//			//printf("%d:%s ",neurons->outneurons.size(),it->second.data());
//			bool add=true;
//			for(int i=0;i<knows.size();i++) {
//				//already in
//				if (knows[i] == it->first){
//					add=false;
//					break;
//				}
//			}
//			if(add) {
//				knows.push_back(it->first);
//				//printf("%d \n",it->second);
//			}
//
//		}
//	}
//	for(set<int>::iterator it=neurons[n].tosynapses.begin();it!=neurons[n].tosynapses.end();it++){
//		statusto(*it,knows);
//	}
//}
void opticnerve::statusto(int n,vector<int> &knows) {
	if(neurons[n].outneurons.size()>0){
		bool add=true;
		for(int i=0;i<knows.size();i++) {
			//already in
			if (knows[i] == neurons[n].outneurons[0]){
				add=false;
				break;
			}
		}
		if(add) {
			knows.push_back( neurons[n].outneurons[0]);
		}

	}
	for(set<int>::iterator it=neurons[n].tosynapses.begin();it!=neurons[n].tosynapses.end();it++){
		statusto(*it,knows);
	}
}


void opticnerve::reappear(int neu) {
	if(neu<FEELEN)
		*neurons[neu].pVal=(255<<8);
	for(vector<int>::iterator it=neurons[neu].fromsynapses.begin();it!=neurons[neu].fromsynapses.end();it++){
		reappear(*it);
	}
}

void opticnerve::reduce() {
	vector<int> knows;
	for(int i=palliumlayers.size();i>0;i--){
		//printf("process layer:%d\n",i);
		for(int j=0;j<palliumlayers[i-1]->size();){
			int n=(*palliumlayers[i-1])[j];
			knows.clear();
			statusto(n,knows);
			if(knows.size()==1){
				neurons[n].outneurons.push_back(knows[0]);
			    //printf("lay:%d id:%d,know:%s\n",i,n,knows[0].data());
			    //clearfeel();
			    //reappear(n);
			    //unfeel();
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

void opticnerve::removeneuron(int n) {
	while(neurons[n].tosynapses.size()>0){
		int cur = *neurons[n].tosynapses.begin();
		int prefrom;
		prefrom=neurons[cur].fromsynapses[0];
		disconnect(n,cur);
		while(neurons[cur].tosynapses.size()>0){
			int nextto= *neurons[cur].tosynapses.begin();
			disconnect(cur,nextto);
			connect(prefrom,nextto);
			neuthreshold[nextto] = neuthreshold[nextto]-neuthreshold[cur]+neuthreshold[prefrom];
		}
		vector<int>::iterator itlay=find((*palliumlayers[neurons[cur].layer]).begin(),
										 (*palliumlayers[neurons[cur].layer]).end(),cur);
		if(itlay!=(*palliumlayers[neurons[cur].layer]).end())
		{
			palliumlayers[neurons[cur].layer]->erase(itlay);
			printf("del: %d\n",cur);
		}else{
			printf("end: %d\n",cur);
		}
	}
	if(neurons[n].fromsynapses.size()!=2){
		printf("%d fromsynapses not 2\n",n);
	}
	disconnect(neurons[n].fromsynapses[0],n);
	disconnect(neurons[n].fromsynapses[1],n);
	unused.push_back(n);

}

int opticnerve::getneuron() {
	if(unused.size()>0){
		int last=*unused.rbegin();
		unused.pop_back();
		return last;

	}
	//neutimes[neuronscnt]=curtimes;
	return this->neuronscnt++;
}

void opticnerve::treenodeto(int n) {
	vector<int> knows;
	statusto(n,knows);
	if(knows.size()==1){
		printf("*%d",knows[0]);
	}
	printf("(");
	for(int i=0;i<neurons[n].fromsynapses.size();i++){
		printf("%d ",neurons[n].fromsynapses[i]);
	}
	printf(")->%d[%d/%d] \n",n,*(neurons[n].pVal),neuthreshold[n]);
	for(vector<int>::iterator it=neurons[n].fromsynapses.begin();it!=neurons[n].fromsynapses.end();it++){
		treenodeto(*it);
	}

}

void opticnerve::reducenode(int n, int *pcut) {
	if(neurons[n].layer<0)
	    return;

	for(int i=neurons[n].fromsynapses.size()-1;i>=0;i--){
		int cur=neurons[n].fromsynapses[i];
		if(neurons[cur].tosynapses.size()==1 && *pcut>1) {
			reducenode(cur, pcut);
		}
	}
	if(neurons[n].tosynapses.size()==1){
		if(neuronsdata[n]< *pcut){
//			int to=*(neurons[n].tosynapses.begin());
//			neuthreshold[to]-=neuthreshold[n];
//			disconnect(n,to);
//			for(int i=neurons[n].fromsynapses.size()-1;i>=0;i--){
//				disconnect(neurons[n].fromsynapses[i],n);
//			}
//
//			vector<int>::iterator itlay=find((*palliumlayers[neurons[n].layer]).begin(),
//								 (*palliumlayers[neurons[n].layer]).end(),n);
//			if(itlay!=(*palliumlayers[neurons[n].layer]).end())
//			{
//				palliumlayers[neurons[n].layer]->erase(itlay);
//				printf("del: %d\n",n);
//			}else{
//				printf("end: %d\n",n);
//			}
//			unused.push_back(n);

			vector<int>::iterator itlay=find((*palliumlayers[neurons[n].layer]).begin(),
								 (*palliumlayers[neurons[n].layer]).end(),n);
			if(itlay!=(*palliumlayers[neurons[n].layer]).end())
			{
				palliumlayers[neurons[n].layer]->erase(itlay);
				//printf("del: %d\n",n);
			}else{
				//printf("end: %d\n",n);
			}
			removeneuron(n);
			*pcut -=neuronsdata[n];
			return;
		}
	}
//	if(neurons[n].fromsynapses.size()==1){
//		removeneuron(n);
//	}else{
//		printf("aftcut size:%d\n",neurons[n].fromsynapses.size());
//	}


}

void opticnerve::think() {
	status();
	printf("begin think...");
	for(KNOWLEDGES::iterator it=knowledgs.begin();it!=knowledgs.end();it++) {
		neuron neu = neurons[it->second];
		for(int i=0;i<neu.inneurons.size();i++) {
			int n = neu.inneurons[i];

			clearfeel();
			reappear(n);
			vector<int> allmax;
			calculate(allmax);
			if (allmax.size() > 1) {
				int k0 = allmax[0];
				int k1 = allmax[1];
				vector<int> actived0, actived1;
				printf("%s:%d  %s:%d \n", knowidx[k0].data(), neuronsdata[neurons[k0].actor],
					   knowidx[k1].data(), neuronsdata[neurons[k1].actor]);
				int cut = neuronsdata[neurons[k0].actor] - neuronsdata[neurons[k1].actor];
				reducenode(neurons[k0].actor, &cut);
				allmax.clear();
			} else {
				printf("only one max know get\n");
			}
		}
	}
	printf("end think.");
	status();
}


int opticnerve::predict() {
	vector<int> outs=vector<int>();
    for (vector<vector<int>*>::iterator it=this->palliumlayers.begin();it!=palliumlayers.end();it++) {
        //#pragma omp parallel for //gen man more slower
        for (int i = 0; i < (*it)->size(); i++) {
            int idx = (*(*it))[i];
            neuron *nu = &(this->neurons[idx]);
			if(nu->fromsynapses.size()==2){
				this->neuronsdata[idx] = this->neuronsdata[this->neurons[idx].fromsynapses[0]]
										 +this->neuronsdata[this->neurons[idx].fromsynapses[1]];
			}else{
	            this->neuronsdata[idx] =0;
				for(vector<int>::iterator itfrom= nu->fromsynapses.begin() ;itfrom!= nu->fromsynapses.end();itfrom++){
					//#pragma omp critical
					this->neuronsdata[idx] += this->neuronsdata[*itfrom];
				}
				printf("neu[%d] have %d fromsynapes\n",idx,nu->fromsynapses.size());
			}
			if(neuronsdata[idx]>=neuthreshold[idx] && neurons[idx].outneurons.size()>0){
				//actived
				if(outs.size()==0)
					outs.assign(nu->outneurons.begin(),nu->outneurons.end());
				else{
					vector<int> newout=vector<int>();
					set_intersection(outs.begin(),outs.end(),nu->outneurons.begin(),nu->outneurons.end(),
									 std::back_inserter(newout));
					outs.assign(newout.begin(),newout.end());
				}
				if(outs.size()==1){
					return *outs.begin();
				}
			}
        }
    }

	return 0;
}

void opticnerve::getfocus() {
	//init focus connect
	int data[FEELWIDTH][FEELWIDTH]={0};
	int max,mcnt;
	for(int i=0;i<FEELWIDTH;i++){
		mcnt=0;max=0;
		for(int r=i;r<FEELWIDTH-i;r++){
			for (int c = i; c < FEELWIDTH-i; c++) {
				for(int r1=-i;r1<i+1;r1++){
					for(int c1=-i;c1<i+1;c1++){
						if(r1==-i || r1==i-1 || c1==-i || c1==i-1)//add broad
							data[r][c]+=neuronsdata[(r+r1)*FEELWIDTH+(c+c1)];
					}
				}
				if(data[r][c]>max){
					max=data[r][c];mcnt=1;
				}else if(data[r][c]==max){
					mcnt++;
				}
			}
		}
		printf("i:%d, max:%d, mcnt:%d\n",i,max,mcnt);
		if(mcnt==1){
			//get one max;
            //imshow("",data);
		}

	}

	//focus calculate
	//return focus point


}

void opticnerve::removealone() {
	int curlay=-1;
    for (vector<vector<int>*>::iterator it=this->palliumlayers.begin();it!=palliumlayers.end();it++) {
    	curlay++;
		for (int i = 0; i < (*it)->size(); i++) {
			int idx = (*(*it))[i];
			neuron *nu = &(this->neurons[idx]);
			if (nu->fromsynapses.size() == 1) {
				int from = nu->fromsynapses[0];
				printf("%d layer remove %d \n",curlay,idx);
				vector<int> tos;
				tos.assign(nu->tosynapses.begin(),nu->tosynapses.end());
				disconnect(from,idx);
				neuthreshold[idx]-=neuthreshold[from];
				for(vector<int>::iterator ittt=tos.begin();ittt!=tos.end();ittt++){
					disconnect(idx,*ittt);
					connect(from,*ittt);
					neuthreshold[*ittt]+=(neuthreshold[from]-neuthreshold[idx]);
				}
				vector<int>::iterator itlay=find((*palliumlayers[neurons[idx].layer]).begin(),
										 (*palliumlayers[neurons[idx].layer]).end(),idx);
				if(itlay!=(*palliumlayers[neurons[idx].layer]).end())
				{
					palliumlayers[neurons[idx].layer]->erase(itlay);
					printf("%d lay del %d\n",neurons[idx].layer,idx);
				}else{
					printf("%d lay not found %d\n",neurons[idx].layer,idx);
				}
			}
		}
	}
}

void opticnerve::setoutneurons() {
	printf("have any erros, not use it,remember  proces neurons already.");
	for (int i = palliumlayers.size(); i > 0; i--) {
		for (int j = 0; j < palliumlayers[i - 1]->size();j++) {
			int n = (*palliumlayers[i - 1])[j];
			if(neurons[n].tosynapses.size()==0)
				continue;
			//neurons[n].outneurons.clear();
			for (set<int>::iterator it = neurons[n].tosynapses.begin(); it != neurons[n].tosynapses.end(); it++) {
				int to = *it;
				for(vector<int>::iterator outit=neurons[to].outneurons.begin();outit!=neurons[to].outneurons.end();outit++){
					vector<int>::iterator loc=find(neurons[n].outneurons.begin(),neurons[n].outneurons.end(),*outit);
					if(loc==neurons[n].outneurons.end()){
						printf(" %s ",knowidx[ *outit].data());
						neurons[n].outneurons.push_back(*outit);
					}
				}
			}
			printf(" <=%d\n",n);
		}
	}
}

int opticnerve::createtree(vector<int> &actived) {
	int cnt=actived.size();//must >1
	int ilay=0;
	if(cnt>1){
		int n = getneuron();
		for(int i=0;i<cnt;i++){
			connect(actived[i],n);
			neuthreshold[n]+=neuthreshold[actived[i]];
			if(ilay<neurons[actived[i]].layer)
				ilay=neurons[actived[i]].layer;
		}
		ilay++;
		if(palliumlayers.size() < ilay + 1){
			vector<int>* lay=new vector<int>();
			palliumlayers.push_back(lay);
		}
		palliumlayers[ilay]->push_back(n);
		neurons[n].layer=ilay;
		return n;
	}
	else if (cnt==1){
		return actived[0];
	}else{
		printf("create tree error, no actived \n");
		return -1;
	}
}

void opticnerve::see(vector<int> &allmax) {

	calculate(allmax);
}
