#ifndef _OPTICNERVE_H_
#define _OPTICNERVE_H_

#include <string>
#include <hash_map>
#include <map>
#include <list>
#include <vector>
#include "neuron.hpp"

#define FEELWIDTH 128
#define FEELEN (FEELWIDTH*FEELWIDTH*2)
//#define NEURONBUFFERLEN    (FEELEN+1024*1024*1024)
#define NEURONBUFFERLEN    (1024*1024*100)

using namespace std;
using namespace __gnu_cxx;

struct neuron;


typedef map<string, int> KNOWLEDGES;
typedef pair<string, int> KNOWLEDGE;
class opticnerve {
public:
	opticnerve(int maxneucnt=NEURONBUFFERLEN);
	void clearfeel();
	void input(unsigned char * img);
	void learn(unsigned char*img,string label);
	void remember(string label);
	void calculate(vector<int> &allmax);
	void see(vector<int> &allmax);

	void look(unsigned char*img,vector<int> &allmax);
	int predict();
	void getactived(int nuid,vector<int> *actived);
	int getneuron();
	void setzero(int nuid);
	void layerafter(int currentlayer,int nuid);
	void save(const char* filename);
	void load(const char* filename);
	void save1(const char* filename);
	void load1(const char* filename);
	void save2(const char* filename);
	void load2(const char* filename);
	void createlayer();
	void status();
	void setoutneurons();
	void statusto(int n,vector<int> &knows);
	void treenodeto(int n);
	void reducenode(int neuid,int *cut);
	void reduce();
	void think();
	void removeneuron(int n);
	void removealone();
	void reappear(int neu);

	void conv2d();





	void connect(int neufrom,int neuto );
	void disconnect(int neuid,int neufrom);

	void reform();


	int countsynapse();


	int inputlen=0;

	neuron *neurons;
	int *neuronsdata;
	int *neuthreshold;
	//int *neutimes;//
	//int curtimes;//
	set<SYNAPSE,synapseSortCriterion> synapses;
	vector<vector<int> *> palliumlayers;
	//int *palliumidx;
	//int palliumcnt = 0;
	KNOWLEDGES knowledgs;
	hash_map<int,string> knowidx;

	int createbtree(vector<int> &actived);//return root idx
	int createtree(vector<int> &actived);//return root idx
private:
	int neuronscnt = 0;
	vector<int> unused;
};

#endif