#include <string>
#include <map>
#include <list>
#include <vector>
using namespace std;


struct synapse;
struct neuron;


typedef map<string, int> KNOWLEDGES;
typedef pair<string, int> KNOWLEDGE;
class opticnerve {
public:
	opticnerve();
	void input(unsigned char * img);
	void remember(unsigned char*img,string label);
	KNOWLEDGES::iterator calculate();
	KNOWLEDGES::iterator look(unsigned char*img);
	void getactived(int nuid,vector<int> *actived);
	void setzero(int nuid);
	void layerdown(list<list<int>*>::iterator currentlayer,int nuid);


	void connect(int neufrom,int neuto ,int polarity);
	void disconnect(synapse *s);
	void disconnectfrom(int neuid,int neufrom, int polarity) ;


	neuron *neurons;
	int *neuronsdata;
	int *neuthreshold;
	int neuronscnt = 0;
	list<list<int> *> palliumlayers;
	//int *palliumidx;
	//int palliumcnt = 0;
	KNOWLEDGES knowledgs;
};