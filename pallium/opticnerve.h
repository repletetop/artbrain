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
	void calculate(vector<KNOWLEDGES::iterator> &allmax);
	void look(unsigned char*img,vector<KNOWLEDGES::iterator> &allmax);
	void getactived(int nuid,vector<int> *actived);
	void setzero(int nuid);
	void layerdown(list<list<int>*>::iterator currentlayer,int nuid);


	void connect(int neufrom,int neuto ,int polarity);
	void disconnect(synapse *s);
	void disconnectfrom(int neuid,int neufrom, int polarity) ;

	void reform();


	int countsynapse();


	int inputlen=0;

	neuron *neurons;
	int *neuronsdata;
	int *neuthreshold;
	int neuronscnt = 0;
	list<list<int> *> palliumlayers;
	//int *palliumidx;
	//int palliumcnt = 0;
	KNOWLEDGES knowledgs;
};