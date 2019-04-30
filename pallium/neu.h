//
// Created by tiansheng on 4/18/19.
//

#ifndef TESTBENCH_NEU_H
#define TESTBENCH_NEU_H
#include <vector>
#include <string>
#include "Timer.h"

using  namespace std;

namespace brn {
#define NATHRESHOLD 20
#define OPEN    0
#define CLOSE   1
#define NAOUT   150
#define INTERVAL    1 //ms

    class Synapse;

    struct axon;

    class neu;

    struct dendritic;

    struct ranvier;
    class axon;


    class neubase {
    public:
        neubase(){
            Na=0;
        }
        int Na;
    };

    class Synapse:public neubase {
    public:
        Synapse();

        static void run(void *objme,void *objtm);

        axon *from;
        dendritic *dd;
        int distance;
        int dweight;//link weight for Na send to 4=>1/4 pass Na= V/dweight
    };

    struct dendritic:public neubase {
        dendritic(neu *nu);
        vector<Synapse *> synapses;
        neu *nu;
        static void run(void *objme,void* objtm);
    };



    struct ranvier:public neubase {
        axon *ax;
        ranvier *nextrv;
        vector<Synapse *> synapses;

        static void run(void *objme,void* objtm);

    };


    class neu;

    class bpneu : public neubase {
    public:
        bpneu();

        void connect(neu *pto) {
            to = pto;
        }

        neu *to;
        static void run(void *objme, void *tm);
    };

    class brain {
    public:
        brain();

        static TimerManager tm;
        bpneu (*bpneus)[28];
        neu (*neus)[28][28];
        int Na, K;
    };

    class neu:public neubase {
    public:
        neu();

        void openNa();
        //static void run(void* objme);

        dendritic dd;
        axon* ax;
        int value;
        int hist;
        int threshold = 1;
        int K;
        int status;
        int opencnt;
        string id;
    private:
        void openK();

        static void run(void *objme, void *tm);

        static void nachannel(void *objme, void *tm);

        static void kchannel(void *objme, void *tm);
    };

        //髓鞘分支,髓鞘分支 segment
    //trunk
    class axon :public neubase{
    public:
        axon();
        static void run(void *objme,void *tm);

        ranvier *rv;
        vector<Synapse *> synapses;

    };


}
#endif //TESTBENCH_NEU_H
