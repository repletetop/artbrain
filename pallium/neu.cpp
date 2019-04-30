//
// Created by tiansheng on 4/18/19.
//

#include "neu.h"


namespace brn {
    extern TimerManager brain::tm;

    brain::brain() {
        Na = 150, K = 5;
        bpneus = new bpneu[28][28];
        neus = new neu[3][28][ 28];
        for (int i = 0; i < 28 ; i++)
        for(int j=0;j<28;j++)
        {
            bpneus[i][j].connect(&neus[0][i][j]);
        }
        for(int k=0;k<1;k++)
        for (int i = 1; i < 27 ; i++)
        for(int j=1;j<27;j++)
        {
//            //neus[i,j].connect(&neus[i]);
//            Synapse *s = new Synapse();
//            s->from = neus[k][i-1][j-1].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i-1][j-1].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//            s = new Synapse();
//            s->from = neus[k][i-1][j].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i-1][j].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//            s = new Synapse();
//            s->from = neus[k][i-1][j+1].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i-1][j+1].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//
//            s = new Synapse();
//            s->from = neus[k][i+1][j-1].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i+1][j-1].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//            s = new Synapse();
//            s->from = neus[k][i+1][j].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i+1][j].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//            s = new Synapse();
//            s->from = neus[k][i+1][j+1].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i+1][j+1].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//
//            s = new Synapse();
//            s->from = neus[k][i][j-1].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i][j-1].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);
//            s = new Synapse();
//            s->from = neus[k][i][j+1].ax;
//            s->dd = &neus[k][i][j].dd;
//            neus[k][i][j+1].ax->synapses.push_back(s);
//            neus[k][i][j].dd.synapses.push_back(s);

            //neus[i].connect(&neus1[i]);
            Synapse *s1 = new Synapse();
            s1->from = neus[k][i][j].ax;
            s1->dd = &neus[k+1][i][j].dd;
            neus[k][i][j].ax->synapses.push_back(s1);
            neus[k+1][i][j].dd.synapses.push_back(s1);
        }
        for (int i = 1; i < 27 ; i++)
        for(int j=1;j<27;j++)
        {
            Synapse *s1 = new Synapse();
            s1->from = neus[0][i][j].ax;
            s1->dd = &neus[2][i][j].dd;
            s1->dweight=2;
            s1->distance=0;
            neus[0][i][j].ax->synapses.push_back(s1);
            neus[2][i][j].dd.synapses.push_back(s1);

            s1 = new Synapse();
            neus[0][i-1][j].ax->rv=new ranvier();
            neus[0][i-1][j].ax->rv->synapses.push_back(s1);
            s1->dd = &neus[2][i][j].dd;
            s1->dweight=2;
            s1->distance=1;
            neus[0][i-1][j].ax->synapses.push_back(s1);
            neus[2][i][j].dd.synapses.push_back(s1);

        }

    }

    void neu::openNa() {
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(neu::nachannel), (void *) this, INTERVAL / 5, Timer::ONCE);
    }

    void neu::openK() {
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(neu::kchannel), (void *) this, INTERVAL / 5, Timer::ONCE);
    }

    void neu::run(void *objme, void *objtm) {
        neu *me = (neu *) objme;
        Timer *tm = (Timer *) objtm;
        //printf("inside neuron  Na:%3d,K:%3d, Total V:%3d.\n", me->Na, me->K, me->Na + me->K);
        if (me->Na > NATHRESHOLD && me->K >= 100) {
            //me->openNa();
            me->Na=NAOUT;
            me->ax->Na=NAOUT;
            me->K=5;
            printf("Channel Na opened, Na:%3d,K:%3d, Total V:%3d.\n", me->Na, me->K, me->Na + me->K);
        }
        if (me->Na > 15)//Na K channel count=10
            me->Na -= 3 * 1;
        if (me->K < 100)
            me->K += 2 * 1;
    }

    void neu::nachannel(void *objme, void *objtm) {
        neu *me = (neu *) objme;
        Timer *tm = (Timer *) objtm;
        me->Na = NAOUT;
        printf("Channel Na opened, Na:%3d,K:%3d, Total V:%3d.\n", me->Na, me->K, me->Na + me->K);
        //axon fire;
        me->ax->Na = NAOUT;
        me->openK();
    }

    void neu::kchannel(void *objme, void *objtm) {
        neu *me = (neu *) objme;
        Timer *tm = (Timer *) objtm;
        me->K = 5;
    }

    neu::neu():dd(this) {
        Na = 15, K = 100;
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(neu::run), (void *) this, INTERVAL);
        ax = new axon();
    }

    bpneu::bpneu() {
        to = nullptr;
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(bpneu::run), (void *) this, INTERVAL);
    }


    axon::axon() {
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(axon::run), (void *) this, INTERVAL);
        rv = nullptr;
    }

    void bpneu::run(void *objme, void *objtm) {
        {
            bpneu *me = (bpneu *) objme;
            Timer *tm = (Timer *) objtm;
            if (me->to != nullptr) {
                int v = me->Na - me->to->Na;
                if (v > 1) {
                    me->to->Na += v / 2;
                    me->Na -= v / 2;
                }
            }
        }
    }

    void ranvier::run(void *objme, void *tmobj) {
        ranvier *me = (ranvier *) objme;
        if (me->Na > 0) {
            if (me->nextrv != NULL){
                me->nextrv->Na = NAOUT;
                me->Na = 0;
            }
            for (int i = 0; i < me->synapses.size(); i++) {
                //fire
                me->synapses[i]->Na = NAOUT;
                me->Na = 0;
            }
        }
    }

    void axon::run(void *objme, void *objtm) {
        axon *me = (axon *) objme;
        if (me->Na > 0) {
            if (me->rv != nullptr) {
                me->rv->Na = NAOUT;//Fire
                me->Na = 0;
            }
            for (int i = 0; i < me->synapses.size(); i++) {
                me->synapses[i]->Na = NAOUT;
                me->Na = 0;
            }
        }
    }

    void dendritic::run(void *objme, void *objtm) {
        dendritic *me = (dendritic *) objme;
        int v = me->Na - me->nu->Na;//auto weight by dendritic na
        if (v > 1) {
            me->nu->Na += v / 2;
            me->Na -= v / 2;
        }
    }

    dendritic::dendritic(neu* pnu) {
        nu=pnu;
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(dendritic::run), (void *) this, INTERVAL);
    }

    Synapse::Synapse() {
        dweight=1;
        distance=0;
        Timer *ntm = new Timer(brain::tm);
        ntm->Start(&(Synapse::run), (void *) this, INTERVAL);
    }

    void Synapse::run(void *objme, void *objtm) {
        Synapse *me = (Synapse *) objme;
        if (me->dd != nullptr) {
            int v = (me->Na - me->dd->Na)/me->dweight;//auto weight by dendritic na
            if (v > 1) {
                me->dd->Na += v / 2;
                me->Na -= v / 2;
            }
        }
    }

}