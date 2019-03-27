//
// Created by tiansheng on 3/22/19.
//

#ifndef TESTBENCH_HZK16_H
#define TESTBENCH_HZK16_H


struct hzk16 {
    char* fnhzk16 = "../HZK16";
    char buf[65536];

    hzk16(){
        FILE *fid;
        fid=fopen(fnhzk16,"rb");
        fread(buf,sizeof(char),65536,fid);
        fclose(fid);
    };
    void loadmat(char hz,cv::Mat& mat){//203-213:'0-9'
        uint16_t *pbuf=(uint16_t*)(buf+(203+hz-'0')*32);
        for(int i=0;i<16;i++){
            uchar *data = mat.ptr<uchar>(i);
            uint16_t v0=*(pbuf+i);
            uint16_t v= (v0<<8) + (v0>>8);
            for(int j=0;j<16;j++){
                int flag = (v & (0x8000 >> (15 - j)))>>j;
                data[15-j]=flag*255;
            }
        }
    }
//    void loadmat(char hz,cv::Mat& mat){//203-213:'0-9'
//        char *pbuf=buf+(203+hz-'0')*32;
//        for(int i=0;i<16;i++){
//            uchar *data = mat.ptr<uchar>(i);
//            uint16_t v=((*(pbuf+2*i))*256)+(*(pbuf+2*i+1));
//            //uint16_t v= (v0<<8) + (v0>>8);
//            for(int j=0;j<16;j++){
//                int flag = (v & (0x8000 >> (15 - j)))>>j;
//                data[15-j]=flag*255;
//            }
//        }
//    }

};


#endif //TESTBENCH_HZK16_H
