typedef struct pallium_s pallium_t;
struct pallium_s{
    int palliumcnt;
    int *imgdata;
    int *pallimudata;
    neuron_t neurons[28][28];
    neuron_t *pallium;
}
__global__ void look(uint8* img){
    int tidx=threadIdx.x;
    int tidy=threadIdx.y;
    img[x,y]

}