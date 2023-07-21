#include "tensor.h"
#include <vector>
#include "time.h"
#include "conv.h"
#include <cstdio>



void timeConv(uint32_t input_channels, uint32_t input_width, uint32_t kernel_size,
              uint32_t output_channels, int N, int select);

void testConv(const char * infile, int select);


int main(int argc , char * argv[])
{
    if(argc < 3){
        printf("Usage (test) : ./lab3.bin 0 optim\n");
        printf("Usage (bench): ./lab3.bin 1 optim input_channels input_width kernel_size output_channels N\n");
        return 1;
    }
    int bench = atoi(argv[1]);
    uint32_t select = atoi(argv[2]);
    if(bench){
        if(argc < 8){
            printf("Usage: ./lab3.bin 1 optim input_channels input_width kernel_size output_channels N\n");
            return 1;
        }
        uint32_t ic = atoi(argv[3]);
        uint32_t iw = atoi(argv[4]);
        uint32_t ks = atoi(argv[5]);
        uint32_t oc = atoi(argv[6]);
        uint32_t N = atoi(argv[7]);
        timeConv(ic,iw,ks,oc,N,select);
    }
    else{
        testConv("data/conv_test.dat",select);
    }
    return 0;
}






void timeConv(uint32_t input_channels, uint32_t input_width, uint32_t kernel_size,
              uint32_t output_channels, int N, int select)
{
    Tensor * X = new Tensor[N];
    Tensor * W = new Tensor[N * output_channels];
    Tensor * B = new Tensor[N];
    uint32_t out_width = input_width - kernel_size + 1;
    Tensor Z(output_channels, out_width, out_width);
    for(int i =0 ; i  < N ; i++){
        X[i].allocate(input_channels,input_width, input_width);
        X[i].randomize(-1,1);
        B[i].allocate(1,1,output_channels);
        B[i].randomize(-1,1);
        for(int j = 0; j < output_channels; j++){
            W[i*output_channels + j].allocate(input_channels,kernel_size,kernel_size);
            W[i*output_channels + j].randomize(-1,1);
        }
    }
    printf("Benchmark %d, N: %d, input size: %d,%d,%d kernel_size: %d output_channels: %d \n"
            ,select, N,input_channels,input_width,input_width, kernel_size,
           output_channels);
    double total_time = 0;
    if(select == 0){
        auto start = mtick();
        for(int i = 0; i < N ; i++){
            convBasic(&(X[i]),&(W[i * output_channels]),&(B[i]),&(Z));
        }
        total_time = mtock(start);
    }
    else if(select == 1){
        C_Tensor ** U = new C_Tensor*[N];
        for(int i =0 ; i < N; i++)
            U[i] = fftWeights(&(W[i*output_channels]),Z.size[0]);
        auto start = mtick();
        for(int i = 0; i < N ; i++){
            convFFT(&(X[i]),U[i],&(B[i]),&Z,W->size[2]);
        }
        total_time = mtock(start);
        for(int i =0 ; i < N; i++)
            //delete U[i];
            delete [] U;
    }
    else if(select == 2){
        Tensor ** U = new Tensor*[N];
        for(int i =0; i < N; i++){
            U[i] = winoWeights(&(W[i*output_channels]),Z.size[0]);
        }
        auto start = mtick();
        for(int i =0; i < N ; i++){
            convWinograd(&(X[i]),U[i],&(B[i]),&Z,W->size[2]);
        }
        total_time = mtock(start);
        for(int i =0 ; i < N ; i++)
            delete [] U[i];
        delete [] U;
    }
    else{
        printf("%d not implemented yet!\n",select);
    }
    printf("Total Time [ms]: %lf \n",total_time);
    printf("Avg. Time [ms]: %lf \n",total_time/N);
}


Tensor * readConv(Tensor * X, Tensor * Ref, Tensor * B , FILE * f);

void testConv(const char * infile,int select)
{
    FILE * f = fopen(infile,"rb");
    Tensor X,R,B;
    printf("------------------------------\n");
    printf("Testing Convolutional Layer...\n");
    while(1){
        Tensor * W = readConv(&X,&R,&B,f);
        if(W == NULL)
            break;
        Tensor Z(R.size[0],R.size[1],R.size[2]);
        Z.randomize(-1,1);
        printf("Test X:[%dx%dx%d] W:[%dx%d] Output channels: %d!\n",X.size[0],X.size[1],X.size[2],
               W->size[1],W->size[2],R.size[0]);
        /* Select Optimization */
        if(select == 0){
            auto start = mtick();
            convBasic(&X,W,&B,&Z);
            double end = mtock(start);
            printf("Time needed: %lf mseconds\n", end);
        }
        else if(select == 1){
            auto start = mtick();
            C_Tensor * U = fftWeights(W,Z.size[0]);
            convFFT(&X,U,&(B),&(Z),W->size[2]);
            double end = mtock(start);
            printf("Time needed: %lf mseconds\n", end);
            delete [] U;
        }
        else if(select == 2){
            auto start = mtick();
            Tensor * U = winoWeights(W,R.size[0]);
            convWinograd(&X,U,&B,&Z,W->size[2]);
            double end = mtock(start);
            printf("Time needed: %lf mseconds\n", end);
            delete [] U;
        }
        else
            printf("Not implemented %d!\n",select);
        compareTensors(&Z,&R,1,1e-3);
        delete [] W;
    }
    fclose(f);
}




Tensor * readConv(Tensor * X, Tensor * Ref, Tensor * B , FILE * f)
{
    if(X->read(f) == TENSOR_READ_FAILED)
        return NULL;
    if(Ref->read(f) == TENSOR_READ_FAILED)
        return NULL;
    Tensor * W = new Tensor[Ref->size[0]]();
    for(int i = 0; i < Ref->size[0] ; i++){
        if(W[i].read(f) == TENSOR_READ_FAILED){
            delete [] W;
            return NULL;
        }
    }
    if(B->read(f) == TENSOR_READ_FAILED){
        delete [] W;
        return NULL;
    }
    return W;
}