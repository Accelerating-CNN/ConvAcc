#include "cnn.h"
#include <iostream>
#include <unistd.h>
extern "C" {
#include "pynq_api.h"
}


using namespace std;



FLOAT* FLATTEN_INDEX_weight(Tensor * W, int number_of_filter){

    int w_channel = W->size[0];
    int w_height = W->size[1];
    int w_width = W->size[2];

    FLOAT* full_array = new FLOAT[number_of_filter *  w_channel * w_height * w_width];
    int flatten_index = 0;
    for(int i = 0;i<number_of_filter;i++){
        for(int j=0;j<w_channel;j++){
            for(int k=0;k<w_height;k++){
                for(int l=0;l<w_width;l++){
                    full_array[flatten_index] = W[i][j][k][l];
                    flatten_index++;
                }
            }
        }
    }

    return  full_array;
}

FLOAT* FLATTEN_INDEX_FPGA(Tensor * X){
    int x_channel = X->size[0];
    int x_height = X->size[1];
    int x_width = X->size[2];
    FLOAT* flatten_x = new FLOAT[x_channel*x_height*x_width];
    for(int i = 0; i<x_channel*x_height*x_width;i++){ 
        flatten_x[i]=0; 
    }
    int flatten_index =0;
    for(int i = 0;i<x_channel;i++){
        for(int j=0;j<x_height;j++){
            for(int k=0;k<x_width;k++){
                flatten_x[flatten_index] = X[0][i][j][k];
                flatten_index++;
            }
        }
    }
    return  flatten_x;
}

namespace ml {

CNN::CNN(std::vector<CNN_layer_struct> in_layers)
{
	uint32_t insize = 0;
	layers = in_layers;
	// The tricky part is allocating the proper tensors
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU: case Layer_Type::Softmax:
				lay.output_size[0] = layers[i - 1].output_size[0];
				lay.output_size[1] = layers[i - 1].output_size[1];
				lay.output_size[2] = layers[i - 1].output_size[2];
				if(lay.in_place)
					lay.Z = layers[i - 1].Z;
				else
					lay.Z = new Tensor(layers[i-1].output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::Pool:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::Conv:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				lay.W = new Tensor[lay.output_size[0]]();
				for(int i =0 ; i < lay.output_size[0]; i++){
					lay.W[i].allocate(lay.input_channels,lay.kernel_width,lay.kernel_width);
				}
				lay.B = new Tensor(1,1,lay.output_size[0]);
				break;
			case Layer_Type::Linear:
				insize = layers[i-1].output_size[0] * layers[i-1].output_size[1] * layers[i-1].output_size[2];
				lay.Z = new Tensor(1,1,lay.output_size[2]);
				lay.W = new Tensor(1,lay.output_size[2],insize);
				lay.B = new Tensor(1,1,lay.output_size[2]);
				break;
			default:
				throw std::runtime_error("Layer not implemented !\n");
		}
	}
}



CNN::~CNN()
{
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU:
				if(!(lay.in_place))
					delete lay.Z;
				break;
			case Layer_Type::Softmax:
			case Layer_Type::Pool:
				delete lay.Z;
				break;
			case Layer_Type::Conv:
				delete lay.Z;
				delete [] lay.W;
				delete lay.B;
				break;
			case Layer_Type::Linear:
				delete lay.Z;
				delete lay.W;
				delete lay.B;
				break;
			default:
				printf("Rogue unimplemented layer found during deallocation !\n");
		}
	}
}
    float * testFunction(int input_channel, int input_height, int input_width, float * x, int filter_channel
,int filter_height, int filter_width, float * w, int z_channel, int z_width, int z_height,  float * z, float * b)
{       
	  	int number_of_filter = z_channel;
        int bias_channel = z_channel;
        int bias_height = 1;
        int bias_width = 1;
        int input_size = input_channel * input_width * input_height;
        int z_size = z_channel * z_width * z_height;
        int filter_size = number_of_filter * filter_channel * filter_width * filter_height;
        int bias_size = bias_channel * bias_width * bias_height;
        
        //L1
        if(input_channel == 3 && z_channel == 128)
        {
            
            //printf("Trying to load L1 bitstream\n");
            if (PYNQ_loadBitstream((char *)"src/L1.bit") == PYNQ_SUCCESS) 
            {
                printf("----------------------  Loaded Layer 1 in FPGA ----------------------\n");
            }
            else
            {
                printf("Failed to load L1 bitstream\n");
            }
        }
        //L2
        else if(input_channel == 128 && z_channel == 256) {
            //printf("Trying to load L2 bitstream\n");
            if (PYNQ_loadBitstream((char *)"src/L2.bit") == PYNQ_SUCCESS) 
            {
                printf("----------------------  Loaded Layer 2 in FPGA ----------------------\n");
            }
            else
            {
                printf("Failed to load L2 bitstream\n");
            }
        }
        //L3
        else if(input_channel == 256 && z_channel == 256) 
        {   
            //printf("Trying to load L3 bitstream\n");
            if (PYNQ_loadBitstream((char *)"src/L3.bit") == PYNQ_SUCCESS) 
            {
                printf("----------------------  Loaded Layer 3 in FPGA ----------------------\n");
            }
            else
            {
                printf("Failed to load L3 bitstream\n");
            }
        

        }
        //L4
        else if(input_channel == 256 && z_channel == 512) 
        {
            
            //printf("Trying to load L4 bitstream\n");
            if (PYNQ_loadBitstream((char *)"src/L4.bit") == PYNQ_SUCCESS) 
            {
                printf("----------------------  Loaded Layer 4 in FPGA ----------------------\n");
            }
            else
            {
                printf("Failed to load L4 bitstream\n");
            }
            
        }
        //L5
        else if(input_channel == 512 && z_channel == 512) 
        {
            //printf("Trying to load L5 bitstream\n");
            if (PYNQ_loadBitstream((char *)"src/L5.bit") == PYNQ_SUCCESS) 
            {
                //printf("----------------------  Loaded Layer 5 in FPGA ----------------------\n");
            }
            else
            {
                printf("Failed to load L5 bitstream\n");
            }
        }
        PYNQ_MMIO_WINDOW led, hls;
        PYNQ_createMMIOWindow(&led, 0x40010000, 8);
        PYNQ_createMMIOWindow(&hls, 0x40000000, 128);
        PYNQ_SHARED_MEMORY sm_x, sm_w, sm_z;
        PYNQ_allocatedSharedMemory(&sm_x, input_width * input_height * sizeof(float), 0);
        PYNQ_allocatedSharedMemory(&sm_w, filter_width * filter_height * sizeof(float), 0);
        PYNQ_allocatedSharedMemory(&sm_z, z_width * z_height * sizeof(float), 0);
        uint32_t* b_led = (uint32_t*)led.buffer;
        b_led[1] = 0;
        b_led[0] = 3;
        // print w in a nice way
        float* virt_x = (float*)sm_x.pointer;
        float* virt_w = (float*)sm_w.pointer;
        float* virt_z = (float*)sm_z.pointer;
        memcpy(hls.buffer + 0x1c, &(sm_x.physical_address), sizeof(size_t));
        memcpy(hls.buffer + 0x28, &(sm_w.physical_address), sizeof(size_t));
        memcpy(hls.buffer + 0x10, &(sm_z.physical_address), sizeof(size_t));
        float* result = new float[z_width*z_height*z_channel];
        for(int i = 0; i<z_width*z_height*z_channel;i++)
        { 
            z[i]=0; 
            result[i]=0; 
        }
        float* b_channel = new float[bias_height*bias_width];
        float* x_channel = new float[input_height*input_width];
        float* w_channel = new float[filter_height*filter_width];
        float* z_channels = new float[z_width*z_height];
        for(int i = 0; i<input_channel;i++)
        {  
            for(int k = 0; k<input_height*input_width;k++)
            { 
                x_channel[k]=x[i*input_height*input_width+k]; 
            }
            for(int k = 0; k<bias_height*bias_width;k++)
            { 
                b_channel[k]=b[i*bias_height*bias_width+k]; 
            } 
            for(int j = 0; j<number_of_filter;j++)
            {  
                for(int k = 0; k<filter_height*filter_width;k++)
                {
                    w_channel[k] = w[i*filter_height*filter_width+j*input_channel*filter_height*filter_width+k]; 
                }
                memcpy(virt_x, x_channel, sizeof(float) * input_height*input_width);
                memcpy(virt_w, w_channel, sizeof(float) * filter_height*filter_width);
                volatile uint32_t* hls_ctrl = (uint32_t*)hls.buffer; 
                *hls_ctrl = 0b1; 
                while(!(*hls_ctrl & 0b100)){};
                memcpy(z_channels, virt_z, sizeof(float) * z_width*z_height);
                int offset = j * z_width * z_height;
                for (int m = 0; m < z_width * z_height; m++)
                {
                    float bias_p = b[j] / static_cast<float>(input_channel);
                    result[offset + m] = z_channels[m] + bias_p;
                }
            }
            for (int m = 0; m < z_size; m++) 
            {
                z[m] = z[m] + result[m];
            }
        }
    delete[] x_channel;
    delete[] w_channel;
    delete[] b_channel;
    delete[] z_channels;
    delete[] result;
    PYNQ_closeMMIOWindow(&led);
    PYNQ_freeSharedMemory(&sm_x);
    PYNQ_freeSharedMemory(&sm_w);
    PYNQ_freeSharedMemory(&sm_z);
}

Tensor * CNN::inference(Tensor * input)
{
	Tensor * X = input;
    for(int i = 0; i < layers.size(); i++){
        CNN_layer_struct & lay = layers[i];

        switch (lay.type) {
            case Layer_Type::Linear:
            {
                auto start = mtick();
                Linear(X, lay.W, lay.B, lay.Z);
                double time = mtock(start);
                runtime[0] += time;
                //printf("Type: Linear\n");
                break;
            }
            case Layer_Type::Pool:
            {
                auto start = mtick();
                maxPool(X, lay.Z);
                double time = mtock(start);
                runtime[1] += time;
                //printf("Type: Pool\n");
                break;
            }
            case Layer_Type::ReLU:
            {
                auto start = mtick();
                ReLU(X, lay.Z);
                double time = mtock(start);
                runtime[2] += time;
                //printf("Type: ReLU\n");
                break;
            }
            case Layer_Type::Conv:
            {
                if((X->size[1] == 128) || (X->size[1] == 8))
                {
                    auto start = mtick();
	                X = padTensor(X, lay.pad);
                    FLOAT * x_r= FLATTEN_INDEX_FPGA(X);
                    FLOAT * w_r= FLATTEN_INDEX_weight(lay.W,lay.Z->size[0]);
                    FLOAT * b_r= FLATTEN_INDEX_FPGA(lay.B);
                    FLOAT * z_r= FLATTEN_INDEX_FPGA(lay.Z);
                    int output_channel = lay.Z->size[0];
                    int output_height = lay.Z->size[1];
                    int output_width = lay.Z->size[2];

                    int input_channel = X->size[0];
                    int input_height = X->size[1];
                    int input_width = X->size[2];
                    delete X;

                    int number_of_filters = output_channel;
                    int filter_channel = lay.W->size[0];
                    int filter_height = lay.W->size[1];
                    int filter_width = lay.W->size[2];
                    int maybe_output = lay.W->size[3];
                    int b_channel = lay.B->size[0];
                    int b_height = lay.B->size[1];
                    int b_width = lay.B->size[2];
                    testFunction(input_channel, input_height, input_width, x_r, filter_channel,
                     filter_height, filter_width, w_r, output_channel, output_width, output_height, z_r, b_r);
                    delete [] x_r;
                    delete [] w_r;
                    delete [] b_r;
                    for (int i = 0; i < output_channel; i++)
                    {
                        for (int j = 0; j < output_height; j++)
                        {
                            for (int k = 0; k < output_width; k++)
                            {
                                lay.Z[0][i][j][k] = z_r[i * output_height * output_width + j * output_width + k];
                            }
                        }
                    }
                    delete [] z_r;
                    double time = mtock(start);
                    printf("-------------------------------------------\n");
                    printf("Time taken for FPGA convolution is %f\n",time);
                    printf("-------------------------------------------\n");
                    runtime[3] += time;
                    break;
                }
                else{
                    printf("Now, we are doing Winograd convolution\n");
                    auto start2 = mtick();
                    X = padTensor(X, lay.pad);
                    Tensor* padded_X = padTensor(X, lay.pad);
                    Tensor * W_wino = winoWeights(lay.W, lay.Z->size[0]);
                    auto start = mtick();
                    convWinograd(padded_X, W_wino, lay.B, lay.Z,  lay.W->size[2]);
                    double time = mtock(start2);
                    printf("-------------------------------------------\n");
                    printf("Time taken for Wino convolution is %f\n",time);
                    printf("-------------------------------------------\n");
                    delete [] W_wino;
                    runtime[3] += time;
                    delete padded_X;
                    break;
                    
                }

            }
            case Layer_Type::Softmax:
            {
                auto start = mtick();
                Softmax(X, lay.Z);
                double time = mtock(start);
                runtime[4] += time;
                //printf("Type: Softmax\n");
                break;
            }
            default:
                printf("Rogue unimplemented layer found during inference !\n");
        }
        X = lay.Z;
    }
    return X;
}
}