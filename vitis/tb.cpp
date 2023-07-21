#include <cstdio>
#include "conv.h"
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cwchar>
#include "hls_stream.h"
#include <ap_fixed.h>
using namespace std;


bool testEqual(const float* z_to_test, const float* z_from_conv ,float limit) {

    bool isEqual = true;
    double abs_diff = 0.0;

    // Iterate over each element in the tensors
    for (int c = 0; c < z_channel; ++c) {
        for (int h = 0; h < z_height; ++h) {
            for (int w = 0; w < z_width; ++w) {
                int index = c * (z_width * z_channel) + h * z_width + w;
                float diff = 0.0;
                diff = fabs(z_to_test[index] - z_from_conv[index]);
                if (diff > limit && isEqual) {
                    cout << "Values differ at channel " << c << ", height " << h << ", width " << w << " by " << diff << endl;
                    isEqual = false;
                }

                abs_diff += diff;
            }
        }
    }

    double avg_diff = abs_diff / (1.0 * z_channel * z_width * z_height);
    if (isEqual) {
        cout << "The tensors are equal." << endl;
    }

    return isEqual;
}



float* readConv(FILE* f, int size0, int size1, int size2) {
    int dims[3];
    if (1 != fread(&(dims[0]), sizeof(dims[0]), 1, f) ||
        1 != fread(&(dims[1]), sizeof(dims[1]), 1, f) ||
        1 != fread(&(dims[2]), sizeof(dims[2]), 1, f)) {
        cout << "Failed to read tensor dimensions from the file." << endl;
        //return nullptr;
    }

    size0 = dims[0];
    size1 = dims[1];
    size2 = dims[2];

    float* data = new float[size0 * size1 * size2];
    float num_params = size0 * size1 * size2;

    if (num_params != fread(data, sizeof(float), num_params, f)) {
	cout << "Failed to read tensor data from the file." << endl;
        delete[] data;
        //return nullptr;
    
    }

    return data;
}

int main(){
    const char* filePath = "/home/oktopus/Downloads/correct_vitis_version/conv_test.dat";
    FILE* f = fopen(filePath, "r");
    if (!f) {
	cout<< "Failed to open file"<<endl;
        return 1;
    }

    
    float * x = readConv(f,input_channel,input_width,input_height);
    float * z_to_test= readConv(f,z_channel,z_width,z_height);
    float * z_from_conv = new float[z_size];
    float * w = new float[filter_size];
    for (int i =0;i<number_of_filter;i++){
        float * temp = readConv(f,filter_channel,filter_width,filter_height);
        for (int j = 0; j < filter_channel*filter_width*filter_height; ++j) {
            w[i*filter_channel*filter_width*filter_height+j] = temp[j];
        }
    }
    float * b = readConv(f,bias_channel,bias_width,bias_height);
    

    EntryConv(z_from_conv,x,w);
    
   for (int i =0;i<z_size;i++){
	z_from_conv[i] += b[0];
   }

 float limit = 0.001;
  if(testEqual(z_to_test,z_from_conv,limit)){
	cout<<"equal"<<endl;
 }

     delete [] x ;
    delete []z_to_test;
    delete [] z_from_conv;
    delete [] w;
    delete [] b;
    return 0 ;
 }


