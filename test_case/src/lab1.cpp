#include "kernels.h"
#include "tensor.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;




//Tensor * readConv(Tensor * X, Tensor *B , Tensor * Ref, FILE * f);
void testConv(const char * infile);
//void testLinear(const char * infile);
void testPool(const char * infile);
//void testSoftmax(const char * infile);
//void testReLU(const char * infile);

bool testEqual(const float * z_to_test, const float * z_from_conv, float limit );

int main(int argc , char * argv[])
{
	testConv("data/conv_test.dat");
//	testLinear("data/linear_test.dat");
//	testPool("data/pool_test.dat");
//	testSoftmax("data/softmax_test.dat");
//	testReLU("data/relu_test.dat");
	return 0;
}

float* readConv(FILE* f, int size0, int size1, int size2) {
    int dims[3];
    if (1 != fread(&(dims[0]), sizeof(dims[0]), 1, f) ||
        1 != fread(&(dims[1]), sizeof(dims[1]), 1, f) ||
        1 != fread(&(dims[2]), sizeof(dims[2]), 1, f)) {
        cout << "Failed to read tensor dimensions from the file." << endl;
        return nullptr;
    }

    size0 = dims[0];
    size1 = dims[1];
    size2 = dims[2];

    float* data = new float[size0 * size1 * size2];
    float num_params = size0 * size1 * size2;

    if (num_params != fread(data, sizeof(float), num_params, f)) {
        cout << "Failed to read tensor data from the file." << endl;
        delete[] data;
        return nullptr;
    }

    return data;
}


bool testEqual(const float* z_to_test, const float* z_from_conv ,float limit) {

    bool isEqual = true;
    double abs_diff = 0.0;

    // Iterate over each element in the tensors
    for (int c = 0; c < z_channel; ++c) {
        for (int h = 0; h < z_height; ++h) {
            for (int w = 0; w < z_width; ++w) {
                // Compute the index for the current element
                int index = c * (z_width * z_channel) + h * z_width + w;

                // Compute the absolute difference
                float diff = fabs(z_to_test[index] - z_from_conv[index]);

                // Check if the difference exceeds the limit
                if (diff > limit && isEqual) {
                    cout << "Values differ at channel " << c << ", height " << h << ", width " << w << " by " << diff << endl;
                    isEqual = false;
                }

                abs_diff += diff;
            }
        }
    }
    if (isEqual) {
        cout << "The tensors are equal." << endl;
    }
    return isEqual;
}

void testConv(const char * infile)
{

    //const char* filePath = "data/conv_test.dat";
    FILE* f = fopen(infile, "r");
    if (!f) {
        cout << "Failed to open the file." << endl;

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
    fclose(f);
    conv2d(x, w, b, z_from_conv);
    float limit = 0.0001; // Adjust the limit as needed
    if(testEqual(z_to_test, z_from_conv ,limit)){
        cout<<"Channel-by-channel convolution is correct."<<endl;
    }
    delete [] x ;
    delete []z_to_test;
    delete [] z_from_conv;
    delete [] w;
    delete [] b;
    fclose(f);

}


//void testConv(const char * infile)
//{
//	FILE * f = fopen(infile,"rb");
//	Tensor X,R,B;
//	printf("------------------------------\n");
//	printf("Testing Convolutional Layer...\n");
//	while(1){
//		Tensor * W = readConv(&X,&R,&B,f);
//		if(W == NULL)
//			break;
//		Tensor Z(R.size[0],R.size[1],R.size[2]);
//		conv2d(&X,W,&B,&Z);
//		compareTensors(&Z,&R,1,0.001);
//		delete [] W;
//	}
//	fclose(f);
//}

//void testLinear(const char * infile)
//{
//	printf("------------------------------\n");
//	printf("Testing Linear Layer...\n");
//	FILE * f = fopen(infile,"rb");
//	Tensor X,W,B,Ref;
//	Tensor * tp[4] = {&X,&Ref,&W,&B};
//	while(1){
//		for(int i = 0; i < 4; i++){
//			if(tp[i]->read(f) == TENSOR_READ_FAILED){
//				fclose(f);
//				return;
//			}
//		}
//		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
//		Linear(&X, &W , &B , &Z);
//		compareTensors(&Z,&Ref, 1, 0.001);
//	}
//}
//
//void testPool(const char * infile)
//{
//	FILE * f = fopen(infile,"rb");
//	printf("------------------------------\n");
//	printf("Testing Pool Layer...\n");
//	Tensor X,Ref;
//	while(1){
//		if(X.read(f) == TENSOR_READ_FAILED){
//			fclose(f);
//			return;
//		}
//		if(Ref.read(f) == TENSOR_READ_FAILED){
//			fclose(f);
//			return;
//		}
//		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
//		maxPool(&X,&Z);
//		compareTensors(&Z, &Ref , 1, 0.001);
//	}
//}
//
//
//void testSoftmax(const char * infile)
//{
//	printf("------------------------------\n");
//	printf("Testing Softmax Layer...\n");
//	FILE * f = fopen(infile,"rb");
//	Tensor X,Ref;
//	Tensor * tp[2] = {&X,&Ref};
//	while(1){
//		for(int i = 0; i < 2; i++){
//			if(tp[i]->read(f) == TENSOR_READ_FAILED){
//				fclose(f);
//				return;
//			}
//		}
//		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
//		Softmax(&X,&Z);
//		compareTensors(&Z,&Ref, 1, 0.001);
//	}
//}
//
//
//void testReLU(const char * infile)
//{
//	printf("------------------------------\n");
//	printf("Testing ReLU Layer...\n");
//	FILE * f = fopen(infile,"rb");
//	Tensor X,Ref;
//	Tensor * tp[2] = {&X,&Ref};
//	while(1){
//		for(int i = 0; i < 2; i++){
//			if(tp[i]->read(f) == TENSOR_READ_FAILED){
//				fclose(f);
//				return;
//			}
//		}
//		Tensor Z(Ref.size[0],Ref.size[1],Ref.size[2]);
//		ReLU(&X,&Z);
//		compareTensors(&Z,&Ref, 1, 0.001);
//	}
//}


Tensor * readConv(Tensor * X, Tensor * Ref, Tensor * B , FILE * f)
{
	if(X->read(f) == TENSOR_READ_FAILED)
		return NULL;
	if(Ref->read(f) == TENSOR_READ_FAILED)
		return NULL;
	// For multiple output channels we need a weight 
	// Tensor for every output feature map!
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

