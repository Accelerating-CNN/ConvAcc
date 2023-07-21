#include "fft.h"
#include <iostream>
#include <cmath>
#include <vector>
using  namespace std;
void C_Tensor::allocate(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	size[0] = dim_z;
	size[1] = dim_y;
	size[2] = dim_x;
	size_t dim_zy = size[1]*size[0];
	size_t dim_zyx = dim_zy * size[2]; 
	data = new C_FLOAT **[size[0]];
	C_FLOAT ** tmp_y = new C_FLOAT*[dim_zy];
	C_FLOAT * tmp_x = new C_FLOAT[dim_zyx];
	for(uint32_t i = 0; i < size[0]*size[1]; i++){
		tmp_y[i] = &(tmp_x[i * size[2]]);
	}
	for(uint32_t i = 0; i < size[0]; i++){
		data[i] = &(tmp_y[i * size[1]]);
	}
}

C_Tensor::C_Tensor(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	allocate(dim_z,dim_y,dim_x);
}

C_Tensor::~C_Tensor()
{
	if(data != NULL){
		if(data[0] != NULL){
			if(data[0][0] != NULL)
				delete [] data[0][0];
			delete [] data[0];
		}
		delete [] data;
	}
}

void fft(C_FLOAT * x_in, C_FLOAT * X_out, int N)
{
    if (N == 1)
    {
        X_out[0] = x_in[0];
        return;
    }

    C_FLOAT * x_even = new C_FLOAT[N / 2];
    C_FLOAT * x_odd = new C_FLOAT[N / 2];
    C_FLOAT * X_even = new C_FLOAT[N / 2];
    C_FLOAT * X_odd = new C_FLOAT[N / 2];

    for (int i = 0; i < N / 2; ++i)
    {
        x_even[i] = x_in[2 * i];
        x_odd[i] = x_in[2 * i + 1];
    }

    fft(x_even, X_even, N / 2);
    fft(x_odd, X_odd, N / 2);

    for (int i = 0; i < N / 2; ++i)
    {
        C_FLOAT twiddle = std::exp(C_FLOAT(0, -2 * M_PI * i / N));
        X_out[i] = X_even[i] + twiddle * X_odd[i];
        X_out[i + N / 2] = X_even[i] - twiddle * X_odd[i];
    }

    delete[] x_even;
    delete[] x_odd;
    delete[] X_even;
    delete[] X_odd;

}

void ifft(C_FLOAT * x_in, C_FLOAT * X_out, int N)
{
    for (int i = 0; i < N; ++i)
    {
        x_in[i] = std::conj(x_in[i]);
    }

    // Apply FFT on the conjugated input
    fft(x_in, X_out, N);

    // Take the conjugate and divide by N to obtain IFFT
    for (int i = 0; i < N; ++i)
    {
        X_out[i] = std::conj(X_out[i]) / static_cast<float>(N);
    }
}

/*for (int c=0;c<output_channels;c++){
for (int y=0;y<x_in->size[1];y++){
fft((*x_in)[c][y], (*X_f)[c][y], x_in->size[2]);
}
}*/

void fft2d(C_Tensor * x_in, C_Tensor * X_f, int output_channels)
{
    int input_channels = x_in->size[0];
    int kernel_height = x_in->size[1];
    int kernel_width = x_in->size[2];

    //first fft

    C_Tensor * tmp = new C_Tensor(input_channels, kernel_width, kernel_width);
    for (int c=0;c<input_channels;c++){
        for(int i =0;i<kernel_width;i++){
            fft((*x_in)[c][i], (*tmp)[c][i], kernel_height);
        }
    }


    //first transpose
    for (int c = 0; c < input_channels; c++) {
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                x_in[0][c][j][i] = tmp[0][c][i][j];
            }
        }
    }


    //second fft
    for (int c=0;c<input_channels;c++){
        for(int i =0;i<kernel_width;i++){
            fft((*x_in)[c][i], (*tmp)[c][i], kernel_height);
        }
    }

    //last transpose
    for (int c = 0; c < input_channels; c++) {
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                X_f[0][c][j][i] = tmp[0][c][i][j];
            }
        }
    }

    delete tmp;

}


void ifft2d(C_Tensor * X_f , C_Tensor * x_out,int output_channels)
{
    int input_channels = X_f->size[0];
    int kernel_height = X_f->size[1];
    int kernel_width = X_f->size[2];
    //first fft
    C_Tensor * tmp = new C_Tensor(input_channels, kernel_width, kernel_width);
    for (int c=0;c<input_channels;c++){
        for(int i =0;i<kernel_width;i++){
            ifft((*X_f)[c][i], (*tmp)[c][i], kernel_height);
        }
    }


    //first transpose
    for (int c = 0; c < input_channels; c++) {
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                X_f[0][c][j][i] = tmp[0][c][i][j];
            }
        }
    }

    //second fft
    for (int c=0;c<input_channels;c++){
        for(int i =0;i<kernel_width;i++){
            ifft((*X_f)[c][i], (*tmp)[c][i], kernel_height);
        }
    }


    //last transpose
    for (int c = 0; c < input_channels; c++) {
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                x_out[0][c][j][i] = tmp[0][c][i][j];
            }
        }
    }

    delete tmp;
}


