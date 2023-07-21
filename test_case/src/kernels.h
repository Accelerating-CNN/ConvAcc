#ifndef CNN_KERNELS_H
#define CNN_KERNELS_H 0

#define input_channel 1
#define input_width 256
#define input_height 256

#define z_channel 1
#define z_width 250
#define z_height 250

#define number_of_filter 1
#define filter_channel 1
#define filter_width 7
#define filter_height 7

#define bias_channel 1
#define bias_width 1
#define bias_height 1

#define WIN_SIZE 3 // must be odd
#define HALF_SIZE (((WIN_SIZE) - 1) / 2)

#define WIDTH 8
#define HEIGHT 8



#define input_size  (input_channel * input_width * input_height)
#define z_size  (z_channel * z_width * z_height)
#define filter_size  (number_of_filter * filter_channel * filter_width * filter_height)
#define bias_size  (bias_channel * bias_width * bias_height)




#include "tensor.h"
#include <cmath>

/* 
 * Applies a 2d convolution on a 3D X using W: Z= W (conv) X + b
 * Tensor * X:		Input Tensor
 * Tensor * W:		Array of N weight Tensors (N == Z.size[0]) 
 * Tensor * Z:		Output Tensor 
 * Tensor * b:		Bias 
 */
//void conv2d(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z);
void conv2d(const float x_in[input_size] ,const float w_in[filter_size],  const float b_in[bias_size] , float z_out[z_size]);

/*
 * Applies a max pool layer on X (stride = size)
 * Tensor * X:	input Tensor
 * Tensor * Z:	output Tensor
 * int size:	size of max pool kernel
 */
void maxPool(Tensor * X, Tensor * Z);

/*
 * Applies a Linear layer: z = Wx + b 
 * Flatten the input if required 
 * Tensor *	X: input Tensor
 * Tensor *	W: weight Matrix (in Tensor form)
 * Tensor *	B: bias array (in Tensor form)
 * Tensor *	Z: output array (in Tensor form)
 */
void Linear(Tensor * X, Tensor * W, Tensor * b, Tensor * Z);

/*
 * Applies the ReLU activation function: Z = ReLU(X)
 * Tensor * X: input Tensor
 * Tensor * Z: output Tensor
 */
void ReLU(Tensor * X , Tensor * Z);

/*
 * Applies the Softmax activation function z = exp(x_i)/sum(exp(x_j))
 * This is a stable Softmax implementation
 * Tensor * X: input vector in Tensor form
 * Tensor * Z: output vector in Tensor form
 */
void Softmax(Tensor * X, Tensor * Z);

#endif

