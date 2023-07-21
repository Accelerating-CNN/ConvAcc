#ifndef CNN_KERNELS_H
#define CNN_KERNELS_H 0

#include "tensor.h"
#include <cmath>

/* 
 * Applies a 2d convolution on a 3D X using W: Z= W (conv) X + b
 * Tensor * X:		Input Tensor
 * Tensor * W:		Array of N weight Tensors (N == Z.size[0]) 
 * Tensor * Z:		Output Tensor 
 * Tensor * b:		Bias 
 */
void conv2d(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z);

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

