#ifndef CNN_CONV
#define CNN_CONV 0

#include "math.h"
#include "tensor.h"
#include <complex>
#include "fft.h"
#include "winograd.h"


// Assume Square tiles !
typedef struct{
	int tile_size; // Size of tile including overlap
	int overlap; // Size of overlap = w.x - 1
	int tile_stride; // Size of tile without overlap
}FFT_STRUCT;






/* Your Basic Convolution */
void convBasic(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z);

/* 
 * Pre Transform the Weights
 * Tensor 			*W		: Untransformed Weight Tensor
 * int		output_channels	: Number of output channels
 * Return:		Tensor *	: New Tensor containing transformed Weights
 */
Tensor *winoWeights(Tensor * W, int output_channels);

/*
 * Convolution using inputs and converted Weights
 * Tensor 			*U_wino	: Transformed Weight Tensor
 * Tensor			*B		: Bias 
 * Tensor			*Z		: Output Tensor
 * int 			k_size		: Width and Height of weight kernel
 */
void convWinograd(Tensor * X, Tensor * U_wino , Tensor * B, Tensor * Z, int k_size);

/*
 * Pre Transform the weights depending on the tile size
 * Tensor 			*W		: Untransformed Weight Tensor
 * int		output_channels	: Number of output channels
 * Return:		C_Tensor *	: New Tensor containing transformed Weights
 */
C_Tensor * fftWeights(Tensor * W, int output_channels);

/*
 * Convolution using inputs and converted Weights
 * C_Tensor			*U_fft:	: Complex Transformed Weight Tensor
 * Tensor			*B		: Bias 
 * Tensor			*Z		: Output Tensor
 * int 			k_size		: Width and Height of weight kernel
 */
void convFFT(Tensor * X, C_Tensor * U_fft, Tensor * B, Tensor * Z, int k_size);


#endif
