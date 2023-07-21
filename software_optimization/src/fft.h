#ifndef FFT_H
#define FFT_H 0
#include <complex>
#include <utility>
#include <cstring>
#include "stdint.h"

#define C_FLOAT  std::complex<float>
/* Dummed down Version of the Tensor class for Complex data-type
 */
class C_Tensor 
{
	public:
	void allocate(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x);
	C_FLOAT *** data;
	uint32_t size[3];
	C_Tensor(){data = NULL;};
	C_Tensor(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x);
	~C_Tensor();
	C_FLOAT ** operator[](uint32_t i){
		return data[i];
	}
	C_FLOAT ** operator[](int i){
		return data[i];
	}
};

void fft(C_FLOAT * x_in, C_FLOAT * X_out, int N);
void ifft(C_FLOAT * x_in, C_FLOAT * X_out, int N);

void fft2d(C_Tensor * x_in, C_Tensor * X_out, int output_channels);
void ifft2d(C_Tensor * X_f , C_Tensor * x_out, int output_channels);


#endif
