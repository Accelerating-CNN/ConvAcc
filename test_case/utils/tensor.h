#ifndef TENSOR_H
#define TENSOR_H


#include "stdint.h"
#include <cstdio>
#include <stdio.h>
#include <stdexcept>
#include "time.h"

#define FLOAT float


#define TENSOR_READ_SUCCSEFULL	0
#define TENSOR_READ_RESIZED		1
#define TENSOR_READ_FAILED 		2


class Tensor{
	private:
		void allocate(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x);
		int allocated;

	public:
		FLOAT *** data;
		uint32_t size[3];
		// Constructor and Destructor
		Tensor(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x);
		// Creates an empty un- allocated Tensor
		Tensor();
		// Destructor
		~Tensor();
		// Read and write Tensor to file
		int read(FILE * f);
		void write(FILE * f);
		void resize(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x);
		// Fill Tensor with random values between start and stop
		void randomize(FLOAT start, FLOAT stop);
		// Easy access to the data
		FLOAT ** operator[](uint32_t i){
			return data[i];
		}
};

// Compare two Tensor arrays of length N 
int compareTensors(Tensor * y, Tensor * ref, int N, float limit);


struct timespec mtick();
double mtock(struct timespec start);


#endif
