#include "tensor.h"

void Tensor::resize(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	if(data != NULL){
		delete [] data[0][0];
		delete [] data[0];
		delete [] data;
	}
	allocate(dim_z,dim_y,dim_x);
}

void Tensor::allocate(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	allocated = 1;
	data = new FLOAT ** [dim_z];
	FLOAT ** tmp_y = new FLOAT * [(uint64_t) dim_y *
		(uint64_t) dim_z];
	FLOAT * tmp_x = new FLOAT[(uint64_t) dim_z * 
		(uint64_t) dim_y * (uint64_t) dim_x]();
	for(int i = 0; i < dim_y*dim_z ; i++){
		tmp_y[i] = &(tmp_x[i * dim_x]);
	}
	for(int i = 0; i < dim_z ; i++){
		data[i] = &(tmp_y[i * dim_y]);
	}
	size[0] = dim_z;
	size[1] = dim_y;
	size[2] = dim_x;
}


static float randomf(float start, float stop)
{
	float range = stop - start;
	float val = ((std::rand() % ((int)(range * 10)))/10.0f) 
			+ start;
	return val;
}

void Tensor::randomize(float start, float stop)
{
	int i,j,k,n;
	for(i = 0; i < size[0]; i++){
		for(j = 0; j< size[1] ; j++){
			for(k = 0; k< size[2] ; k++){
				data[i][j][k] = randomf(start,stop);
			}
		}
	}
}



Tensor::Tensor(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	allocate(dim_z,dim_y,dim_x);
}

Tensor::Tensor()
{
	allocated = 0;
	data = NULL;
	size[0] = 0;
	size[1] = 0;
	size[2] = 0;
}




Tensor::~Tensor()
{
	if(data != NULL){
		if(data[0] != NULL){
			if((data[0][0] != NULL) && (allocated))
				delete [] data[0][0];
			delete [] data[0];
		}
		delete [] data;
	}
}


int Tensor::read(FILE * f)
{
	uint32_t dims[3];
	for(int i = 0; i < 3 ;i++){
		if( 1 != fread(&(dims[i]),sizeof(dims[0]),1,f))
			return TENSOR_READ_FAILED;
	}

	int retval = TENSOR_READ_SUCCSEFULL;
	if(dims[0] != size[0] || (dims[1] != size[1]) || (dims[2] != size[2])){
		retval = TENSOR_READ_RESIZED;
		resize(dims[0],dims[1],dims[2]);
	}
	uint32_t num_params = dims[0] * dims[1] * dims[2];
	if(num_params != fread(&(data[0][0][0]),sizeof(data[0][0][0]),num_params,f))
		return TENSOR_READ_FAILED;
	return retval;
}

void Tensor::write(FILE *f)
{
	uint64_t num_params = size[0]*size[1]*size[2];
	fwrite(&(size[0]),sizeof(size[0]),1,f);
	fwrite(&(size[1]),sizeof(size[1]),1,f);
	fwrite(&(size[2]),sizeof(size[2]),1,f);
	fwrite(data[0][0],sizeof(data[0][0][0]),num_params,f);
}


static FLOAT Fabs(FLOAT a)
{
	return (a > 0) ? a : -a;
}


int compareTensors(Tensor * y, Tensor * ref, int N, float limit){
	int n,i,j,k;
	for(n = 0; n < N; n++){
		for(i = 0; i < y[n].size[0]; i++){
			if(y[n].size[0] != ref[n].size[0] || (y[n].size[1] != ref[n].size[1]) 
					|| (y[n].size[2] != ref[n].size[2])){
				throw std::runtime_error("Tensor dimensions don't match ! \n");
			}
			FLOAT ** my = y[n][i];
			FLOAT ** mr = ref[n][i];
			for(int j =0; j < y[n].size[1]; j++){
				for(int k =0; k < y[n].size[2];k++){
					FLOAT diff = Fabs( my[j][k] -  mr[j][k]);
					if(diff > limit){
						printf("Tensors differ at: [%d][%d][%d] by %f \n",
								i,j,k,diff);
						return 1; 
					}
				}
			}
		}
	}
	printf("Tensors are equal!\n");
	return 0;
}



Tensor * padTensor(Tensor * X , uint32_t pad)
{
	int N = X->size[1] +  pad *2;
	int K = X->size[2] +  pad *2;
	Tensor * Xpad = new Tensor(X->size[0], N,K);
	for(int z = 0; z < X->size[0]; z++){
		FLOAT ** xpad = (*Xpad)[z];
		FLOAT ** x = (*X)[z];
		for(int i = 0; i < N ;i++){
			for(int j = 0; j < K; j++){
				if((i >= pad) && (i < (N-pad)) &&
							(j >=  pad) && (j <  (K- pad)))
					xpad[i][j] = x[i-pad][j-pad];
			}
		}
	}
	return Xpad;
}


struct timeval mtick(){
	struct timeval start;
	gettimeofday(&start,0);
	return start;
}

double mtock(struct timeval start){
	struct timeval stop;
	gettimeofday(&stop,0);
	double time = (stop.tv_sec - start.tv_sec)*1000 + 
		(stop.tv_usec - start.tv_usec)/(1000.f);
	return time;
}





