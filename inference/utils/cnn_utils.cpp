#include "cnn.h"

namespace ml {

int checkLayerMeta(CNN_layer_struct * lay, FILE * f)
{
	CNN_layer_struct tmp;
	if(fread(&(tmp.type),				sizeof(tmp.type),		1,f) == 0){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.input_channels),	sizeof(tmp.input_channels),1,f) ==0){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.output_size[0]),			sizeof(tmp.output_size[0]),	1,f) == 0 ){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.kernel_width),		sizeof(tmp.kernel_width),	1,f) == 0 ){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(fread(&(tmp.pad),				sizeof(tmp.pad),		1,f) == 0 ){
		printf("Meta Check failed!\n");
		return CNN_RETURN_FAILED;
	}
	// Check correctness
	if(!(tmp.type == lay->type))
		return CNN_RETURN_FAILED;
	if(lay->type == Layer_Type::Conv){
		if(((tmp.input_channels == lay->input_channels) &&
			(tmp.kernel_width == lay->kernel_width) &&
			(tmp.pad == lay->pad) &&
			(tmp.output_size[0] == lay->output_size[0])) == 0){
			return CNN_RETURN_FAILED;
		}
	}
	if(lay->type == Layer_Type::Linear){
		if(!(tmp.output_size[0] == tmp.output_size[0]))
			return CNN_RETURN_FAILED;
	}
	return CNN_RETURN_SUCCESS;
}


int readConvWeights(CNN_layer_struct * lay, FILE * f)
{
	int res;
	for(int i = 0; i < lay->output_size[0]; i++){
		Tensor * W = &(lay->W[i]);
		if(res = W->read(f) ){
			printf("Weight tensor read has failed code %d!\n", res);
			return CNN_RETURN_FAILED;
		}
	}
	Tensor * B = lay->B;
	if(res = B->read(f)){
		printf("Bias tensor read has failed code %d!\n", res);
		return CNN_RETURN_FAILED;
	}
	return CNN_RETURN_SUCCESS;
}



int readFCWeights(CNN_layer_struct * lay, uint32_t insize, FILE * f)
{
	int res;
	Tensor * W = lay->W;
	if(res = W->read(f)){
		printf("Linear Weight Tensor read failed code %d \n!",res);
		return CNN_RETURN_FAILED;
	}
	Tensor * B = lay->B;
	if(res = B->read(f)){
		printf("Linear Bias Tensor read failed code %d \n!",res);
		return CNN_RETURN_FAILED;
	}
	return CNN_RETURN_SUCCESS;
}






int CNN::read(const char * infile)
{
	FILE * f = fopen(infile,"rb");
	uint32_t nlayers;
	if(fread(&(nlayers),sizeof(nlayers),1,f) == 0){
		printf("Reading number of layers failed!\n");
		return CNN_RETURN_FAILED;
	}
	if(layers.size() != nlayers){
		printf("Layer sizes do not match reading weights failed!\n");
		return CNN_RETURN_FAILED;
	}
	uint32_t prev_size = 0;
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct * lay = &(layers[i]);
		if(checkLayerMeta(lay,f) == CNN_RETURN_FAILED){
			printf("Layer check failed!\n");
			return CNN_RETURN_FAILED;
		} 
		switch(lay->type){ 
			case Layer_Type::Pool:
			case Layer_Type::ReLU:
			case Layer_Type::Softmax:
				break;
			case Layer_Type::Conv:
				if(readConvWeights(lay,f) == CNN_RETURN_FAILED)
					return CNN_RETURN_FAILED;
				break;
			case Layer_Type::Linear:
				if(readFCWeights(lay,prev_size,f) == CNN_RETURN_FAILED)
					return CNN_RETURN_FAILED;
				break;
			default:
				printf("%d not a valid layer type!\n",(uint32_t) lay->type);
		}
		prev_size = lay->output_size[2] * lay->output_size[1] * lay->output_size[0];
	}
	fclose(f);
	return CNN_RETURN_SUCCESS;
}


void CNN::print_timing()
{
	const char * names[] = {"Linear","Pool","ReLU","Conv","Softmax"};
	printf("Execution Time [ms]:\n");
	printf("--------------------\n");
	for(int i = 0; i < 5; i++){
		printf("%s Layer: %lf \n",names[i],runtime[i]);
	}
	printf("--------------------\n");
}

/* ----------------------------- Functions to initialize Layers -----------------------------*/

CNN_layer_struct LinearLayer(uint32_t outputs)
{
	CNN_layer_struct linear;
	linear.type = Layer_Type::Linear;
	linear.output_size[0] = 1;
	linear.output_size[1] = 1;
	linear.output_size[2] = outputs;
	return linear;
}

CNN_layer_struct PoolLayer(uint32_t output_channels, uint32_t output_height, uint32_t output_width)
{
	CNN_layer_struct pool;
	pool.type = Layer_Type::Pool;
	pool.output_size[0] = output_channels;
	pool.output_size[1] = output_height;
	pool.output_size[2] = output_width;
	return pool;
}

CNN_layer_struct ConvLayer(uint32_t input_channels, uint32_t output_channels, uint32_t output_height, 
		uint32_t output_width, uint32_t kernel_width, uint32_t pad)
{
	CNN_layer_struct conv;
	conv.type = Layer_Type::Conv;
	conv.kernel_width = kernel_width;
	conv.pad = pad;
	conv.output_size[0] = output_channels;
	conv.output_size[1] = output_height;
	conv.output_size[2] = output_width;
	conv.input_channels = input_channels;
	return conv;
}


CNN_layer_struct ReLULayer(bool in_place )
{
	CNN_layer_struct relu;
	relu.type = Layer_Type::ReLU;
	relu.in_place = in_place;
	return relu;
}

CNN_layer_struct SoftmaxLayer()
{
	CNN_layer_struct lay;
	lay.type = Layer_Type::Softmax;
	lay.in_place = false;
	return lay;
}


}
