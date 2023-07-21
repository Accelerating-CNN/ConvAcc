#include "cnn.h"
#include "nets.h"
#include "tensor.h"
#include "inf_utils.h"
#include <string>

using namespace ml;

void benchNet(CNN & cnn, const char * images[],
		const int ref[], int N_in);


/* Insert the weight file name here! */
const char * netfiles[] = 
{
	"weights/smallnet_weights.dat",
	"weights/mediumnet_weights.dat",
	"weights/largenet_weights.dat",
	"weights/giantnet_weights.dat"
};

const std::vector<CNN_layer_struct> * nets[] = 
{
	&smallNet,
	&mediumNet,
	&largeNet,
	&giantNet
};


int main(int argc, char * argv[])
{
	//printf("Entering main\n");	     
	if(argc < 3){
		printf("Usage: ./class net_select N\n");
		return 1;
	}
	int select = atoi(argv[1]);
	int N = atoi(argv[2]);
	CNN cnn(*nets[select]);
	cnn.read(netfiles[select]);
	printf("Using %s and %d images.\n",netfiles[select],N);
	benchNet(cnn,input_imgs,imgs_class,N);
	return 0;
}



void benchNet(CNN & cnn, const char * images[],
		const int ref[], int N_in)
{
	int N = N_in;
	if(N > IMAGES_MAX_TEST){
		printf("%d above amount of available images!\n", N);
		N = IMAGES_MAX_TEST;
	}
	std::vector<Tensor *> X(N);
	std::vector<int> pred(N);
	
	for(int i =0; i < N; i++){
		//printf("%s\n",images[i]);
		X[i] = readBMP(images[i]);
	}
	double total_time = 0;
	for(int i =0; i < N; i++){
		printf("Image: %s\n",images[i]);
		auto start = mtick();
		Tensor * Z = cnn.inference(X[i]);
		total_time += mtock(start);
		pred[i] = classImage(Z);
		printf("Actual Class: %s\n",classes_img100[ref[i]]);
	}
	printf("Total Time [ms]: %lf\n",total_time);
	printf("Frames/s : %lf\n",(N*1000)/total_time);
	cnn.print_timing();
	/* Correctness */
	int corr = 0;
	for(int i =0; i < N; i++){
		if(ref[i] == pred[i])
			corr++;
	}
	printf("Accuracy: %d of %d images (%f %% )!\n",corr,N,(corr/(N*1.0f)));
	for(int i = 0; i < N; i++){
		delete X[i];
	}
}





