//#include "kernels.h"
//#include "tensor.h"
//#include <vector>
//#include "cnn.h"
//#include "nets.h"
//#include <string>
//#include "imagenet_classes.h"
//#include <algorithm>
//
//using namespace ml;
//
//
//void test_net(const char * weight_file , const char * data_file, std::vector<CNN_layer_struct> net);
//void predict_image(const char * weight_file , const char * data_file, std::vector<CNN_layer_struct> net,
//		const char * imagefile);
//void bench_net(const char * weight_file , std::vector<CNN_layer_struct> net, int N);
//
//int main(int argc, char * argv[])
//{
//	if(argc < 2){
//		printf("Usage: $./lab2.bin select\n");
//		return 1;
//	}
//	int select = atoi(argv[1]);
//	if(select == 0){
//		printf("Running test!\n");
//		test_net("data/testnet_weights.dat","data/testnet_test.dat",testNet);
//		test_net("data/smallnet_weights.dat","data/smallnet_test.dat",smallNet);
//		test_net("data/mediumnet_weights.dat","data/mediumnet_test.dat",mediumNet);
//		test_net("data/largenet_weights.dat","data/largenet_test.dat",largeNet);
//		test_net("data/giantnet_weights.dat","data/giantnet_test.dat",giantNet);
//	}
//	else if(select == 1){
//		if(argc < 3){
//			printf("Usage $./lab2.bin 1 image_tensor\n");
//			return 1;
//		}
//		printf("Image Classification\n");
//		predict_image("data/vgg16_weights.dat","data/vgg16_test.dat",VGG16,argv[2]);
//	}
//	else{
//		printf("Running Benchmarks\n");
//		bench_net("data/smallnet_weights.dat",smallNet,4);
//		bench_net("data/mediumnet_weights.dat",mediumNet,4);
//		bench_net("data/largenet_weights.dat",largeNet,4);
//		bench_net("data/giantnet_weights.dat",giantNet,4);
//	}
//	return 0;
//}
//
//
//void bench_net(const char * weight_file , std::vector<CNN_layer_struct> net, int N)
//{
//	CNN dut(net);
//	if(dut.read(weight_file) == CNN_RETURN_FAILED){
//		printf("Reading Weights failed!\n");
//		return;
//	}
//	Tensor * X = new Tensor[N];
//	for(int i =0 ; i  < N ; i++){
//		X[i].allocate(3,128,128);
//	}
//	printf("Benchmark: %s , N: %d\n",weight_file,N);
//	auto start = mtick();
//	for(int i =0 ; i < N ; i++){
//		Tensor * Z = dut.inference(&(X[i]));
//	}
//	double extime = mtock(start);
//	printf("Total Time [ms]: %lf\n",extime);
//	dut.print_timing();
//	delete [] X;
//}
//
//
//
//void test_net(const char * weight_file , const char * data_file, std::vector<CNN_layer_struct> net)
//{
//	CNN dut(net);
//	if(dut.read(weight_file) == CNN_RETURN_FAILED){
//		printf("Reading Weights failed!\n");
//		return;
//	}
//	FILE *f;
//	if((f = fopen(data_file,"rb")) == NULL){
//		printf("Reading data file failed!\n");
//		return;
//	}
//	uint32_t ntests;
//	if(fread(&ntests,sizeof(ntests),1,f) == 0){
//		printf("Reading tests failed!\n");
//		return;
//	}
//	Tensor X,R;
//	double total_time = 0;
//	printf("Weights: %s\n", weight_file);
//	printf("Data: %s\n", data_file);
//	printf("N: %d\n", ntests);
//	for(int i = 0; i < ntests ; i++){
//		X.read(f);
//		R.read(f);
//		auto start = mtick();
//		Tensor * Z = dut.inference(&X);
//		total_time += mtock(start);
//		if(Z->size[2] != R.size[2]){
//			printf("Test failed Output Tensor has the wrong Dimensions! \n");
//			return;
//		}
//		compareTensors(&R,Z,1,0.01);
//	}
//	printf("Total time[ms]: %lf\n",total_time);
//	dut.print_timing();
//	fclose(f);
//}
//
//void predict_image(const char * weight_file , const char * data_file, std::vector<CNN_layer_struct> net,
//		const char * imagefile)
//{
//	CNN dut(net);
//	if(dut.read(weight_file) == CNN_RETURN_FAILED){
//		printf("Reading Weights failed!\n");
//		return;
//	}
//	Tensor X;
//	FILE * f;
//	if((f = fopen(imagefile,"rb")) == NULL){
//			printf("Reading imagefile failed!\n");
//			return;
//	}
//	X.read(f);
//	fclose(f);
//	printf("Starting inference !\n");
//	auto start = mtick();
//	Tensor * Z = dut.inference(&X);
//	double time = mtock(start);
//	printf("Total time[ms]: %lf\n",time);
//	dut.print_timing();
//	int maxel = std::distance((*Z)[0][0],std::max_element((*Z)[0][0],(*Z)[0][0] + 1000));
//	printf("Predicted class with %f : %s \n",(*Z)[0][0][maxel],imagenet_classes[maxel]);
//}
//
