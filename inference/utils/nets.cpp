#include "nets.h"

namespace ml{
std::vector<CNN_layer_struct> VGG11= {
	// Block 1
	ConvLayer(3,64,224,224,3,1),
	ReLULayer(),
	PoolLayer(64,112,112),
	// Block 2
	ConvLayer(64,128,112,112,3,1),
	ReLULayer(),
	PoolLayer(128,56,56),
	// Block 3
	ConvLayer(128,256,56,56,3,1),
	ReLULayer(),
	ConvLayer(256,256,56,56,3,1),
	ReLULayer(),
	PoolLayer(256,28,28),
	// Block 4
	ConvLayer(256,512,28,28,3,1),
	ReLULayer(),
	ConvLayer(512,512,28,28,3,1),
	ReLULayer(),
	PoolLayer(512,14,14),
	// Block 5
	ConvLayer(512,512,14,14,3,1),
	ReLULayer(),
	ConvLayer(512,512,14,14,3,1),
	ReLULayer(),
	PoolLayer(512,7,7),
	// Linear Layers
	LinearLayer(4096),
	ReLULayer(),
	LinearLayer(4096),
	ReLULayer(),
	LinearLayer(1000),
	SoftmaxLayer()
};


std::vector<CNN_layer_struct> VGG16= {
	// Block 1
	ConvLayer(3,64,224,224,3,1),
	ReLULayer(),
	ConvLayer(64,64,224,224,3,1),
	ReLULayer(),
	PoolLayer(64,112,112),
	// Block 2
	ConvLayer(64,128,112,112,3,1),
	ReLULayer(),
	ConvLayer(128,128,112,112,3,1),
	ReLULayer(),
	PoolLayer(128,56,56),
	// Block 3
	ConvLayer(128,256,56,56,3,1),
	ReLULayer(),
	ConvLayer(256,256,56,56,3,1),
	ReLULayer(),
	ConvLayer(256,256,56,56,3,1),
	ReLULayer(),
	PoolLayer(256,28,28),
	// Block 4
	ConvLayer(256,512,28,28,3,1),
	ReLULayer(),
	ConvLayer(512,512,28,28,3,1),
	ReLULayer(),
	ConvLayer(512,512,28,28,3,1),
	ReLULayer(),
	PoolLayer(512,14,14),
	// Block 5
	ConvLayer(512,512,14,14,3,1),
	ReLULayer(),
	ConvLayer(512,512,14,14,3,1),
	ReLULayer(),
	ConvLayer(512,512,14,14,3,1),
	ReLULayer(),
	PoolLayer(512,7,7),
	// Linear Layers
	LinearLayer(4096),
	ReLULayer(),
	LinearLayer(4096),
	ReLULayer(),
	LinearLayer(1000),
	SoftmaxLayer()
};


std::vector<CNN_layer_struct> testNet= {
	ConvLayer(1,2,8,8,5,2),
	ReLULayer(),
	PoolLayer(2,4,4),
	LinearLayer(4),
	SoftmaxLayer()
};

std::vector<CNN_layer_struct> smallNet= {
	ConvLayer(3,128,128,128,7,3),
	ReLULayer(),
	PoolLayer(128,64,64),
	// Block 2
	ConvLayer(128,256,64,64,5,2),
	ReLULayer(),
	PoolLayer(256,32,32),
	// Block 3
	ConvLayer(256,256,32,32,3,1),
	ReLULayer(),
	PoolLayer(256,16,16),
	// Block 4
	ConvLayer(256,512,16,16,3,1),
	ReLULayer(),
	PoolLayer(512,8,8),
	// Block 5
	ConvLayer(512,512,6,6,3,0),
	ReLULayer(),
	PoolLayer(512,3,3),
	// Classifier
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(512),
	ReLULayer(),
	LinearLayer(100),
	SoftmaxLayer(),
};


std::vector<CNN_layer_struct> mediumNet= {
	ConvLayer(3,96,128,128,7,3),
	ReLULayer(),
	PoolLayer(96,64,64),
	// Block 2
	ConvLayer(96,256,64,64,5,2),
	ReLULayer(),
	PoolLayer(256,32,32),
	// Block 3
	ConvLayer(256,384,32,32,3,1),
	ReLULayer(),
	PoolLayer(384,16,16),
	// Block 4
	ConvLayer(384,384,14,14,3,0),
	ReLULayer(),
	// Block 5
	ConvLayer(384,256,12,12,3,0),
	ReLULayer(),
	PoolLayer(256,6,6),
	// Classifier
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(100),
	SoftmaxLayer(),
};

std::vector<CNN_layer_struct> largeNet= {
	ConvLayer(3,64,128,128,3,1),
	ReLULayer(),
	PoolLayer(64,64,64),
	// Block 2
	ConvLayer(64,128,64,64,3,1),
	ReLULayer(),
	PoolLayer(128,32,32),
	// Block 3
	ConvLayer(128,256,32,32,3,1),
	ReLULayer(),
	ConvLayer(256,256,32,32,3,1),
	ReLULayer(),
	PoolLayer(256,16,16),
	// Block 4
	ConvLayer(256,512,16,16,3,1),
	ReLULayer(),
	ConvLayer(512,512,16,16,3,1),
	ReLULayer(),
	PoolLayer(512,8,8),
	// Block 5
	ConvLayer(512,512,8,8,3,1),
	ReLULayer(),
	ConvLayer(512,512,8,8,3,1),
	ReLULayer(),
	PoolLayer(512,4,4),
	// Classifier
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(100),
	SoftmaxLayer(),
};
std::vector<CNN_layer_struct> giantNet= {
	ConvLayer(3,64,128,128,3,1),
	ReLULayer(),
	PoolLayer(64,64,64),
	// Block 2
	ConvLayer(64,128,64,64,3,1),
	ReLULayer(),
	PoolLayer(128,32,32),
	// Block 3
	ConvLayer(128,256,32,32,3,1),
	ReLULayer(),
	ConvLayer(256,256,32,32,3,1),
	ReLULayer(),
	ConvLayer(256,256,32,32,3,1),
	ReLULayer(),
	ConvLayer(256,256,32,32,3,1),
	ReLULayer(),
	PoolLayer(256,16,16),
	// Block 4
	ConvLayer(256,512,16,16,3,1),
	ReLULayer(),
	ConvLayer(512,512,16,16,3,1),
	ReLULayer(),
	ConvLayer(512,512,16,16,3,1),
	ReLULayer(),
	ConvLayer(512,512,16,16,3,1),
	ReLULayer(),
	PoolLayer(512,8,8),
	// Block 5
	ConvLayer(512,512,8,8,3,1),
	ReLULayer(),
	ConvLayer(512,512,8,8,3,1),
	ReLULayer(),
	ConvLayer(512,512,8,8,3,1),
	ReLULayer(),
	ConvLayer(512,512,8,8,3,1),
	ReLULayer(),
	PoolLayer(512,4,4),
	// Classifier
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(1024),
	ReLULayer(),
	LinearLayer(100),
	SoftmaxLayer(),
};

}
