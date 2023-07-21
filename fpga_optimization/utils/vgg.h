#include "cnn.h"

CNN_layer_struct vgg11[27] = {
		// in_channels, k_width , pad , out z, y x 
		// Block 1
		mlConv(3,3,1,64,224,224),
		mlReLU(),
		// width ,out z y x
		mlPool(2,64,112,112),
		// Block 2
		mlConv(64,3,1,128,112,112),
		mlReLU(),
		mlPool(2,128,56,56),
		// Block 3
		mlConv(128,3,1,256,56,56),
		mlReLU(),
		mlConv(256,3,1,256,56,56),
		mlReLU(),
		mlPool(2,256,28,28),
		// Block 4
		mlConv(256,3,1,512,28,28),
		mlReLU(),
		mlConv(512,3,1,512,28,28),
		mlReLU(),
		mlPool(2,512,14,14),
		// Block 5
		mlConv(512,3,1,512,14,14),
		mlReLU(),
		mlConv(512,3,1,512,14,14),
		mlReLU(),
		mlPool(2,512,7,7),
		// FC
		mlFC(4096),
		mlReLU(),
		mlFC(4096),
		mlReLU(),
		mlFC(1000),
		mlSoftmax()
};


CNN_layer_struct vgg16[37] = {
		// in_channels, k_width , pad , out z, y x 
		// Block 1
		mlConv(3,3,1,64,224,224),
		mlReLU(),
		mlConv(64,3,1,64,224,224),
		mlReLU(),
		// width ,out z y x
		mlPool(2,64,112,112),
		// Block 2
		mlConv(64,3,1,128,112,112),
		mlReLU(),
		mlConv(128,3,1,128,112,112),
		mlReLU(),
		mlPool(2,128,56,56),
		// Block 3
		mlConv(128,3,1,256,56,56),
		mlReLU(),
		mlConv(256,3,1,256,56,56),
		mlReLU(),
		mlConv(256,3,1,256,56,56),
		mlReLU(),
		mlPool(2,256,28,28),
		// Block 4
		mlConv(256,3,1,512,28,28),
		mlReLU(),
		mlConv(512,3,1,512,28,28),
		mlReLU(),
		mlConv(512,3,1,512,28,28),
		mlReLU(),
		mlPool(2,512,14,14),
		// Block 5
		mlConv(512,3,1,512,14,14),
		mlReLU(),
		mlConv(512,3,1,512,14,14),
		mlReLU(),
		mlConv(512,3,1,512,14,14),
		mlReLU(),
		mlPool(2,512,7,7),
		// FC
		mlFC(4096),
		mlReLU(),
		mlFC(4096),
		mlReLU(),
		mlFC(1000),
		mlSoftmax()
};



CNN_layer_struct cnn_test[5] = {
		mlConv(1,5,2,2,8,8),
		mlReLU(),
		mlPool(2,2,4,4),
		mlFC(4),
		mlSoftmax()
};
