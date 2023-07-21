#include "cnn.h"

namespace ml {

CNN::CNN(std::vector<CNN_layer_struct> in_layers)
{
	uint32_t insize = 0;
	layers = in_layers;
	// The tricky part is allocating the proper tensors
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU: case Layer_Type::Softmax:
				lay.output_size[0] = layers[i - 1].output_size[0];
				lay.output_size[1] = layers[i - 1].output_size[1];
				lay.output_size[2] = layers[i - 1].output_size[2];
				if(lay.in_place)
					lay.Z = layers[i - 1].Z;
				else
					lay.Z = new Tensor(layers[i-1].output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::Pool:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::Conv:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				lay.W = new Tensor[lay.output_size[0]]();
				for(int i =0 ; i < lay.output_size[0]; i++){
					lay.W[i].allocate(lay.input_channels,lay.kernel_width,lay.kernel_width);
				}
				lay.B = new Tensor(1,1,lay.output_size[0]);
				break;
			case Layer_Type::Linear:
				insize = layers[i-1].output_size[0] * layers[i-1].output_size[1] * layers[i-1].output_size[2];
				lay.Z = new Tensor(1,1,lay.output_size[2]);
				lay.W = new Tensor(1,lay.output_size[2],insize);
				lay.B = new Tensor(1,1,lay.output_size[2]);
				break;
			default:
				throw std::runtime_error("Layer not implemented !\n");
		}
	}
}



CNN::~CNN()
{
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU:
				if(!(lay.in_place))
					delete lay.Z;
				break;
			case Layer_Type::Softmax:
			case Layer_Type::Pool:
				delete lay.Z;
				break;
			case Layer_Type::Conv:
				delete lay.Z;
				delete [] lay.W;
				delete lay.B;
				break;
			case Layer_Type::Linear:
				delete lay.Z;
				delete lay.W;
				delete lay.B;
				break;
			default:
				printf("Rogue unimplemented layer found during deallocation !\n");
		}
	}
}



/* Implement Inference here !*/
Tensor * CNN::inference(Tensor * input)
{
	Tensor * X = input;
    for(int i = 0; i < layers.size(); i++){
        CNN_layer_struct & lay = layers[i];

        switch (lay.type) {
            case Layer_Type::Linear:
            {
                auto start = mtick();
                Linear(X, lay.W, lay.B, lay.Z);
                double time = mtock(start);
                runtime[0] += time;
                //printf("Type: Linear\n");
                break;
            }
            case Layer_Type::Pool:
            {
                auto start = mtick();
                maxPool(X, lay.Z);
                double time = mtock(start);
                runtime[1] += time;
                //printf("Type: Pool\n");
                break;
            }
            case Layer_Type::ReLU:
            {
                auto start = mtick();
                ReLU(X, lay.Z);
                double time = mtock(start);
                runtime[2] += time;
                //printf("Type: ReLU\n");
                break;
            }
            case Layer_Type::Conv:
            {
                auto start = mtick();
                X = padTensor(X, lay.pad);
                Tensor * W_wino = winoWeights(lay.W, lay.Z->size[0]);
                convWinograd(X, W_wino, lay.B, lay.Z,  lay.W->size[2]);
                double time = mtock(start);
                runtime[3] += time;
                delete [] W_wino;
                break;
            }
            case Layer_Type::Softmax:
            {
                auto start = mtick();
                Softmax(X, lay.Z);
                double time = mtock(start);
                runtime[4] += time;
                //printf("Type: Softmax\n");
                break;
            }
            default:
                printf("Rogue unimplemented layer found during inference !\n");
        }
        X = lay.Z;
    }
    return X;
}



}




