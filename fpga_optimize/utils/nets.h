#ifndef NETS_H 
#define NETS_H
#include "cnn.h"
namespace ml{


extern std::vector<CNN_layer_struct> testNet;
extern std::vector<CNN_layer_struct> smallNet;
extern std::vector<CNN_layer_struct> mediumNet;
extern std::vector<CNN_layer_struct> largeNet;
extern std::vector<CNN_layer_struct> giantNet;

extern std::vector<CNN_layer_struct> VGG11;

extern std::vector<CNN_layer_struct> VGG16;

}
#endif
