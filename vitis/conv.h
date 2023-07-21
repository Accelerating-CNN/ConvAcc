#define input_channel 1
#define input_width 134
#define input_height 134

#define z_channel 1
#define z_width 128
#define z_height 128

#define number_of_filter 1
#define filter_channel 1
#define filter_width 7
#define filter_height 7

#define bias_channel 1
#define bias_width 1
#define bias_height 1


#define input_size  (input_channel * input_width * input_height)
#define z_size  (z_channel * z_width * z_height)
#define filter_size  (number_of_filter * filter_channel * filter_width * filter_height)
#define bias_size  (bias_channel * bias_width * bias_height)
#include "hls_stream.h"

 void EntryConv(float z_out[z_size], float x_in[input_size], float  w_in[filter_size]);

