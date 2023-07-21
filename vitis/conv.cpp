#include "conv.h"
#include <iostream>
#include "hls_stream.h"
#include <ap_fixed.h>

using namespace std;
typedef ap_fixed<24, 2> my_float;


inline my_float single_operation(my_float window[filter_height][filter_width],my_float weight[filter_height][filter_width])
{

        my_float sum = 0.0;
        int count = 0;
        for (int i = 0; i <filter_height; i++){
          for (int j = 0; j <filter_width; j++){
         sum += window[i][j]  * weight[i][j];
        }
        }
         return sum;
}





void conv2D( float out_stream[z_size], float in_stream[input_size], float weight_stream[filter_size]){
  int line_buffer_size = ((filter_height-1) * input_width) +filter_width;
  my_float line_buffer[((filter_height-1) * input_width) +filter_width];
  my_float weight[filter_height][filter_width];
  my_float window[filter_height][filter_width];
  int read_count = 0;
  int ab = 0;
  int output_c = 0;
#pragma HLS ARRAY_PARTITION variable=line_buffer complete
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight complete dim=0
  for (int i = 0 ; i<line_buffer_size ; i ++){
	  line_buffer[i] = in_stream[read_count];
        read_count  +=1  ;
  }
  for (int y = 0; y < filter_height; y++){
           for (int x = 0; x < filter_width; x++){
                   weight[y][x] = weight_stream[ab];
  	  	  	  	   ab +=1;
           }
  }
  for(int i = 0 ; i<filter_height ; i++ ){
        for(int j = 0 ;j<filter_width; j++){
                if(i==(filter_height-1)){
                        window[i][j] = line_buffer[i*input_width+j];
                }else{
                        window[i][j] = line_buffer[i*input_width+j];
                }
        }
 }

for (int y = 0; y<input_height;y++){
        for(int x = 0;x<input_width;x++){
#pragma HLS PIPELINE
        if (x<(input_width-filter_height+1)){
         float v = 0.0;
     my_float utku = single_operation(window,weight);
     v = static_cast<float>(utku);

     if(output_c<z_size){
     out_stream[output_c] = v;
     output_c +=1;
     }
        }
        for (int i = 0; i<line_buffer_size;i++){
                if (i+1 != line_buffer_size){
                line_buffer[i] = line_buffer[i+1];
                }
        }

 if(read_count<input_width * input_height){
        line_buffer[line_buffer_size-1] = in_stream[read_count];
 }
 read_count+=1;
        for (int i = 0; i<filter_height;i++){
            for (int j = 0; j<filter_width-1;j++){
                window[i][j]= window[i][j+1];
                }
        }

        for (int i = 0 ; i < filter_height ; i++){
           window[i][(filter_width-1)] = line_buffer[i*input_width+(filter_width-1)];
        }

}


}

}
void EntryConv(float z_out[z_size], float x_in[input_size], float  w_in[filter_size])
{

   #pragma HLS interface m_axi port=x_in depth=65536
   #pragma HLS interface m_axi port=w_in depth=49
   #pragma HLS interface m_axi port=z_out depth=65536
   #pragma HLS interface s_axilite port=x_in
   #pragma HLS interface s_axilite port=w_in
   #pragma HLS interface s_axilite port=z_out
   #pragma HLS INTERFACE s_axilite port=return

 conv2D(z_out,x_in,w_in);

}

