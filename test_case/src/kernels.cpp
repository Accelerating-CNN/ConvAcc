#include "kernels.h"
#include "iostream"
#include "../utils/tensor.h"
extern "C" {
#include "pynq_api.h"
}
using namespace std;

FLOAT* FLATTEN_INDEX(Tensor * X){
    int x_channel = X->size[0];
    int x_height = X->size[1];
    int x_width = X->size[2];
    FLOAT* flatten_x = new FLOAT[x_channel*x_height*x_width];
    int flatten_index =0;
    for(int i = 0;i<x_channel;i++){
        for(int j=0;j<x_height;j++){
            for(int k=0;k<x_width;k++){
                flatten_x[flatten_index] = X[0][i][j][k];
                flatten_index++;
            }
        }
    }
    return  flatten_x;
}

FLOAT** FlOAT_CHANNEL(Tensor * X){
    int x_channel = X->size[0];
    int x_height = X->size[1];
    int x_width = X->size[2];

    FLOAT** flatten_x = new FLOAT*[x_channel];
    for(int i = 0;i<x_channel;i++){
        int flatten_index =0;
        FLOAT* per_flatten_x = new FLOAT[x_height*x_width];
        for(int j=0;j<x_height;j++){
            for(int k=0;k<x_width;k++){
                per_flatten_x[flatten_index] = X[0][i][j][k];
                flatten_index++;
            }
        }
        flatten_x[i] = per_flatten_x;
    }
    return  flatten_x;
}

/*
 * Applies a 2d convolution on a 3D X using W: Z= W (conv) X + b
 * Tensor * X:		Input Tensor
 * Tensor * W:		Array of N weight Tensors (N == Z.size[0])
 * Tensor * Z:		Output Tensor
 * Tensor * b:		Bias
 */

void conv2d(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z)
{
    int x_channel = X->size[0];
    int x_height = X->size[1];
    int x_width = X->size[2];

    int w_channel = W->size[0];
    int w_height = W->size[1];
    int w_width = W->size[2];

    int z_channel = Z->size[0];
    int z_height = Z->size[1];
    int z_width = Z->size[2];

    for(int i=0;i<z_channel;i++) {
        for(int j=0;j<z_height;j++) {
            for(int k=0;k<z_width;k++) {
                float sum = 0;c
                for (int c = 0; c < x_channel; c++) {
                    for (int p = 0; p < w_height; p++) {
                        for (int q = 0; q < w_width; q++) {
                            sum += X[0][c][j+p][k+q] * W[i][c][p][q];
                        }
                    }
                }
                sum += b[0][0][0][i];
                Z[0][i][j][k]=sum;
            }
        }
    }

}

/*
 * Applies a max pool layer on X (size = stride = 2)
 * Tensor * X:	input Tensor
 * Tensor * Z:	output Tensor
 */
/*
void maxPool(Tensor * X, Tensor * Z)
{
    int channel_size = X->size[0];
    int height_size = X->size[1];
    int width_size = X->size[2];


    int stride = 2;
    int kernel_size = 2;
    int padding = 0;
    int dilation = 1;

    int output_height = (height_size + 2*padding - dilation*(kernel_size-1)-1)/stride + 1;
    int output_width = (width_size + 2*padding - dilation*(kernel_size-1)-1)/stride + 1;

    // Max pooling
    for (int i = 0; i < channel_size ; ++i) {
        for (int j = 0; j <output_height ; ++j) {
            for (int k = 0; k < output_width ; ++k) {
                FLOAT max = 0.0f;
                for (int l = 0; l < kernel_size ; ++l) {
                    for (int m = 0; m < kernel_size ; ++m) {
                        int row = stride*j + l;
                        int col = stride*k + m;
                        if (row < height_size && col < width_size){
                            FLOAT element = X[0][i][row][col];
                            if (element > max){
                                max = element;
                            }
                        }
                    }
                }
                Z[0][i][j][k] = max;

            }
        }
    }
}
*/


/*
 * Applies a Linear layer: z = Wx + b 
 * Flatten the input if required 
 * Tensor *	X: input Tensor
 * Tensor *	W: weight Matrix (in Tensor form)
 * Tensor *	B: bias array (in Tensor form)
 * Tensor *	Z: output array (in Tensor form)
 */
/*
void Linear(Tensor * X, Tensor * W, Tensor * B, Tensor * Z)
{

    int x_channel = X->size[0];
    int x_height = X->size[1];
    int x_width = X->size[2];

    int w_channel = W->size[0];
    int w_height = W->size[1];
    int w_width = W->size[2];

    int b_channel = B->size[0];
    int b_height = B->size[1];
    int b_width = B->size[2];

    int z_channel = Z->size[0];
    int z_height = Z->size[1];
    int z_width = Z->size[2];

    //FLATTEN ALL THE INPUTS IN ONE VECTOR
    FLOAT* flatten_X = FLATTEN_INDEX(X);

    // FIRST GO TO THE ROW
    for(int i = 0; i<w_height ; i++){
        FLOAT sum = 0.0f;
        //THEN GO TO THE COLUMN INDEX
        for(int j = 0;j<w_width ; j++){
            // MUL EVERY ELEMENT INSIDE OF VECTOR X WITH CORRESPOND WEIGHT ROW
            sum += flatten_X[j]* W[0][0][i][j];
        }
        // ADD BIAS THEN ADD OUTPUT
        FLOAT bias = B[0][0][0][i];
        sum+= bias;
        Z[0][0][0][i] = sum;
    }
}
*/

/*
 * Applies the ReLU activation function: Z = ReLU(X)
 * Tensor * X: input Tensor
 * Tensor * Z: output Tensor
 */
/*
void ReLU(Tensor * X , Tensor * Z)
{

    int x_channel = X->size[0];
    int x_height = X->size[1];
    int x_width = X->size[2];

    for (int i = 0;i<x_channel;i++){
        for(int j=0;j<x_height;j++ ){
            for(int k=0;k<x_width;k++){
                FLOAT element = X[0][i][j][k];
                if (element > 0){
                    Z[0][i][j][k] = element;
                }else{
                    Z[0][i][j][k]=0.0f;
                }
            }
        }
    }
}
*/
/*
 * Applies the Softmax activation function z = exp(x_i)/sum(exp(x_j))
 * This is a stable Softmax implementation
 * Tensor * X: input vector in Tensor form
 * Tensor * Z: output vector in Tensor form
 */
/*
void Softmax(Tensor * X, Tensor * Z)
{
    FLOAT* flatten_X = FLATTEN_INDEX(X);

    int flatten_size = X->size[0]*X->size[1]*X->size[2];
    double softmax_sum_score = 0;

    for(int i=0;i<flatten_size;i++){
        softmax_sum_score += exp(flatten_X[i]);
    }

    //IT'S ALREADY FLATTEN !
    for(int i=0;i<flatten_size;i++){
        double one_softmax= exp(flatten_X[i])/softmax_sum_score;
        Z[0][0][0][i] = one_softmax;
    }

}
*/