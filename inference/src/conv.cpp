#include "conv.h"
#include <iostream>
#include <cstring>
/*----------------------------- Helper Functions -------------------------------------*/
/*
 * This takes the input matrix c (*W_in)[c] flips it and stores it
 * in the real values of the complex Matrix W_out (1 channel)
 * Inputs:
 * Tensor	*	W_in:	Pointer to input Tensor (real)
 * C_Tensor	*	W_out:	Pointer to the output Tensor values stored in real part (complex)
 * int 			c:		Channel of input to use
 */
using namespace std;
void flip_Matrix(Tensor * W_in, C_Tensor * W_out, int c)
{
    int k_width = W_in->size[1];
    int upper_lim = k_width/2 + k_width%2;
    int extra = -1 + k_width%2;
    for(int i =-k_width/2; i < upper_lim; i++){
        for(int j =-k_width/2; j<upper_lim; j++){
            (*W_out)[0][i+k_width/2][j+k_width/2].real(
                    (*W_in)[c][-i + k_width/2 + extra][-j + k_width/2 + extra]);
        }
    }
}

/* You can experiment with these as well */
const static FFT_STRUCT 	FFT_3 = {8,2,6};
const static FFT_STRUCT 	FFT_5 = {16,4,12};
const static FFT_STRUCT 	FFT_7 = {16,6,10};
const static FFT_STRUCT 	FFT_11= {32,10,22};
const FFT_STRUCT * getFFT(uint32_t k_size)
{
    const FFT_STRUCT * fft;
    switch(k_size){
        case 3:
            fft = &FFT_3;
            break;
        case 5:
            fft = &FFT_5;
            break;
        case 7:
            fft = &FFT_7;
            break;
        case 11:
            fft = &FFT_11;
            break;
        default:
            printf("Kernel Size %d not supported by FFT\n",k_size);
            return NULL;
    }
    return fft;
}

const WINOGRAD_STRUCT * getWino(uint32_t k_size)
{
    const WINOGRAD_STRUCT * wino;
    switch(k_size){
        case 3:
            wino = &Wino_F2_3; // Try Wino F4_3
            break;
        case 5:
            wino = &Wino_F4_5;
            break;
        case 7:
            wino = &Wino_F4_7;
            break;
        case 11:
            wino = &Wino_F4_11;
            break;
        default:
            printf("Kernel Size %d not supported by Winograd \n",k_size);
            return NULL;
    }
    return wino;
}

/*-------------------------------- Winograd -------------------------------------------*/
/*
 * Pre Transform the Weights
 * WINOGRAD_STRUCT 	*wino 	: Struct containing tile size, A^T, G, B^T
 * Tensor 			*W		: Untransformed Weight Tensor
 * int		output_channels	: Number of output channels
 * Return:		Tensor *	: New Tensor containing transformed Weights
 */

Tensor * winoWeights(Tensor * W, int output_channels)
{
    const WINOGRAD_STRUCT* vino = getWino(W->size[1]);
    //get G matrix
    const float** G = vino->G;

    int g_rows = vino->tile_size;
    int g_col = vino->kernel_size;

    WINOGRAD_STRUCT* wino = nullptr;
    Tensor *U_wino = new Tensor[output_channels];

    for (size_t filters = 0; filters < output_channels; filters++)
    {
        wino = getWino(W[filters].size[1]);
        //Tensor& W[filters] = W[filters];

        U_wino[filters].allocate(W[filters].size[0], wino->tile_size, wino->tile_size);
        Tensor interim;
        interim.allocate(W[filters].size[0], wino->tile_size, W[filters].size[1]);
        //calculation of interim results of G*W
        for (size_t c = 0; c < W[filters].size[0]; c++)
        {
            for (size_t i = 0; i < g_rows; i++)
            {
                for (size_t j = 0; j < W[filters].size[1]; j++)
                {
                    float temp = 0.0f;
                    for (size_t k = 0; k < g_col; k++)
                    {
                        temp += wino->G[i][k] * W[filters].data[c][k][j];
                    }
                    interim.data[c][i][j] = temp;
                }
            }
        }
        //final weights transformation. Multiplication of interims result times G^T
        for (size_t c = 0; c < interim.size[0]; c++)
        {
            for (size_t i = 0; i < interim.size[1]; i++)
            {
                for (size_t j = 0; j < interim.size[1]; j++)
                {
                    float temp = 0.0f;
                    for (size_t k = 0; k < g_col; k++)
                    {
                        temp += interim.data[c][i][k] * wino->G[j][k];
                    }
                    U_wino[filters].data[c][i][j] = temp;
                }
            }
        }
    }
    return U_wino;


}
/*
 * Convolution using inputs and converted Weights
 * Tensor 			*U_wino	: Transformed Weight Tensor
 * Tensor			*B		: Bias
 * Tensor			*Z		: Output Tensor
 * int 			k_size		: Width and Height of weight kernel
 */

void convWinograd(Tensor* X, Tensor* U_wino, Tensor* B, Tensor* Z, int k_size)
{
    const WINOGRAD_STRUCT* wino = getWino(k_size);

    // Iterate over X tensor with a stride of wino->tile_stride
    for (int x = 0; x < X->size[1]; x += wino->tile_stride) {
        for (int y = 0; y < X->size[2]; y += wino->tile_stride) {
            // Create a temporary tensor to hold a tile from X
            Tensor* convTens = new Tensor(X->size[0], wino->tile_size, wino->tile_size);

            // Extract the tile from X and store it in convTens
            for (int in_channel = 0; in_channel < X->size[0]; in_channel++) {
                int xi = 0;
                int xi_limit = min(x + wino->tile_size, (int)X->size[1]);
                int yi_limit = min(y + wino->tile_size, (int)X->size[2]);
                for (int xi_cur = x; xi_cur < xi_limit; xi_cur++, xi++) {
                    int yi = 0;
                    for (int yi_cur = y; yi_cur < yi_limit; yi_cur++, yi++) {
                        (*convTens)[in_channel][xi][yi] = (*X)[in_channel][xi_cur][yi_cur];
                    }
                }
            }
            // Perform matrix multiplication on convTens and wino->Bt to get transT
            Tensor* transT = new Tensor(X->size[0], wino->tile_size, wino->tile_size);

            // Create a new tensor called 'temp' with dimensions convTens->size[0], wino->tile_size, convTens->size[2]
            Tensor* temp = new Tensor(convTens->size[0], wino->tile_size, convTens->size[2]);

            // Perform calculations for the interim 'temp' tensor
            for (int channel = 0; channel < convTens->size[0]; channel++) {
                for (int row = 0; row < wino->tile_size; row++) {
                    for (int col = 0; col < wino->tile_size; col++) {
                        for (int depth = 0; depth < convTens->size[2]; depth++) {
                            // Update the value at [channel][row][depth] in 'temp' tensor
                            (*temp)[channel][row][depth] += (wino->Bt[row][col] * (*convTens)[channel][col][depth]);
                        }
                    }
                }
            }

            // Perform calculations for 'transT' tensor
            for (int channel = 0; channel < convTens->size[0]; channel++) {
                for (int row = 0; row < wino->tile_size; row++) {
                    for (int col = 0; col < wino->tile_size; col++) {
                        FLOAT sum = 0.0f;
                        for (int depth = 0; depth < wino->tile_size; depth++) {
                            // Update the value at [channel][row][col] in 'transT' tensor
                            sum += ((*temp)[channel][row][depth] * wino->Bt[col][depth]);
                        }
                        (*transT)[channel][row][col] = sum;
                    }
                }
            }
            delete temp;
            delete convTens;
            // Iterate over output channels of Z tensor
            for (int out_channel = 0; out_channel < Z->size[0]; out_channel++) {
                Tensor* W = &U_wino[out_channel];
                Tensor* addT = new Tensor(1, wino->tile_size, wino->tile_size);

                // Perform element-wise multiplication and accumulation
                for (int in_channel = 0; in_channel < X->size[0]; in_channel++) {
                    for (int xi = 0; xi < wino->tile_size; xi++) {
                        for (int yi = 0; yi < wino->tile_size; yi++) {
                            (*addT)[0][xi][yi] += ((*transT)[in_channel][xi][yi] * (*W)[in_channel][xi][yi]);
                        }
                    }
                }

                // Perform matrix multiplication on addT and wino->At to get invT
                Tensor* invT = new Tensor(1, wino->out_size, wino->out_size);
                // Create a interim tensor called 'temp' with dimensions addT->size[0], wino->out_size, addT->size[2]
                Tensor* temp = new Tensor(addT->size[0], wino->out_size, addT->size[2]);

                // Perform calculations for 'temp' tensor
                for (int channel = 0; channel < addT->size[0]; channel++) {
                    for (int row = 0; row < wino->out_size; row++) {
                        for (int col = 0; col < wino->tile_size; col++) {
                            for (int depth = 0; depth < addT->size[2]; depth++) {
                                // Update the value at [channel][row][depth] in 'temp' tensor
                                (*temp)[channel][row][depth] += (wino->At[row][col] * (*addT)[channel][col][depth]);
                            }
                        }
                    }
                }

                // Perform calculations for 'invT' tensor
                for (int channel = 0; channel < addT->size[0]; channel++) {
                    for (int row = 0; row < wino->out_size; row++) {
                        for (int col = 0; col < wino->out_size; col++) {
                            FLOAT sum = 0.0f;
                            for (int depth = 0; depth < wino->tile_size; depth++) {
                                // Update the value at [channel][row][col] in 'invT' tensor
                                sum += ((*temp)[channel][row][depth] * wino->At[col][depth]);
                            }
                            (*invT)[channel][row][col] = sum;
                        }
                    }
                }
                delete temp;
                delete addT;

                // Compute limits for storing invT values into Z
                int x_limit = min(x + wino->tile_stride - 1, (int)Z->size[1] - 1);
                int y_limit = min(y + wino->tile_stride - 1, (int)Z->size[2] - 1);
                int counter_x = x;

                // Store the computed values into Z tensor
                for (int xi = 0; xi < wino->tile_stride && counter_x <= x_limit; xi++, counter_x++) {
                    int counter_y = y;
                    for (int yi = 0; yi < wino->tile_stride && counter_y <= y_limit; yi++, counter_y++) {
                        (*Z)[out_channel][counter_x][counter_y] = (*invT)[0][xi][yi] + (*B)[0][0][out_channel];
                    }
                }
                delete invT;
            }
            delete transT;
        }
    }
}


/*-------------------------------- FFT  -----------------------------------------------*/
/*
 * Pre Transform the weights depending on the tile size
 * FFT_STRUCT 		*fft	: Struct containing tile size (N), overlap and stride
 * Tensor 			*W		: Untransformed Weight Tensor
 * int		output_channels	: Number of output channels
 * Return:		C_Tensor *	: New Tensor containing transformed Weights
 */
C_Tensor * fftWeights(Tensor * W, int output_channels)
{

    int kernel_channels = W->size[0];
    int k_width = W->size[1];
    int k_height = W->size[2];

    const FFT_STRUCT * fftStruct = getFFT(W->size[2]);
    int tile_size = fftStruct->tile_size;
    int overlap = fftStruct->overlap;
    int stride = fftStruct->tile_stride;

    C_Tensor * W_out = new C_Tensor[output_channels];

    for(int i = 0; i < output_channels; i++){
        W_out[i].allocate(kernel_channels, tile_size, tile_size);
    }
    for(int c = 0 ; c<output_channels;c++) {
        for (int i = 0; i < kernel_channels; i++) {
            C_Tensor temp(1, tile_size, tile_size);
            C_Tensor temp2(1, tile_size, tile_size);
            flip_Matrix(&W[c], &temp, i);
            fft2d(&temp, &temp2, 1);
            for (int j = 0; j < tile_size; j++) {
                for (int k = 0; k < tile_size; k++) {
                    W_out[c][i][j][k] = temp2.data[0][j][k];
                }
            }
        }
    }
    return W_out;
}


/*
 * Convolution using inputs and converted Weights
 * FFT_STRUCT 		*fft	: Struct containing tile size (N), overlap and stride
 * C_Tensor			*U_fft:	: Complex Transformed Weight Tensor
 * Tensor			*B		: Bias
 * Tensor			*Z		: Output Tensor
 * int 			k_size		: Width and Height of weight kernel
 */


void convFFT(Tensor * X, C_Tensor * U_fft, Tensor * B,
             Tensor * Z, int k_size) {
    int input_channels = X->size[0];
    int i_width = X->size[1];
    int i_height = X->size[2];

    int weight_channels = U_fft->size[0];
    int w_width = U_fft->size[1];
    int w_height = U_fft->size[2];

    const FFT_STRUCT *fftStruct = getFFT(k_size);
    int tile_size = fftStruct->tile_size;
    int discard = fftStruct->overlap;
    int stride = fftStruct->tile_stride;

    int z_channels = Z->size[0];
    int z_width = Z->size[1];
    const int z_height = Z->size[2];

    Tensor* X_padded = nullptr;
    if (i_width < tile_size) {
        X_padded = new Tensor(input_channels, tile_size, tile_size);
        for (int c = 0; c < input_channels; c++) {
            for (int i = 0; i < tile_size; i++) {
                for (int j = 0; j < tile_size; j++) {
                    X_padded->operator[](c)[i][j] = (i < i_width && j < i_height) ? X->operator[](c)[i][j] : 0;
                }
            }
        }
    } else {
        X_padded = new Tensor(input_channels, i_width + w_width - 1, i_width + w_width - 1);
        for (int c = 0; c < input_channels; c++) {
            for (int i = 0; i < i_height; i++) {
                for (int j = 0; j < i_width; j++) {
                    X_padded->operator[](c)[i][j] = X->operator[](c)[i][j];
                }
            }
        }
    }


    int i_padded_height = X_padded->size[2];

    int size = ceil((i_padded_height - w_height + 1) / (stride*1.0f)) * ceil((i_padded_height - w_height + 1) / (stride*1.0f));

    size = (size == 0) ? 1 : size;

    C_Tensor* temp_for_output = new C_Tensor[size];
    for (int i = 0; i < size; i++) {
        temp_for_output[i].allocate(z_channels, tile_size, tile_size);
    }


    C_Tensor* temp_for_input = new C_Tensor[size];
    for (int i = 0; i < size; i++) {
        temp_for_input[i].allocate(input_channels, tile_size, tile_size);
    }

    for (int c =0;c<input_channels;c++){
        int how_many_tiles = 0;
        for (int slide_i=0; slide_i < i_width; slide_i += stride) {
            for (int slide_j = 0; slide_j < i_width; slide_j += stride ) {
                C_Tensor temp(1, tile_size, tile_size);
                C_Tensor temp_to_fft2d (1, tile_size, tile_size);
                int temp_i = 0;
                int temp_j = 0;
                for (int k = slide_i; k < slide_i + w_height; k++) {
                    for (int m = slide_j; m < slide_j + w_width; m++) {
                        temp[0][temp_i][temp_j] = X_padded[0][c][k][m];
                        temp_j++;
                    }
                    temp_i++;
                    temp_j = 0;
                }
                temp_i = 0;

                fft2d(&temp, &temp_to_fft2d, 1);

                memcpy(temp_for_input[how_many_tiles][c][0],  temp_to_fft2d[0][0], tile_size * tile_size * sizeof(C_FLOAT));

                how_many_tiles++;

            }
        }
    }

    for (int number_of_weights = 0; number_of_weights < z_channels; number_of_weights++) {
        for (int c =0;c<weight_channels;c++){
            int how_many_times = 0;
            for (int slide_i=0; slide_i < i_width; slide_i += stride) {
                for (int slide_j = 0; slide_j < i_width; slide_j += stride) {
                    C_Tensor temp_to_fft2d (1, tile_size, tile_size);
                    for (int i =0;i<tile_size;i++){
                        for (int j =0;j<tile_size;j++){
                            temp_to_fft2d[0][i][j] = temp_for_input[how_many_times][c][i][j];
                        }
                    }
                    for (int k = 0; k < tile_size; ++k) {
                        for (int l = 0; l < tile_size; ++l) {
                            temp_to_fft2d[0][k][l] = temp_to_fft2d[0][k][l] * U_fft[number_of_weights][c][k][l];
                        }
                    }
                    //slide it
                    if(how_many_times==size){
                        how_many_times--;
                    }

                    for (int k = 0; k < tile_size; ++k) {
                        for (int l = 0; l < tile_size; ++l) {
                            temp_for_output[how_many_times][number_of_weights][k][l] += temp_to_fft2d[0][k][l];
                        }
                    }
                    how_many_times++;
                }
            }

        }
        for (int i = 0;i<size;i++){
            C_Tensor temp_for_i (1, tile_size, tile_size);
            C_Tensor temp_for_inverse (1, tile_size, tile_size);

            memcpy(temp_for_i[0][0], temp_for_output[i][number_of_weights][0], tile_size * tile_size * sizeof(C_FLOAT));

            ifft2d(&temp_for_i, &temp_for_inverse, 1);
            for (int j = discard;j<tile_size;j++){
                for (int k = discard;k<tile_size;k++){
                    temp_for_output[i][number_of_weights][j-discard][k-discard] = C_FLOAT(temp_for_inverse[0][j][k].real(),0);
                }
            }
        }
    }

    int number_of_tiles_in_a_row = ceil((i_padded_height - tile_size) / stride) + 1;
    for (int tile_number = 0; tile_number < size; tile_number++) {
        for (int c = 0; c < z_channels; c++) {
            for (int k = 0; k < stride; k++) {
                for (int l = 0; l < stride; l++) {
                    int i = ceil(tile_number / number_of_tiles_in_a_row) * stride + k;
                    int j = tile_number % number_of_tiles_in_a_row * stride + l;
                    if ((i < z_width) && (j < z_height)) {
                        Z->operator[](c)[i][j] = temp_for_output[tile_number][c][k][l].real() + B[0][0][0][c];
                    }
                }
            }
        }
    }

    delete[] temp_for_input;
    delete[] temp_for_output;
    delete X_padded;

}


/*--------------------------------------- Basic ------------------------------------------*/
/* Copy your basic function in here! */
void convBasic(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z)
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

    for (int i = 0; i < z_channel; i++) {
        for (int j = 0; j < z_height; j++) {
            for (int k = 0; k < z_width; k++) {
                float sum = 0;
                for (int c = 0; c < x_channel; c++) {
                    for (int p = 0; p < w_height; p++) {
                        for (int q = 0; q < w_width; q++) {
                            sum += X[0][c][j + p][k + q] * W[i][c][p][q];
                        }
                    }
                }
                sum += b[0][0][0][i];
                Z[0][i][j][k] = sum;
            }
        }
    }
}
