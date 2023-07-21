import numpy as np
import torch
from torch import nn
from functools import reduce
import struct



def write_uint(a,f):
    f.write(a.to_bytes(4,byteorder='little',signed=False))


def binTensor(T,f):
    if(len(T.size()) == 1):
        l = T.tolist()
        f.write(struct.pack('<%sf' % len(l),*l))
    else:
        for i in range(T.size()[0]):
            binTensor(T[i],f)


def writeTensor(T,f):
    if(len(T.size()) > 4):
        print("Higher dim than 4 not supported !\n")
    elif(len(T.size()) < 3):
        writeTensor(T.unsqueeze(0),f)
    elif(len(T.size()) == 4):
        for i in range(0,T.size()[0]):
            writeTensor(T[i],f)
    else:
        write_uint(T.size()[0],f)
        write_uint(T.size()[1],f)
        write_uint(T.size()[2],f)
        binTensor(T,f)


'''
def createConvTest(input_size,output_channels,kernel_width,outfile)
----
Creates a test for the convolution Layer:
Inputs:
    input_size[3] = array of [ #channels, y-dim, x-dim]
    output_channels = #Output Channels
    kernel_width = Kernel width (Square kernel W_y = W_x)
    outfile = File to write to (Has to be open
'''

def createConvTest(input_size,output_channels,kernel_width,outfile):
    conv = nn.Sequential(
            torch.nn.Conv2d(input_size[0],output_channels,kernel_width,1,0,
                            padding_mode='zeros',
                            bias=True,dtype=torch.float32)
            )
    x = torch.rand(input_size[0],input_size[1],input_size[2])
    w = conv[0].weight.data
    b = conv[0].bias.data
    z = conv(x)


    print(f"x_tensor: {x}, x_shape: {x.shape}")
    print(f"w_tensor: {w}, w_shape: {w.shape}")
    print(f"b_tensor: {b}, b_shape: {b.shape}")
    print(f"z_tensor: {z}, z_shape: {z.shape}")



    writeTensor(x,outfile)
    writeTensor(z,outfile)
    writeTensor(w,outfile)
    writeTensor(b,outfile)

# '''
# createLinearTest(input_size,outputs,outfile)
# ----
# Creates a test for the Linear Layer:
# Inputs:
#     input_size[3] = array of [ #channels, y-dim, x-dim]
#     outputs=    #Output neurons
#     outfile=    File to write to has to be open
# '''
# def createLinearTest(input_size,outputs,outfile):
#     fc= nn.Sequential(
#             torch.nn.Flatten(0,2),
#             torch.nn.Linear(input_size[0] * input_size[1] * input_size[2],outputs,bias=True,dtype=torch.float32),
#             )
#     x = torch.rand(input_size[0],input_size[1],input_size[2])
#     w = fc[1].weight.data
#     b = fc[1].bias.data
#     z = fc(x)
#     """
#     print(f"x_tensor: {x}, x_shape: {x.shape}")
#     print(f"w_tensor: {w}, w_shape: {w.shape}")
#     print(f"b_tensor: {b}, b_shape: {b.shape}")
#     print(f"z_tensor: {z}, z_shape: {z.shape}")
#     """
#
#     writeTensor(x,outfile)
#     writeTensor(z,outfile)
#     writeTensor(w,outfile)
#     writeTensor(b,outfile)
#
#
# '''
# createPoolTest(input_size,outfile)
# ----
# Creates a test for the max pool Layer (size = stride = 2):
# Inputs:
#     input_size[3] = array of [ #channels, y-dim, x-dim]
#     outfile =   File to write to (Has to be open
# '''
# def createPoolTest(input_size,outfile):
#     x = torch.rand(input_size[0],input_size[1],input_size[2])
#     pool = torch.nn.MaxPool2d(2)
#     z = pool(x)
#     writeTensor(x,outfile)
#     writeTensor(z,outfile)
#
#     """
#     print(f"x_tensor: {x}, x_shape: {x.shape}")
#     print(f"pool: {pool},")
#     print(f"z_tensor: {z}, z_shape: {z.shape}")
#     """
#
#
#
# '''
# createSoftmaxTest(inputs,outfile)
# ----
# Creates a test for the softmax Layer:
# Inputs:
#     inputs =    Number of Inputs (already flattened)
#     outfile =   File to write to (Has to be open)
# '''
# def createSoftmaxTest(inputs,outfile):
#     x = torch.rand(1,1,inputs)
#     sm = torch.nn.Softmax(2)
#     z = sm(x)
#     writeTensor(x,outfile)
#     writeTensor(z,outfile)
#
# '''
# createReLUTest(input_size,outfile)
# ----
# Creates a test for the ReLU Layer:
# Inputs:
#     input_size[3] = array of [ #channels, y-dim, x-dim]
#     outfile =   File to write to (Has to be open
# '''
# def createReLUTest(input_size,outfile):
#     x = torch.rand(input_size[0],input_size[1],input_size[2])
#     rl = torch.nn.ReLU(2)
#     z = rl(x)
#     writeTensor(x,outfile)
#     writeTensor(z,outfile)
#
#
#
