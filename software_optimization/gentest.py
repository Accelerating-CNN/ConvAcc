import numpy as np
import scipy as sp
import scipy.signal
import os,sys
sys.path.append("utils")
import torch
from torch import nn
from test_utils import *
from pathlib import Path
Path("data").mkdir(parents=True, exist_ok=True)


def createConvTest(in_channels,out_channels,ksize,pad,dimx,outfile):
    conv = nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,ksize,1,pad,
                            padding_mode='zeros',
                            bias=True,dtype=torch.float32)
            )
    x = torch.rand(in_channels,dimx,dimx)
    w = conv[0].weight.data
    b = conv[0].bias.data
    z = conv(x)


    #print(f"in_channels:{in_channels} out_channels:{out_channels} ksize:{ksize} pad:{pad} dimx:{dimx}")
    #print(f"x_tensor: {x}, x_shape: {x.shape}")
    #print(f"w_tensor: {w}, w_shape: {w.shape}")
    #print(f"b_tensor: {b}, b_shape: {b.shape}")
    #print(f"z_tensor: {z}, z_shape: {z.shape}")

    """print(f"x_shape: {x.shape}")
    print(f"w_shape: {w.shape}")
    print(f"b_shape: {b.shape}")
    print(f"z_shape: {z.shape}")
    print("---------------------------------")"""

    writeTensor(x,outfile)
    writeTensor(z,outfile)
    writeTensor(w,outfile)
    writeTensor(b,outfile)


f = open("data/conv_test.dat","wb")
createConvTest(1,1,3,0,6,f)
createConvTest(2,1,3,0,6,f)
createConvTest(2,2,3,0,6,f)
createConvTest(1,1,3,0,24,f)
createConvTest(8,8,5,0,24,f)
createConvTest(64,128,3,0,112,f)
createConvTest(128,256,3,0,64,f)
createConvTest(256,512,3,0,32,f)
createConvTest(16,33,5,0,512,f)
createConvTest(13,52,7,0,248,f)
createConvTest(24,24,11,0,124,f)
createConvTest(12,1,11,0,114,f)
f.close()
