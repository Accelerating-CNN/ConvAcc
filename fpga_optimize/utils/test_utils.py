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
    writeTensor(x,outfile)
    writeTensor(z,outfile)
    writeTensor(w,outfile)
    writeTensor(b,outfile)

'''
createLinearTest(input_size,outputs,outfile)
----
Creates a test for the Linear Layer:
Inputs:
    input_size[3] = array of [ #channels, y-dim, x-dim]
    outputs=    #Output neurons
    outfile=    File to write to has to be open
'''
def createLinearTest(input_size,outputs,outfile):
    fc= nn.Sequential(
            torch.nn.Flatten(0,2),
            torch.nn.Linear(input_size[0] * input_size[1] * input_size[2],outputs,bias=True,dtype=torch.float32),
            )
    x = torch.rand(input_size[0],input_size[1],input_size[2])
    w = fc[1].weight.data
    b = fc[1].bias.data
    z = fc(x)
    writeTensor(x,outfile)
    writeTensor(z,outfile)
    writeTensor(w,outfile)
    writeTensor(b,outfile)


'''
createPoolTest(input_size,outfile)
----
Creates a test for the max pool Layer (size = stride = 2):
Inputs:
    input_size[3] = array of [ #channels, y-dim, x-dim]
    outfile =   File to write to (Has to be open
'''
def createPoolTest(input_size,outfile):
    x = torch.rand(input_size[0],input_size[1],input_size[2])
    pool = torch.nn.MaxPool2d(2)
    z = pool(x)
    writeTensor(x,outfile)
    writeTensor(z,outfile)
    
'''
createSoftmaxTest(inputs,outfile)
----
Creates a test for the softmax Layer:
Inputs:
    inputs =    Number of Inputs (already flattened) 
    outfile =   File to write to (Has to be open)
'''
def createSoftmaxTest(inputs,outfile):
    x = torch.rand(1,1,inputs)
    sm = torch.nn.Softmax(2)
    z = sm(x)
    writeTensor(x,outfile)
    writeTensor(z,outfile)

'''
createReLUTest(input_size,outfile)
----
Creates a test for the ReLU Layer:
Inputs:
    input_size[3] = array of [ #channels, y-dim, x-dim]
    outfile =   File to write to (Has to be open
'''
def createReLUTest(input_size,outfile):
    x = torch.rand(input_size[0],input_size[1],input_size[2])
    rl = torch.nn.ReLU(2)
    z = rl(x)
    writeTensor(x,outfile)
    writeTensor(z,outfile)


def writeNetTest(cnn,insize,num_test,outfile):
    f = open(outfile,"wb")
    f.write(num_test.to_bytes(4,byteorder='little',signed=False))
    for i in range(0,num_test):
        input_batch = torch.rand(insize);
        output_batch = cnn(input_batch)
        if(len(output_batch.size()) == 1):
            output_batch = output_batch.unsqueeze(0)
        output = torch.nn.functional.softmax(output_batch[0], dim=0)
        writeTensor(input_batch[0],f)
        writeTensor(output,f)
    f.close()


# Has to be in sync with c definitions
DEF_LAYER_TYPE_FC 		=0
DEF_LAYER_TYPE_POOL		=1
DEF_LAYER_TYPE_RELU		=2
DEF_LAYER_TYPE_CONV		=3
DEF_LAYER_TYPE_SOFTMAX	=4


# Class holding the weights and some information of a layer
class CNN_layer:
    lay_type = 0
    in_channels = 0
    out_channels = 0
    k_width = 0
    pad = 0
    num_params = 0
    weights = None
    bias = None
    def __init__(self,m,in_channels):
        self.in_channels = in_channels
        self.out_channels = in_channels
        if(isinstance(m,nn.ReLU)):
            self.lay_type = DEF_LAYER_TYPE_RELU
        elif(isinstance(m,nn.MaxPool2d)):
            self.lay_type = DEF_LAYER_TYPE_POOL
        elif(isinstance(m,nn.Softmax)):
            self.lay_type = DEF_LAYER_TYPE_SOFTMAX
        elif(isinstance(m,nn.Linear)):
            self.lay_type = DEF_LAYER_TYPE_FC
            self.out_channels = 1
            self.weights = m.weight
            self.bias = m.bias
            self.num_params = reduce(lambda x,y: x*y,m.weight.size()) + \
                    reduce(lambda x,y: x*y, m.bias.size())
        elif(isinstance(m,nn.Conv2d)):
            self.lay_type = DEF_LAYER_TYPE_CONV
            self.out_channels = m.out_channels
            self.weights = m.weight
            self.k_width = m.kernel_size[0]
            self.pad = m.padding[0]
            self.num_params = reduce(lambda x,y: x*y,m.weight.size())
            if(m.bias != None ):
                self.num_params += reduce(lambda x,y: x*y,m.bias.size())
            self.bias = m.bias
        else:
            self.lay_type = None


# Converts a pytorch neural network and writes the layer weights to 
# a file
class CNN:
    def __init__(self,cnn,in_channels,add_softmax = True):
        self.layers = []
        self.in_channels =0
        self.in_channels = in_channels
        csize = in_channels 
        self.model = cnn
        self.model.eval()
        lay = [module for module in self.model.modules() if not isinstance(module, nn.Sequential)]
        for m in lay:
            l = CNN_layer(m,csize)
            csize = l.out_channels
            if(l.lay_type != None):
                self.layers.append(l)
        if(add_softmax):
            self.add_softmax()

# Add the Softmax layer to the CNN
    def add_softmax(self):
        m = nn.Softmax()
        l = CNN_layer(m,self.layers[-1].out_channels)
        self.layers.append(l)
        return self

# Writes all the layers to the outfile
    def write_layers(self,outfile):
        self.printModelSize()
        f = open(outfile,"wb")
        write_uint(len(self.layers),f)
        for l in self.layers:
            write_uint(l.lay_type,f)
            write_uint(l.in_channels,f)
            write_uint(l.out_channels,f)
            write_uint(l.k_width,f)
            write_uint(l.pad,f)
            if(l.weights != None):
                writeTensor(l.weights,f)
            if(l.bias != None):
                writeTensor(l.bias,f)
        f.close()
        return self

    def write_test(self,input_size,num_test,outfile):
        f = open(outfile,"wb")
        f.write(num_test.to_bytes(4,byteorder='little',signed=False))
        for i in range(0,num_test):
            input_batch = torch.rand(input_size,requires_grad=False);
            output_batch = self.model(input_batch.unsqueeze(0))
            if(len(output_batch.size()) == 1):
                output_batch = output_batch.unsqueeze(0)
            output = torch.nn.functional.softmax(output_batch[0], dim=0)
            writeTensor(input_batch,f)
            writeTensor(output,f)
        f.close()

    def printModelSize(self):
        params = sum(p.numel() for p in self.model.parameters())
        paramkb= params * 4 /(1024)
        print("Model (%s) size [kB] : %lf"%(type(self.model).__name__,paramkb))

        


