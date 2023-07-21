import sys
sys.path.append("utils")
from test_utils import *
from pathlib import Path
Path("data").mkdir(parents=True, exist_ok=True)
'''
createConvTest(input_size,output_channels,kernel_width,outfile)
----
Creates a test for the convolution Layer:
Inputs:
    input_size[3] = array of [ #channels, y-dim, x-dim]
    output_channels = #Output Channels
    kernel_width = Kernel width (Square kernel W_y = W_x)
    outfile = File to write to (Has to be open
'''
f = open("data/conv_test.dat","wb")

createConvTest([1,134,134],1,7,f)
f.close()

# '''
# createLinearTest(input_size,outputs,outfile)
# ----
# Creates a test for the Linear Layer:i
# Inputs:
#     input_size[3] = array of [ #channels, y-dim, x-dim]
#     outputs=    #Output neurons
#     outfile=    File to write to has to be open
# '''
# f = open("data/linear_test.dat","wb")
# createLinearTest([1,1,100],10,f)
# createLinearTest([1,1,200],500,f)
# # Test non-flattened inputs
# createLinearTest([256,12,12],1000,f)
# createLinearTest([512,4,4],200,f)
# f.close()
#
#
# '''
# createPoolTest(input_size,outfile)
# ----
# Creates a test for the max pool Layer (size = stride = 2):
# Inputs:
#     input_size[3] = array of [ #channels, y-dim, x-dim]
#     outfile =   File to write to (Has to be open)
# '''
# f = open("data/pool_test.dat","wb")
#
#
# createPoolTest([64,24,24],f)
# createPoolTest([128,12,12],f)
# createPoolTest([16,224,224],f)
#
# f.close()
#
# '''
# createSoftmaxTest(inputs,outfile)
# ----
# Creates a test for the softmax Layer:
# Inputs:
#     inputs =    Number of Inputs (already flattened)
#     outfile =   File to write to (Has to be open)
# '''
# f = open("data/softmax_test.dat","wb")
# createSoftmaxTest(20,f)
# createSoftmaxTest(10,f)
# createSoftmaxTest(1000,f)
# f.close()
#
# '''
# createReLUTest(input_size,outfile)
# ----
# Creates a test for the ReLU Layer:
# Inputs:
#     input_size[3] = array of [ #channels, y-dim, x-dim]
#     outfile =   File to write to (Has to be open
# '''
# f = open("data/relu_test.dat","wb")
# createReLUTest([1,1,100],f)
# createReLUTest([16,224,224],f)
# createReLUTest([512,16,16],f)
# f.close()
