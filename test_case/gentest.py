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


"""
x_ten    [[3.7675e-01, 6.1596e-01, 4.6187e-02, 1.9729e-05, 5.7698e-02, 8.1506e-01, 2.6274e-01, 7.8525e-01],
         [1.6604e-01, 5.9564e-01, 5.8755e-01, 2.0669e-01, 8.1185e-01,9.5683e-01, 7.5073e-01, 1.1130e-01],
         [2.8830e-01, 6.4722e-01, 1.2932e-01, 2.2815e-01, 4.7797e-01, 7.5663e-01, 2.9540e-01, 7.0703e-01],
         [6.4088e-01, 7.6655e-01, 1.0682e-01, 4.6361e-01, 6.3186e-01,
          3.6386e-01, 9.9764e-01, 5.6622e-01],
         [4.4591e-01, 5.6303e-01, 8.9155e-01, 6.9242e-02, 3.6713e-01,
          9.2206e-01, 9.0694e-01, 1.3046e-01],
         [4.4551e-01, 1.3300e-01, 6.4212e-01, 8.5080e-01, 7.6202e-01,
          2.9940e-03, 2.9903e-01, 9.8364e-01],
         [4.9995e-01, 3.1400e-01, 5.2115e-01, 4.4970e-02, 7.6703e-01,
          6.0845e-01, 4.3708e-01, 7.9391e-02],
         [7.6813e-01, 5.9715e-01, 1.0873e-01, 9.1692e-01, 2.1563e-01,
          8.4179e-01, 4.7271e-01, 2.2743e-01]]]), x_shape: torch.Size([1, 8, 8])
w_tensor: tensor([[[[-0.0939,  0.1895, -0.0996],
          [ 0.2052, -0.1472, -0.2777],
          [-0.3104, -0.1561,  0.2261]]]]), w_shape: torch.Size([1, 1, 3, 3])
b_tensor: tensor([0.1466]), b_shape: torch.Size([1])
z_tensor: tensor([[[-0.1547, -0.0937,  0.0335, -0.2409, -0.1130, -0.0246],
         [-0.1812,  0.0820, -0.0528, -0.2082,  0.0542, -0.0373],
         [ 0.1935, -0.1979, -0.2963,  0.1711, -0.0381, -0.5667],
         [-0.0318,  0.0645,  0.0724, -0.4917, -0.4248,  0.4393],
         [-0.0703, -0.2164, -0.1610,  0.1832,  0.0351, -0.3368],
         [-0.3295,  0.1512, -0.0692, -0.1898, -0.0992, -0.1403]]],
       grad_fn=<SqueezeBackward1>), z_shape: torch.Size([1, 6, 6])
"""

