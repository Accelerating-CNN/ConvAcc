import sys
sys.path.append("utils")
from test_utils import *
from pathlib import Path
Path("data").mkdir(parents=True, exist_ok=True)
from nets import *

tn = CNN(testNet(),1)
tn.write_layers("data/testnet_weights.dat")
tn.write_test([1,8,8],8,"data/testnet_test.dat")

# Load vgg11
# vgg11 = CNN(torch.hub.load('pytorch/vision:v0.10.0', 'vgg11' , pretrained=True),3)

# vgg11.write_layers("data/vgg11_weights.dat")
# vgg11.write_test([3,224,224],1,"data/vgg11_test.dat")

# Load vgg11
vgg16 = CNN(torch.hub.load('pytorch/vision:v0.10.0', 'vgg16' , pretrained=True),3)
vgg16.write_layers("data/vgg16_weights.dat")
vgg16.write_test([3,224,224],1,"data/vgg16_test.dat")

# SmallNet
sn = CNN(smallNet(),3)
sn.write_layers("data/smallnet_weights.dat")
sn.write_test([3,128,128],2,"data/smallnet_test.dat")

#MediumNet (#reduced ZFNet)
mn = CNN(mediumNet(),3)
mn.write_layers("data/mediumnet_weights.dat")
mn.write_test([3,128,128],2,"data/mediumnet_test.dat")

# LargeNet (reduced VGG 11)
ln = CNN(largeNet(),3)
ln.write_layers("data/largenet_weights.dat")
ln.write_test([3,128,128],2,"data/largenet_test.dat")

# GiantNet (reduced VGG 19)
gn = CNN(giantNet(),3)
gn.write_layers("data/giantnet_weights.dat")
gn.write_test([3,128,128],2,"data/giantnet_test.dat")

