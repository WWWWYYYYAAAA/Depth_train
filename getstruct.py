import torch
from torchviz import make_dot
from ResCNN import ResNetUNet

model = ResNetUNet(in_channels=3, out_channels=1)
g= make_dot(model(torch.rand(1, 3,200,200)),params=dict(model.named_parameters()))
g.format = "png"
g.directory = "./"
g.view()