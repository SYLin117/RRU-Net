import os.path

import torch.backends.cudnn as cudnn
from torch import optim
from torch.nn import Dropout, Linear, Sigmoid

from eval import eval_net
from unet.unet_model import *
from utils import *

import matplotlib.pyplot as plt
import time

import wandb
from torchinfo import summary
from self_attention_cv import TransUnet, ResNet50ViT, ViT
from torchvision.models import vgg16, resnet18, resnet50
from efficientnet_pytorch import EfficientNet
from efficientnet import EfficientNet_b0, EfficientNet_b1, EfficientNet_b2, EfficientNet_b3, EfficientNet_b4, \
    EfficientNet_b5
from u2net_model import U2NET, U2NETP
from combine_model import CombineModel

if __name__ == '__main__':
    print("===main===")
    # net = MyTransUNet2(classes=1, img_dim=256)
    # summary(net, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))

    # net = ViT(img_dim=512, patch_dim=16, num_classes=1, classification=True)
    # summary(net, input_size=(1, 3, 512, 512), col_names=("input_size", "output_size"))
    # print(net)

    # a = torch.rand(2, 3, 128, 128)
    # net = TransUnet(in_channels=3, img_dim=512, vit_blocks=8,
    #                   vit_dim_linear_mhsa_block=512, classes=1)
    # summary(net, input_size=(1, 3, 512, 512), col_names=("input_size", "output_size"))
    # import pathlib
    # current_path = str(pathlib.Path().resolve())

    # net = efficientnet_b2(pretrained=True, progress=True)
    # net.classifier = nn.Sequential(
    #     Dropout(p=0.3, inplace=True),
    #     Linear(in_features=1408, out_features=1, bias=True),
    #     Sigmoid(),
    # )
    # print(net)
    # net = model = vgg16(pretrained=True)
    # print(net)

    # net = EfficientNet_b4(num_classes=1)
    # summary(net, input_size=(1, 3, 300, 300), col_names=("input_size", "output_size"))
    # print(net)

    # net = EfficientNet_b5(num_classes=1)
    # x = torch.rand(4, 3, 300, 300)
    # y = net(x)
    # summary(net, input_size=(1, 3, 300, 300), col_names=("input_size", "output_size"))

    # net = resnet18(pretrained=True)
    # print(net)
    # summary(net, input_size=(1, 3, 300, 300), col_names=("input_size", "output_size"))

    net1 = Unet(3, 1)
    net2 = Res_Unet(3, 1)
    net = CombineModel(net1, net2, 1)

    summary(net, input_size=(1, 3, 300, 300), col_names=("input_size", "output_size"))

    # net = Res_Unet(n_channels=3, n_classes=1)
    # summary(net, input_size=(1, 3, 300, 300), col_names=("input_size", "output_size"))