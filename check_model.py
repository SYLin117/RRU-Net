import os.path

import torch.backends.cudnn as cudnn
from torch import optim

from eval import eval_net
from unet.unet_model import *
from utils import *

import matplotlib.pyplot as plt
import time

import wandb
from torchinfo import summary
from self_attention_cv import TransUnet

if __name__ == '__main__':
    net = MyTransUNet2(classes=1, img_dim=256)
    summary(net, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))

    # a = torch.rand(2, 3, 128, 128)
    # net = TransUnet(in_channels=3, img_dim=512, vit_blocks=8,
    #                   vit_dim_linear_mhsa_block=512, classes=1)
    # summary(net, input_size=(1, 3, 512, 512), col_names=("input_size", "output_size"))
    # import pathlib
    # current_path = str(pathlib.Path().resolve())