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

if __name__ == '__main__':
    # net = Ringed_Res_Unet(n_channels=3, n_classes=1)
    # summary(net, input_size=(1, 3, 128, 128), col_names=("input_size", "output_size"))
    import pathlib
    current_path = str(pathlib.Path().resolve())