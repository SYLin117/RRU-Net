import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import re
from pathlib import Path
from sys import platform


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    # this function make img to square due to network architecture of UNet
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]


def split_img_into_squares(img):
    """
    return left square and right square image
    """
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    """
    h,w,c -> c, h, w
    """
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def find_latest_epoch(dir):
    epoch_re = re.compile(r'checkpoint_epoch([0-9]+).pth')

    def func(st):  # I am using your first string as a running example in this code
        epoch_no = epoch_re.match(st).groups()[0]
        return int(epoch_no)

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith('.pth')]
    # files.sort()
    files = sorted(files, key=lambda x: func(x))
    latest_epoch = files[-1]
    epoch_no = epoch_re.match(latest_epoch).groups()[0]
    return os.path.join(dir, latest_epoch), int(epoch_no)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def get_dataset_root():
    """
    return 0: linux
    return 1: OS X
    return 2: windows
    """
    if platform == "linux" or platform == "linux2":
        # linux
        return Path(r'/media/ian/WD/datasets')
    elif platform == "darwin":
        # OS X
        raise Exception("not dataset on mac.")
    elif platform == "win32":
        # Windows...
        return Path(r'F:\datasets')
