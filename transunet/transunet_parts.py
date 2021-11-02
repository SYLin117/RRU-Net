""" Parts of the U-Net model """

import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary


class SRMConv(nn.Module):
    """
    The correct sizes should be

    input: (batch_size, in_channels , height, width)
    weight: (out_channels, in_channels , kernel_height, kernel_width)

    please check: https://stackoverflow.com/questions/61269421/expected-stride-to-be-a-single-integer-value-or-a-list-of-1-values-to-match-the
    """

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.filters = self.get_srm_kernel()
        # self.zero_pad_2d = nn.ZeroPad2d((2, 2, 2, 2))
        self.srm_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=2,
                                   bias=False, )
        self.srm_layer.weight = torch.nn.Parameter(self.filters, requires_grad=False)

    def forward(self, x):
        y = self.srm_layer(x)
        # return x + y
        # return torch.cat((x, y), 0)
        return y

    def get_srm_kernel(self):
        q = [4.0, 12.0, 2.0]

        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        weights = torch.tensor(
            [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]).float()
        # print(weights.size())
        return weights


if __name__ == '__main__':
    a = torch.rand(1, 3, 256, 256)
    srm = SRMConv()
    for param in srm.parameters():
        print(param.shape)
    y = srm(a)
    print(y[0].shape)
    summary(srm, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))
