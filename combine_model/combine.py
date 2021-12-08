import torch
import torch.nn as nn
import torch.nn.functional as F
from .inception_block import InceptionA


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


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
        self.srm_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
                                   bias=False, )
        self.srm_layer.weight = torch.nn.Parameter(self.filters, requires_grad=False)

    def forward(self, x):
        y = self.srm_layer(x)
        # return x + y
        # return torch.cat((x, y), 0)
        return y

    def get_srm_kernel(self):
        q = torch.tensor([4.0, 12.0, 2.0])

        filter1 = torch.tensor([[0, 0, 0, 0, 0],
                                [0, -1, 2, -1, 0],
                                [0, 2, -4, 2, 0],
                                [0, -1, 2, -1, 0],
                                [0, 0, 0, 0, 0]])
        filter2 = torch.tensor([[-1, 2, -2, 2, -1],
                                [2, -6, 8, -6, 2],
                                [-2, 8, -12, 8, -2],
                                [2, -6, 8, -6, 2],
                                [-1, 2, -2, 2, -1]])
        filter3 = torch.tensor([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 1, -2, 1, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])

        filter1 = torch.div(filter1, q[0])
        filter2 = torch.div(filter2, q[1])
        filter3 = torch.div(filter3, q[2])
        # weights = torch.tensor(
        #     [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]).float()
        weights = torch.stack((
            torch.stack((filter1, filter1, filter1), dim=0),
            torch.stack((filter2, filter2, filter2), dim=0),
            torch.stack((filter3, filter3, filter3), dim=0)),
            dim=0)
        print(weights.size())
        return weights


class CombineModel(nn.Module):
    def __init__(self, model1, model2, n_classes):
        super(CombineModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.out = outconv(2, n_classes)
        self.srm_layer = SRMConv(3, 3)
        self.inceptionA = InceptionA(8, 1)

    def forward(self, x):
        srm_feature = self.srm_layer(x)
        x1 = self.model1(x)
        x2 = self.model2(x)

        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1, srm_feature, x], dim=1)
        x = self.inceptionA(x)
        return x
