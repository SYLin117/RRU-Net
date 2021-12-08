import torch
import torch.nn as nn
import torch.nn.functional as F


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class CombineModel(nn.Module):
    def __init__(self, model1, model2, n_classes):
        super(CombineModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.out = outconv(2, n_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)

        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)
        x = self.out(x)
        return x

