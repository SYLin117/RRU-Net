""" Full assembly of the parts to form the complete network """

from self_attention_cv import TransUnet
from .transunet_parts import *


class MyTransUNet(nn.Module):
    def __init__(self, in_channels=3, img_dim=256, vit_blocks=8,
                 vit_dim_linear_mhsa_block=512, classes=2):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.srm = SRMConv(in_channels=in_channels, )
        self.transunet = TransUnet(in_channels=in_channels * 2, img_dim=img_dim, vit_blocks=vit_blocks,
                                   vit_dim_linear_mhsa_block=vit_dim_linear_mhsa_block, classes=classes)

    def forward(self, x):
        srm_feature = self.srm(x)
        x = torch.cat((x, srm_feature), 1)
        x = self.transunet(x)
        return x


class MyTransUNet2(nn.Module):
    def __init__(self, in_channels=3, img_dim=256, vit_blocks=8,
                 vit_dim_linear_mhsa_block=512, classes=2):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.srm = SRMConv(in_channels=in_channels, )
        self.transunet = TransUnet(in_channels=in_channels, img_dim=img_dim, vit_blocks=vit_blocks,
                                   vit_dim_linear_mhsa_block=vit_dim_linear_mhsa_block, classes=classes)

    def forward(self, x):
        # srm_feature = self.srm(x)
        # x = torch.cat((x, srm_feature), 1)
        x = self.transunet(x)
        return x


if __name__ == '__main__':
    net = MyTransUNet2(classes=1, img_dim=512)
    summary(net, input_size=(1, 3, 512, 512), col_names=("input_size", "output_size"))
