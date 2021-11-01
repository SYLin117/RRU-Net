""" Full assembly of the parts to form the complete network """
import torch
from torch import nn
from self_attention_cv import TransUnet
from transunet.unet_parts import SRMConv
from torchinfo import summary


class MyTransUNet(nn.Module):
    """
    TransUnet with srm filter
    """
    def __init__(self, img_dim=300, in_channels=3, classes=1,
                 vit_blocks=12,
                 vit_heads=12,
                 vit_dim_linear_mhsa_block=3072,
                 patch_size=8,
                 vit_transformer_dim=768,
                 vit_transformer=None,
                 vit_channels=None, ):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.srm = SRMConv(in_channels=in_channels, )
        self.transunet = TransUnet(img_dim=img_dim, in_channels=in_channels, classes=classes,
                                   vit_blocks=vit_blocks,
                                   vit_heads=vit_heads,
                                   vit_dim_linear_mhsa_block=vit_dim_linear_mhsa_block,
                                   vit_transformer=None,
                                   vit_channels=None, )

    def forward(self, x):
        srm_feature = self.srm(x)
        x = torch.cat((x, srm_feature), 1)
        x = self.transunet(x)
        return x


class MyTransUNet2(nn.Module):
    """
    original TransUnet
    """
    def __init__(self, img_dim=300, in_channels=3, classes=1,
                 vit_blocks=12,
                 vit_heads=12,
                 vit_dim_linear_mhsa_block=3072,
                 patch_size=8,
                 vit_transformer_dim=768,
                 vit_transformer=None,
                 vit_channels=None, ):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = classes
        self.transunet = TransUnet(in_channels=in_channels, img_dim=img_dim, vit_blocks=vit_blocks,
                                   vit_dim_linear_mhsa_block=vit_dim_linear_mhsa_block, classes=classes)

    def forward(self, x):
        x = self.transunet(x)
        return x


if __name__ == '__main__':
    net = MyTransUNet2(classes=1)
    summary(net, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))
