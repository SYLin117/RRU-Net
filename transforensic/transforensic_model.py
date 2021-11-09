import torch
import torch.nn as nn
from torch.nn import functional as F

from self_attention_cv import ViT, TransUnet

from torchinfo import summary
from torchvision import datasets, transforms, models

from einops import rearrange


class Transforensic(nn.Module):
    def __init__(self, image_size=224, dim=1024, blocks=6, heads=16, mlp_dim=2048, vit_transformer=None, classes=1):
        """

        Args:
            image_size:
            patch_size:
            dim:embedding的維度
            blocks:
            heads:
            mlp_dim:mlp_dim是transformer最後輸出FeedForward的維度, dim->mlp_dim
            vit_transformer:自定義的ViT
        """
        super().__init__()
        assert image_size % 4 == 0
        assert image_size % 8 == 0
        assert image_size % 16 == 0
        assert image_size % 32 == 0
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet_part1 = nn.Sequential(*list(self.resnet50.children())[:5])
        self.resnet_part2 = nn.Sequential(*list(self.resnet50.children())[:6])
        self.resnet_part3 = nn.Sequential(*list(self.resnet50.children())[:7])
        self.resnet_part4 = nn.Sequential(*list(self.resnet50.children())[:8])
        for param in self.resnet_part1.parameters():
            param.requires_grad = False
        for param in self.resnet_part2.parameters():
            param.requires_grad = False
        for param in self.resnet_part3.parameters():
            param.requires_grad = False
        for param in self.resnet_part4.parameters():
            param.requires_grad = False
        self.vit_channels = [256, 512, 1024, 2048]
        self.vit_image_sizes = [int(image_size / 4), int(image_size / 8), int(image_size / 16), int(image_size / 32)]
        self.vit_patch_size = [2, 1, 1, 1]
        self.vit_dim = [256, 512, 1024, 2048]  ## transformer feedforward的維度
        # for i in range(5):
        #     locals()['self.vit' + str(i + 1)] = ViT(img_dim=self.vit_image_sizes[i],  #
        #                                             in_channels=self.vit_channels[i],  # 輸入的channel數
        #                                             patch_dim=self.vit_patch_size[i],
        #                                             dim=dim,  # embedding的維度
        #                                             blocks=blocks,
        #                                             heads=heads,
        #                                             dim_linear_block=mlp_dim,
        #                                             classification=False) if vit_transformer is None else vit_transformer
        self.vit1 = ViT(img_dim=self.vit_image_sizes[0],  #
                        in_channels=self.vit_channels[0],  # 輸入的channel數
                        patch_dim=self.vit_patch_size[0],
                        dim=dim,  # embedding的維度
                        blocks=blocks,
                        heads=heads,
                        dim_linear_block=self.vit_dim[0],
                        classification=False) if vit_transformer is None else vit_transformer

        self.vit2 = ViT(img_dim=self.vit_image_sizes[1],  #
                        in_channels=self.vit_channels[1],  # 輸入的channel數
                        patch_dim=self.vit_patch_size[1],
                        dim=dim,  # embedding的維度
                        blocks=blocks,
                        heads=heads,
                        dim_linear_block=self.vit_dim[1],
                        classification=False) if vit_transformer is None else vit_transformer

        self.vit3 = ViT(img_dim=self.vit_image_sizes[2],  #
                        in_channels=self.vit_channels[2],  # 輸入的channel數
                        patch_dim=self.vit_patch_size[2],
                        dim=dim,  # embedding的維度
                        blocks=blocks,
                        heads=heads,
                        dim_linear_block=self.vit_dim[2],
                        classification=False) if vit_transformer is None else vit_transformer

        self.vit4 = ViT(img_dim=self.vit_image_sizes[3],  #
                        in_channels=self.vit_channels[3],  # 輸入的channel數
                        patch_dim=self.vit_patch_size[3],
                        dim=dim,  # embedding的維度
                        blocks=blocks,
                        heads=heads,
                        dim_linear_block=self.vit_dim[3],
                        classification=False) if vit_transformer is None else vit_transformer
        self.up1 = U_up(2048, 1024)
        self.up2 = U_up(2048, 1024)
        self.up3 = U_up(2048, 1024)
        self.up4 = U_up(2048, 1024)
        self.out1 = outconv(1024, classes)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.resnet_part1(x)
        x2 = self.resnet_part2(x)
        x3 = self.resnet_part3(x)
        x4 = self.resnet_part4(x)
        y1 = self.vit1(x1)
        y1 = rearrange(y1,
                       'b (x y) dim -> b dim x y ',
                       x=int(self.vit_image_sizes[0] / self.vit_patch_size[0]),
                       y=int(self.vit_image_sizes[0] / self.vit_patch_size[0]))
        y2 = self.vit2(x2)
        y2 = rearrange(y2,
                       'b (x y) dim -> b dim x y ',
                       x=int(self.vit_image_sizes[1] / self.vit_patch_size[1]),
                       y=int(self.vit_image_sizes[1] / self.vit_patch_size[1]))
        y3 = self.vit3(x3)
        y3 = rearrange(y3,
                       'b (x y) dim -> b dim x y ',
                       x=int(self.vit_image_sizes[2] / self.vit_patch_size[2]),
                       y=int(self.vit_image_sizes[2] / self.vit_patch_size[2]))
        y4 = self.vit4(x4)
        y4 = rearrange(y4,
                       'b (x y) dim -> b dim x y ',
                       x=int(self.vit_image_sizes[3] / self.vit_patch_size[3]),
                       y=int(self.vit_image_sizes[3] / self.vit_patch_size[3]))
        x = self.up1(y4, y3)
        x = self.up2(x, y2)
        x = self.up3(x, y1)
        x = self.up(x)
        x = self.up(x)
        x = self.up(x)
        x = self.out1(x)
        x = rearrange(x,
                      'b c x y -> (b c) x y')
        return x


class U_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class U_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(U_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = U_double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == "__main__":
    print("main.......")
    # net = ViT(img_dim=256,
    #           patch_dim=1,
    #           in_channels=1024,
    #           dim=1024,
    #           blocks=6,
    #           heads=16,
    #           dim_linear_block=2048,
    #           dropout=0.1,
    #           classification=False)
    #
    # summary(net, input_size=(1, 1024, 16, 16), col_names=("input_size", "output_size", "num_params"))
    # img = torch.randn(1, 1024, 16, 16)
    # pred = net(img)
    # print(pred.size())
    # net2 = TransUnet(in_channels=3, img_dim=224, vit_blocks=6,
    #                  vit_dim_linear_mhsa_block=2048, classes=1)
    # summary(net2, input_size=(1, 3, 224, 224), col_names=("input_size", "output_size", "num_params"))

    # resnet50 = models.resnet50(pretrained=True)
    # new_model1 = nn.Sequential(*list(resnet50.children())[:5])
    # summary(new_model1, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size", "num_params"))
    # new_model2 = nn.Sequential(*list(resnet50.children())[:6])
    # summary(new_model2, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size", "num_params"))
    # new_model3 = nn.Sequential(*list(resnet50.children())[:7])
    # summary(new_model3, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size", "num_params"))
    # new_model4 = nn.Sequential(*list(resnet50.children())[:8])
    # summary(new_model4, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size", "num_params"))

    # net = TransUnet(in_channels=3, img_dim=256, vit_blocks=6,
    #                 vit_dim_linear_mhsa_block=1024, classes=1)
    # summary(net, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))

    net = Transforensic(image_size=256)
    # summary(net, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))

    img = torch.randn(1, 3, 256, 256)
    net(img)
    # img_patches = rearrange(img,
    #                         'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
    #                         patch_x=1, patch_y=1)
    # print(img_patches.size())

    # resnet50 = models.resnet50(pretrained=True)
    # summary(resnet50, input_size=(1, 3, 256, 256), col_names=("input_size", "output_size"))
