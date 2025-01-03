import torch
import torch.nn as nn
from timm.models import create_model
model_name_list = ["repvit_m0_9","repvit_m1_0","repvit_m1_1","repvit_m1_5","repvit_m2_3"]
model_name_list = ["mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k","mobilenetv4_conv_aa_large.e600_r384_in1k","mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k","mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k",
                   "mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k","mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k"]

def get_layer_dims(model,image_size=(3,3,224,224)):
    input = torch.randn(image_size)
    out = model(input)
    out_dims=[]
    for i in out:
        out_dims.append(i.shape[1])
    print(out_dims)
    return out_dims

class DwConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act
        self.conv1x1 = nn.Conv2d(in_c,out_c,kernel_size=1,padding=0,stride=1,bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(
                out_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=out_c,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inx = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*inx


class Fusion(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(Fusion, self).__init__()
        self.out_channels = out_channels
        assert in_channels % 4 == 0
        hidden_dim = in_channels // 4
        self.dw_conv3x3 = DwConv(in_channels,hidden_dim,kernel_size=3,padding=1)
        self.dw_conv5x5 = DwConv(in_channels,hidden_dim,kernel_size=5,padding=2)
        self.dw_conv9x9 = DwConv(in_channels,hidden_dim,kernel_size=7,padding=3)
        self.dw_conv11x11 = DwConv(in_channels,hidden_dim,kernel_size=11,padding=5)
        self.fuse1 = DwConv(in_channels,out_channels,kernel_size=1,padding=0)
        assert in_channels % 4 == 0
        hidden_dim = out_channels // 4
        self.dw_conv3x3_r1 = DwConv(out_channels,hidden_dim,kernel_size=3,padding=1,dilation=1)
        self.dw_conv3x3_r3 = DwConv(out_channels,hidden_dim,kernel_size=3,padding=3,dilation=3)
        self.dw_conv3x3_r7 = DwConv(out_channels,hidden_dim,kernel_size=3,padding=7,dilation=7)
        self.dw_conv3x3_r11 = DwConv(out_channels,hidden_dim,kernel_size=3,padding=11,dilation=11)
        self.fuse = DwConv(out_channels, out_channels, kernel_size=1, padding=0,act=False)

    def forward(self, x):
        x1 = self.dw_conv3x3(x)
        x2 = self.dw_conv5x5(x)
        x3 = self.dw_conv9x9(x)
        x4 = self.dw_conv11x11(x)
        x = x + self.fuse1(torch.cat([x1,x2,x3,x4],dim=1))
        x5 = self.dw_conv3x3_r1(x)
        x6 = self.dw_conv3x3_r3(x)
        x7 = self.dw_conv3x3_r7(x)
        x8 = self.dw_conv3x3_r11(x)
        x = x + self.fuse(torch.cat([x5,x6,x7,x8],dim=1))
        return x

class E_SegNet_2D(nn.Module):
    def __init__(
        self,
        ch=64,
        pretrained=True,
        freeze_encoder=False,
        model_name="",
        image_size = 224,
        num_classes = 9
    ):
        super(E_SegNet_2D, self).__init__()
        self.model_name = model_name
        self.encoder = create_model(model_name,pretrained=pretrained, features_only=True)
        channels_per_output = get_layer_dims(self.encoder)
        print(f"model_name : {model_name}")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        upsampled_size = image_size
        self.up1 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up2 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up3 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up4 = nn.Upsample(size=upsampled_size, mode="nearest")
        self.up5 = nn.Upsample(size=upsampled_size, mode="nearest")

        self.conv1 = nn.Conv2d(
            channels_per_output[0], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(
            channels_per_output[1], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(
            channels_per_output[2], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(ch)

        self.conv4 = nn.Conv2d(
            channels_per_output[3], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(ch)

        self.conv5 = nn.Conv2d(
            channels_per_output[4], ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm2d(ch)
        self.conv1x1 = nn.Conv2d(ch*5,ch,kernel_size=1,stride=1,padding=0,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(ch)
        self.fusion = Fusion(ch, ch)
        self.conv6 = nn.Conv2d(ch, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1,3,1,1)
        x0, x1, x2, x3, x4 = self.encoder(x)
        # x0, x1, x2, x3 = self.encoder(x)

        x0 = self.conv1(x0)
        x0 = self.relu(x0)
        x0 = self.bn1(x0)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.bn3(x2)

        x3 = self.conv4(x3)
        x3 = self.relu(x3)
        x3 = self.bn4(x3)

        x4 = self.conv5(x4)
        x4 = self.relu(x4)
        x4 = self.bn5(x4)

        x0 = self.up1(x0)
        x1 = self.up2(x1)
        x2 = self.up3(x2)
        x3 = self.up4(x3)
        x4 = self.up5(x4)
        # x = self.conv1x1(torch.cat([x0,x1,x2,x3,x4],dim=1))

        x = x0 + x1 + x2 + x3 + x4
        x = self.bn6(x)
        x = self.fusion(x)
        x = self.conv6(x)
        return x



if __name__ == '__main__':
    model = E_SegNet_2D(model_name="mobilenetv4_conv_medium.e500_r224_in1k",image_size=384)
    input=torch.randn(1,3,384,384)
    out = model(input)
    print(out.shape)
