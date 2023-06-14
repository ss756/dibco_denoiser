# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout= 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim = 2)
    def forward(self, q, k, v, scale = None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention , v)
        return context, attention



class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction = 16 , bias = False):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(channel, channel // reduction , 1, padding = 0, bias = bias)
        self.conv2 = nn.Conv2d(channel, channel// reduction, 1, padding = 0, bias = bias)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # print('in channel attention layer .. ')
        # print("[info] the input shape is", x.shape)
        y = self.avg_pool(x)
        # print("[info] the shape after average pooling is", y.shape)
        y1 = self.conv1(x)
        # print("[info] the shape after conv1", y1.shape)
        y1 = self.conv2(y)
        # print("[info] the shape after conv2", y1.shape)
        y1 = self.sigmoid(y)
        # print("[info] the shape after sigmoid", y1.shape)
        return y * y1
    





class channelAttention(nn.Module):
    def __init__(self, input_channels, kernel_size = 1, bias = False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size, stride = 1, padding = kernel_size//2,  bias = False)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size, stride = 1, padding = kernel_size//2,  bias = False)
        self.CA = ChannelAttentionLayer(input_channels)
    def forward(self, x):
        # print("[info] THE INPUT SHAPE IS", x.shape)
        y = self.conv1(x)
        y = self.act(x)
        y = self.conv2(y)
        y= self.CA(y)
        # print("[info] THE FINAL SHAPE IS", y.shape )
        return y

    






class Model(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels *2, kernel_size = 3, stride = 1, padding = 1)
        self.dropout = nn.Dropout(0.5)
        channels = input_channels * 2
        self.conv2 = nn.Conv2d(channels  , channels * 2, kernel_size = 3, stride = 2, padding = 1)
        channels = channels * 2
        self.conv3 = nn.Conv2d(channels, channels * 2, kernel_size = 3, stride = 2, padding = 1)
        channels = channels * 2 
        self.conv4 = nn.Conv2d(channels  ,channels * 2, kernel_size = 3, stride = 2, padding = 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.selu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.selu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = F.selu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = F.selu(x)
        x = self.dropout(x)
        return x


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.skipconnections = Model(width)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        sc = self.skipconnections(x)
        # print("[info] the shape of the skip connection is ", sc.shape)
        skipconnections1 =  channelAttention(input_channels = sc.shape[1])
        sc1 = skipconnections1(sc)

        
        # print("[info] the shape of the skip connection afterwards is", sc1.shape)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        # print('[info] the shape to the middle latent layer is', x.shape)
        # print("[info] the shape of the skip connection is ", sc.shape)
        x += sc1

        # print("[info] the shape of the input tensor before the middle block is", x.shape)
        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip

            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    import time
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)
    # input = torch.randn(4, 3, 400, 400)
    # now_time = time.time()
    # y = net(input)
    # print(y.size())
    # print("[info] the time taken is", time.time() - now_time)
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
