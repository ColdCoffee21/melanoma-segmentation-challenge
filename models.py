import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3_bn(ci, co):
  return torch.nn.Sequential(torch.nn.Conv2d(ci, co, 3, padding=1), torch.nn.BatchNorm2d(co), torch.nn.ReLU(inplace=True))

def encoder_conv(ci, co):
  return torch.nn.Sequential(torch.nn.MaxPool2d(2), conv3x3_bn(ci, co), conv3x3_bn(co, co))

class deconv(torch.nn.Module):
  def __init__(self, ci, co):
    super(deconv, self).__init__()
    self.upsample = torch.nn.ConvTranspose2d(ci, co, 2, stride=2)
    self.conv1 = conv3x3_bn(ci, co)
    self.conv2 = conv3x3_bn(co, co)

  def forward(self, x1, x2):
    x1 = self.upsample(x1)
    diffX = x2.size()[2] - x1.size()[2]
    diffY = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (diffX, 0, diffY, 0))
    # concatenating tensors
    x = torch.cat([x2, x1], dim=1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x

# class UNet(torch.nn.Module):
#   def __init__(self, n_classes=1, in_ch=3):
#     super().__init__()

#     # number of filter's list for each expanding and respecting contracting layer
#     c = [16, 32, 64, 128]

#     # first convolution layer receiving the image
#     self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
#                                      conv3x3_bn(c[0], c[0]))

#     # encoder layers
#     self.conv2 = encoder_conv(c[0], c[1])
#     self.conv3 = encoder_conv(c[1], c[2])
#     self.conv4 = encoder_conv(c[2], c[3])

#     # decoder layers
#     self.deconv1 = deconv(c[3],c[2])
#     self.deconv2 = deconv(c[2],c[1])
#     self.deconv3 = deconv(c[1],c[0])

#     # last layer returning the output
#     self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

#   def forward(self, x):
#     # encoder
#     x1 = self.conv1(x)
#     x2 = self.conv2(x1)
#     x3 = self.conv3(x2)
#     x = self.conv4(x3)
#     # decoder
#     x = self.deconv1(x, x3)
#     x = self.deconv2(x, x2)
#     x = self.deconv3(x, x1)
#     x = self.out(x)
#     return x
  
class DeeperUNet(torch.nn.Module):
  def __init__(self, n_classes=1, in_ch=3):
    super().__init__()

    # number of filter's list for each expanding and respecting contracting layer
    c = [8, 16, 32, 64, 128, 256]

    # first convolution layer receiving the image
    self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
                                     conv3x3_bn(c[0], c[0]))

    # encoder layers
    self.conv2 = encoder_conv(c[0], c[1])
    self.conv3 = encoder_conv(c[1], c[2])
    self.conv4 = encoder_conv(c[2], c[3])
    self.conv5 = encoder_conv(c[3], c[4])
    self.conv6 = encoder_conv(c[4], c[5])

    # decoder layers
    self.deconv1 = deconv(c[5],c[4])
    self.deconv2 = deconv(c[4],c[3])
    self.deconv3 = deconv(c[3],c[2])
    self.deconv4 = deconv(c[2],c[1])
    self.deconv5 = deconv(c[1],c[0])

    # last layer returning the output
    self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

  def forward(self, x):
    # encoder
    x1 = self.conv1(x)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x4 = self.conv4(x3)
    x5 = self.conv5(x4)
    x = self.conv6(x5)
    # decoder
    x = self.deconv1(x, x5)
    x = self.deconv2(x, x4)
    x = self.deconv3(x, x3)
    x = self.deconv4(x, x2)
    x = self.deconv5(x, x1)
    x = self.out(x)
    return x
  
# class CodeUNet(torch.nn.Module):
#   def __init__(self, n_classes=1, in_ch=3):
#     super().__init__()

#     # number of filter's list for each expanding and respecting contracting layer
#     c = [16, 32, 64, 128]

#     # first convolution layer receiving the image
#     self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
#                                      conv3x3_bn(c[0], c[0]))

#     # encoder layers
#     self.conv2 = encoder_conv(c[0], c[1])
#     self.conv3 = encoder_conv(c[1], c[2])
#     self.conv4 = encoder_conv(c[2], c[3])

#     # Middle Convolution
#     self.code_conv1 = torch.nn.Sequential(conv3x3_bn(c[3], c[3]),
#                                      conv3x3_bn(c[3], c[3]))
#     self.code_conv2 = torch.nn.Sequential(conv3x3_bn(c[3], c[3]),
#                                      conv3x3_bn(c[3], c[3]))

#     # decoder layers
#     self.deconv1 = deconv(c[3],c[2])
#     self.deconv2 = deconv(c[2],c[1])
#     self.deconv3 = deconv(c[1],c[0])

#     # last layer returning the output
#     self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

#   def forward(self, x):
#     # encoder
#     x1 = self.conv1(x)
#     x2 = self.conv2(x1)
#     x3 = self.conv3(x2)
#     x = self.conv4(x3)
#     # code
#     x = self.code_conv1(x)
#     x = self.code_conv2(x)
#     # decoder
#     x = self.deconv1(x, x3)
#     x = self.deconv2(x, x2)
#     x = self.deconv3(x, x1)
#     x = self.out(x)
#     return x
  
class CodeUNet(torch.nn.Module):
    def __init__(self, depth=4, code_size = 2, n_classes=1, in_ch=3):
        super().__init__()

        # number of filter's list for each expanding and respecting contracting layer
        c = [16, 32, 64, 128, 256, 512, 1024, 2048][:depth+1]

        # first convolution layer receiving the image
        self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
                                         conv3x3_bn(c[0], c[0]))

        # encoder layers
        self.encoders = torch.nn.ModuleList([encoder_conv(c[i], c[i+1]) for i in range(depth)])

        # Middle Convolution
        self.code_convs = torch.nn.ModuleList([torch.nn.Sequential(conv3x3_bn(c[-1], c[-1]),
                                                                   conv3x3_bn(c[-1], c[-1])) for _ in range(code_size)])

        # decoder layers
        self.decoders = torch.nn.ModuleList([deconv(c[i+1], c[i]) for i in reversed(range(depth))])

        # last layer returning the output
        self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

    def forward(self, x):
        # encoder
        enc_outs = [self.conv1(x)]
        for encoder in self.encoders:
            enc_outs.append(encoder(enc_outs[-1]))

        # code
        for code_conv in self.code_convs:
            enc_outs[-1] = code_conv(enc_outs[-1])

        # decoder
        x = enc_outs[-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, enc_outs[-(i+2)])

        x = self.out(x)
        return x

# class ResidUNet(torch.nn.Module):
#     def __init__(self, n_classes=1, in_ch=3):
#         super().__init__()

#         # number of filter's list for each expanding and respecting contracting layer
#         c = [16, 32, 64, 128]

#         # first convolution layer receiving the image
#         self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
#                                          conv3x3_bn(c[0], c[0]))

#         # encoder layers
#         self.conv2 = encoder_conv(c[0], c[1])
#         self.conv3 = encoder_conv(c[1], c[2])
#         self.conv4 = encoder_conv(c[2], c[3])

#         # decoder layers
#         self.deconv1 = deconv(c[3],c[2])
#         self.deconv2 = deconv(c[2],c[1])
#         self.deconv3 = deconv(c[1],c[0])

#         # last layer returning the output
#         self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

#     def forward(self, x):
#         # encoder
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x = self.conv4(x3)

#         # decoder with residual connections
#         x = self.deconv1(x, x3) + x3
#         x = self.deconv2(x, x2) + x2
#         x = self.deconv3(x, x1) + x1

#         x = self.out(x)
#         return x
    
class ResidUNet(torch.nn.Module):
    def __init__(self, depth=4, n_classes=1, in_ch=3):
        super().__init__()

        # number of filter's list for each expanding and respecting contracting layer
        c = [16, 32, 64, 128, 256, 512, 1024, 2048][:depth+1]

        # first convolution layer receiving the image
        self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
                                         conv3x3_bn(c[0], c[0]))

        # encoder layers
        self.encoders = torch.nn.ModuleList([encoder_conv(c[i], c[i+1]) for i in range(depth)])

        # decoder layers
        self.decoders = torch.nn.ModuleList([deconv(c[i+1], c[i]) for i in reversed(range(depth))])

        # last layer returning the output
        self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

    def forward(self, x):
        # encoder
        enc_outs = [self.conv1(x)]
        for encoder in self.encoders:
            enc_outs.append(encoder(enc_outs[-1]))

        # decoder with residual connections
        x = enc_outs[-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, enc_outs[-(i+2)]) + enc_outs[-(i+2)]

        x = self.out(x)
        return x
    

class AttentionGate(torch.nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = torch.nn.Sequential(
            torch.nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )

        self.W_x = torch.nn.Sequential(
            torch.nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(F_int)
        )

        self.psi = torch.nn.Sequential(
            torch.nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class AttentionUNet(nn.Module):
    def __init__(self, depth=4, img_ch=3, output_ch=1):
        super(AttentionUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Depth 6 is 1024 here
        # number of filter's list for each expanding and respecting contracting layer
        c = [64, 128, 256, 512, 1024, 2048, 4096, 8192][:depth]
        depth -= 1
        self.init_conv = conv_block(ch_in=img_ch, ch_out=c[0])

        # encoder layers
        self.encoders = nn.ModuleList([conv_block(ch_in=c[i], ch_out=c[i+1]) for i in range(depth)])

        # attention gates and decoder layers
        self.decoders = nn.ModuleList([up_conv(ch_in=c[i+1], ch_out=c[i]) for i in reversed(range(depth))])
        self.attentions = nn.ModuleList([AttentionGate(F_g=c[i], F_l=c[i], F_int=c[i-1]) for i in reversed(range(depth))])
        self.up_convs = nn.ModuleList([conv_block(ch_in=c[i+1], ch_out=c[i]) for i in reversed(range(depth))])

        self.Conv_1x1 = nn.Conv2d(c[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.init_conv(x)
        # encoding path
        enc_outs = [x]
        for encoder in self.encoders:
            enc_outs.append(self.Maxpool(enc_outs[-1]))
            enc_outs[-1] = encoder(enc_outs[-1])
        
        # decoding + concat path
        d = enc_outs[-1]
        for i, (decoder, attention, up_conv) in enumerate(zip(self.decoders, self.attentions, self.up_convs)):
            d = decoder(d)
            att = attention(g=d, x=enc_outs[-(i+2)])
            d = torch.cat((att, d), dim=1)
            d = up_conv(d)

        d = self.Conv_1x1(d)

        return d

# Flexible Unet
class UNet(torch.nn.Module):
    def __init__(self, depth=4, n_classes=1, in_ch=3):
        super().__init__()
        # Depth 4 is 128
        # number of filter's list for each expanding and respecting contracting layer
        c = [16, 32, 64, 128, 256, 512, 1024, 2048][:depth+1]

        # first convolution layer receiving the image
        self.conv1 = torch.nn.Sequential(conv3x3_bn(in_ch, c[0]),
                                         conv3x3_bn(c[0], c[0]))

        # encoder layers
        self.encoders = torch.nn.ModuleList([encoder_conv(c[i], c[i+1]) for i in range(depth)])

        # decoder layers
        self.decoders = torch.nn.ModuleList([deconv(c[i+1], c[i]) for i in reversed(range(depth))])

        # last layer returning the output
        self.out = torch.nn.Conv2d(c[0], n_classes, 3, padding=1)

    def forward(self, x):
        # encoder
        enc_outs = [self.conv1(x)]
        for encoder in self.encoders:
            enc_outs.append(encoder(enc_outs[-1]))

        # decoder
        x = enc_outs[-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, enc_outs[-(i+2)])

        x = self.out(x)
        return x