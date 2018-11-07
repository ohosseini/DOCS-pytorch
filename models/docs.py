import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .correlation_package.correlation import Correlation

#############################################################
# DOCS Encoder
#
class DOCSEncoderNet(nn.Module):
    def __init__(self, features):
        super(DOCSEncoderNet, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)

encoder_archs = {
    'vgg16-based-16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 1024, 1024]
}

def make_encoder_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    output_scale = 1.0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            output_scale /= 2.0
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2,inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.2,inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), in_channels, output_scale

#############################################################

#############################################################
# DOCS Decoder
#
class DOCSDecoderNet(nn.Module):
    def __init__(self, features):
        super(DOCSDecoderNet, self).__init__()
        self.features = features

    def forward(self, x):
        return self.features(x)

decoder_archs = {
    'd16': [1024, 'd512', 512, 512, 'd512', 512, 512, 'd256', 256, 256, 'd128', 128, 128, 'd64', 64, 64, 'c2']
}

def make_decoder_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if type(v) is str:
            if v[0] == 'd':
                v = int(v[1:])
                convtrans2d = nn.ConvTranspose2d(in_channels, v, kernel_size=4, stride=2, padding=1)
                if batch_norm:
                    layers += [convtrans2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                else:
                    layers += [convtrans2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
                in_channels = v
            elif v[0] == 'c':
                v = int(v[1:])
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            else:
                layers += [conv2d, nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#############################################################


#############################################################
# DOCS Network
#
class DOCSNet(nn.Module):
    '''DOCSNet a Siamese Encoder-Decoder for Object Co-segmentation.'''
    def __init__(self, input_size=512, init_weights=True, batch_norm=False, 
                 en_arch='vgg16-based-16', de_arch='d16',
                 has_squeez=True, squeezed_out_channels=512):
        super(DOCSNet, self).__init__()

        self.en_arch = en_arch
        self.de_arch = de_arch

        en_layers, en_out_channels, en_output_scale = make_encoder_layers(encoder_archs[en_arch], batch_norm)
        self.encoder = DOCSEncoderNet(en_layers)
        en_output_size = round(input_size * en_output_scale)

        disp = en_output_size-1
        self.corr = Correlation(pad_size=disp, kernel_size=1, max_displacement=disp, stride1=1, stride2=1)
        corr_out_channels = self.corr.out_channels

        self.has_squeez = has_squeez
        if has_squeez:
            self.conv_squeezed = nn.Conv2d(en_out_channels, squeezed_out_channels, 1, padding=0)
            de_in_channels = int(squeezed_out_channels + corr_out_channels)
        else:
            de_in_channels = int(en_out_channels + corr_out_channels)

        de_layers = make_decoder_layers(decoder_archs[de_arch], de_in_channels, batch_norm)
        self.decoder = DOCSDecoderNet(de_layers)

        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_a, img_b, softmax_out=False):
        self.en_a = self.encoder(img_a)
        self.en_b = self.encoder(img_b)

        self.corr_ab = self.corr(self.en_a, self.en_b)
        self.corr_ba = self.corr(self.en_b, self.en_a)

        if self.has_squeez:
            cat_a = torch.cat((self.conv_squeezed(self.en_a), self.corr_ab),dim=1)
            cat_b = torch.cat((self.conv_squeezed(self.en_b), self.corr_ba),dim=1)
        else:
            cat_a = torch.cat((self.en_a, self.corr_ab),dim=1)
            cat_b = torch.cat((self.en_b, self.corr_ba),dim=1)

        self.out_a = self.decoder(cat_a)
        self.out_b = self.decoder(cat_b)

        if softmax_out:
            self.out_a = torch.softmax(self.out_a, 1)
            self.out_b = torch.softmax(self.out_b, 1)

        return self.out_a, self.out_b

#############################################################
