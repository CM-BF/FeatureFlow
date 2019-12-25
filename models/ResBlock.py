import torch.nn as nn


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = []

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
            #                              output_padding=1),
            #           norm_layer(int(ngf * mult / 2)), activation]
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, inchannel, padding_type='reflect', activation=nn.LeakyReLU(negative_slope=0.1, inplace=True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(inchannel, padding_type, activation, use_dropout)

    def build_conv_block(self, inchannel, padding_type, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=p),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=p), activation]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetBlock_bn(nn.Module):
    def __init__(self, inchannel, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(negative_slope=0.1, inplace=True), use_dropout=False):
        super(ResnetBlock_bn, self).__init__()
        self.conv_block = self.build_conv_block(inchannel, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, inchannel, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=p),
                       norm_layer(inchannel),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=p),
                       norm_layer(inchannel)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SemiResnetBlock(nn.Module):
    def __init__(self, inchannel, outchannel, padding_type='reflect', activation=nn.LeakyReLU(negative_slope=0.1, inplace=True), use_dropout=False, end=False):
        super(SemiResnetBlock, self).__init__()
        self.conv_block1 = self.build_conv_block1(inchannel, padding_type, activation, use_dropout)
        self.conv_block2 = self.build_conv_block2(inchannel, outchannel, padding_type, activation,
                                                  use_dropout, end)

    def build_conv_block1(self, inchannel, padding_type, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=p),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def build_conv_block2(self, inchannel, outchannel, padding_type, activation, use_dropout, end):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if end:
            conv_block += [nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=p)]
        else:
            conv_block += [nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=p),
                       activation]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        x = self.conv_block1(x) + x
        out = self.conv_block2(x)
        return out

class SemiResnetBlock_bn(nn.Module):
    def __init__(self, inchannel, outchannel, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU(negative_slope=0.1, inplace=True), use_dropout=False, end=False):
        super(SemiResnetBlock_bn, self).__init__()
        self.conv_block1 = self.build_conv_block1(inchannel, padding_type, norm_layer, activation, use_dropout)
        self.conv_block2 = self.build_conv_block2(inchannel, outchannel, padding_type, norm_layer, activation,
                                                  use_dropout, end)

    def build_conv_block1(self, inchannel, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=p),
                       norm_layer(inchannel),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def build_conv_block2(self, inchannel, outchannel, padding_type, norm_layer, activation, use_dropout, end):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if end:
            conv_block += [nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=p)]
        else:
            conv_block += [nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=p),
                       norm_layer(outchannel), activation]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        x = self.conv_block1(x) + x
        out = self.conv_block2(x)
        return out
