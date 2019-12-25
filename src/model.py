import torch
import torchvision
import torch.nn as nn
from models.ResBlock import SemiResnetBlock_bn, ResnetBlock_bn, SemiResnetBlock, ResnetBlock
from mmdet.ops.dcn.deform_conv import DeformConv, DCN_sep
import torch.nn.functional as F


class ReVggBlock(nn.Module):

    def __init__(self, inchannel, outchannel, upsampling=False, end=False):
        """
        Reverse Vgg19_bn block
        :param inchannel: input channel
        :param outchannel: output channel
        :param upsampling: judge for adding upsampling module
        :param padding: padding mode: 'zero', 'reflect', by default:'reflect'
        """
        super(ReVggBlock, self).__init__()

        model = []
        model += [nn.ReplicationPad2d(1)]
        model += [nn.Conv2d(inchannel, outchannel, 3)]

        if upsampling:
            model += [nn.UpsamplingBilinear2d(scale_factor=2)]

        if not end:
            model += [nn.LeakyReLU(True), nn.BatchNorm2d(outchannel)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ReVgg19(nn.Module):

    def __init__(self):
        """
        A reverse operation network for vgg19_bn
        """
        super(ReVgg19, self).__init__()

        self.dcn0 = DeformableConv(3)
        self.dcn1 = DeformableConv(64, 2)
        self.dcn2 = DeformableConv(128, 2)
        self.dcn3 = DeformableConv(256, 2)
        self.dcn4 = DeformableConv(512, 2)

        self.model_4 = nn.Sequential(ReVggBlock(512, 512), ReVggBlock(512, 512), ReVggBlock(512, 256, True, True))
        self.model_3 = nn.Sequential(ReVggBlock(256, 256), ReVggBlock(256, 256), ReVggBlock(256, 128, True, True))
        self.model_2 = nn.Sequential(ReVggBlock(128, 128), ReVggBlock(128, 64, True, True))
        self.model_1 = [ReVggBlock(64, 64)]

        self.model_1 += [nn.ReplicationPad2d(1)]
        self.model_1 += [nn.Conv2d(64, 64, 3)]

        self.model_1 = nn.Sequential(*self.model_1)
        self.model_0 = nn.Sequential(SemiResnetBlock_bn(64, 64), ResnetBlock_bn(64), ResnetBlock_bn(64), SemiResnetBlock_bn(64, 3, end=True), nn.Tanh())

        self.offset_tran = nn.Sequential(ReVggBlock(128, 128), ReVggBlock(128, 12, end=True))

    def forward(self, ft_img0, ft_img1):
        ft4, offset, _, _ = self.dcn4(ft_img0[4], ft_img1[4])
        out4 = self.model_4(ft4)
        ft3, offset, _, _ = self.dcn3(ft_img0[3], ft_img1[3], last_offset=offset, last_up_out=out4)
        out3 = self.model_3(ft3)
        ft2, offset, _, _ = self.dcn2(ft_img0[2], ft_img1[2], last_offset=offset, last_up_out=out3)
        out2 = self.model_2(ft2)
        ft1, _, last, test = self.dcn1(ft_img0[1], ft_img1[1], last_offset=offset, last_up_out=out2)
        out1 = self.model_1(ft1)
        # offset = self.offset_tran(offset)
        # ft0, offset = self.dcn0(torch.cat([ft_img0[0], ft_img1[0]], dim=1), last_offset=offset, last_up_out=out1, up=False)
        imgt = self.model_0(out1)
        return imgt, [0, ft1, ft2, ft3, ft4], last, test


class ExtractFeatures(nn.Module):

    def __init__(self):
        super(ExtractFeatures, self).__init__()

        self.net = torchvision.models.resnet50(pretrained=True)
        self.conv1_4in = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.xavier_uniform(self.conv1_4in.weight)


    def forward(self, x):
        output = self.conv1_4in(x)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        ft1 = self.net.layer1(output)
        ft2 = self.net.layer2(ft1)
        ft3 = self.net.layer3(ft2)

        return 0, 0, ft1, ft2, ft3


class ValidationFeatures(nn.Module):

    def __init__(self):
        super(ValidationFeatures, self).__init__()

        vgg = torchvision.models.vgg16_bn(pretrained=True)

        self.extract_feature1 = nn.Sequential(*list(vgg.features.children())[:4])
        for param in self.extract_feature1.parameters(True):
            param.requires_grad = False

        self.extract_feature2 = nn.Sequential(*list(vgg.features.children())[4:11])
        for param in self.extract_feature2.parameters(True):
            param.requires_grad = False

        self.extract_feature3 = nn.Sequential(*list(vgg.features.children())[11:21])
        for param in self.extract_feature3.parameters(True):
            param.requires_grad = False

        self.extract_feature4 = nn.Sequential(*list(vgg.features.children())[21:31])
        for param in self.extract_feature4.parameters(True):
            param.requires_grad = False

    def forward(self, x):
        ft_1 = self.extract_feature1(x)
        ft_2 = self.extract_feature2(ft_1)
        ft_3 = self.extract_feature3(ft_2)
        ft_4 = self.extract_feature4(ft_3)

        return ft_1, ft_2, ft_3, ft_4


class StructureExtractor(nn.Module):

    def __init__(self):
        super(StructureExtractor, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)

        self.extract_feature1 = nn.Sequential(*list(vgg.features.children())[:4])
        for param in self.extract_feature1.parameters(True):
            param.requires_grad = False

        self.ap = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        edge = self.extract_feature1(x)
        ap_edge1 = self.ap(edge)
        ap_edge2 = self.ap(ap_edge1)
        mp_edge1 = self.mp(edge)
        mp_edge2 = self.mp(mp_edge1)

        return ap_edge1, ap_edge2, mp_edge1, mp_edge2


class DeformableConv(nn.Module):

    def __init__(self, inchannel, dg=2):
        super(DeformableConv, self).__init__()
        self.dg = dg

        self.offset_cnn1 = nn.Sequential(nn.Conv2d(2 * inchannel, inchannel, 3, padding=1), nn.BatchNorm2d(inchannel), nn.LeakyReLU(True))
        self.offset_cnn2 = nn.Sequential(nn.Conv2d(2 * 2 * 2 * dg * 9, 2 * 2 * dg * 9, 3, padding=1), nn.BatchNorm2d(2 * 2 * dg * 9), nn.LeakyReLU(True))
        self.offset_cnn3 = nn.Sequential(*([ResnetBlock_bn(inchannel)] * 5 + [ResnetBlock(inchannel)] * 3 + [nn.Conv2d(inchannel, 2 * 2 * dg * 9, 3, padding=1)]))

        self.emb = nn.Conv2d(inchannel, inchannel, 3, padding=1)
        self.mix_map = nn.Sequential(nn.Conv2d(2 * inchannel, inchannel, 3, padding=1), nn.LeakyReLU(True), *([ResnetBlock(inchannel)] * 3), nn.Conv2d(inchannel, 2 * dg, 3, padding=1))

        self.dcn = DeformConv(inchannel, inchannel, 3, padding=1, deformable_groups=dg)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x, y, last_offset=None, up=True):
        offset = None
        if last_offset is not None:
            if up:
                last_offset = self.up(last_offset)
            offset = self.offset_cnn1(torch.cat([x, y], dim=1))
            offset = self.offset_cnn2(torch.cat([offset, last_offset * 2], dim=1))
            offset_com = self.offset_cnn3(offset)
        else:
            offset = self.offset_cnn1(torch.cat([x, y], dim=1))
            offset = self.offset_cnn3(offset)
        offset_x, offset_y = torch.chunk(offset, 2, dim=1)
        out_x = self.dcn(x, offset_x)
        out_y = self.dcn(y, offset_y)
        vmap_x, vmap_y = torch.chunk(torch.sigmoid(self.mix_map(torch.cat([self.emb(out_x), self.emb(out_y)], dim=1))), 2, dim=1)
        vmap_x = torch.chunk(vmap_x, self.dg, dim=1)
        vmap_y = torch.chunk(vmap_y, self.dg, dim=1)
        out_x_d = torch.chunk(out_x, self.dg, dim=1)
        out_y_d = torch.chunk(out_y, self.dg, dim=1)
        out = [vmap_x[i] * out_x_d[i] + vmap_y[i] * out_y_d[i] for i in range(self.dg)]
        out = torch.cat(out, dim=1)

        return out, out_x, out_y


class ExtractAlignedFeatures(nn.Module):
    """
    Extract features
    """

    def __init__(self, nf=64, n_res=5):
        super(ExtractAlignedFeatures, self).__init__()

        self.deblur = Predeblur_ResNet_Pyramid(nf=nf)
        self.fea_L1_conv = nn.Sequential(*([ResnetBlock(nf)] * n_res))
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):

        ft_L1 = self.fea_L1_conv(self.deblur(x))
        ft_L2 = self.lrelu(self.fea_L2_conv2(self.fea_L2_conv1(ft_L1)))
        ft_L3 = self.lrelu(self.fea_L3_conv2(self.fea_L3_conv1(ft_L2)))

        return [ft_L1, ft_L2, ft_L3]



class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))
        # Denoise
        # L1_fea = self.lrelu(self.cas_offset_conv2(L1_fea))
        # L1_fea = self.cas_offset_conv2(L1_fea)

        return L1_fea


class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class Reconstruct(nn.Module):

    def __init__(self, nf=64, n_res=10):
        super(Reconstruct, self).__init__()

        #### reconstruction
        self.recon_trunk = nn.Sequential(*([ResnetBlock(nf)] * n_res))
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.down = nn.AvgPool2d(2, 2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, fea, x_center):

        out = self.recon_trunk(fea)
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        out += x_center
        return out

class Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=128, HR_in=False):
        '''
        HR_in: True if the inputs are high spatial size
        '''

        super(Predeblur_ResNet_Pyramid, self).__init__()
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.RB_L1_1 = ResnetBlock(nf)
        self.RB_L1_2 = ResnetBlock(nf)
        self.RB_L1_3 = ResnetBlock(nf)
        self.RB_L1_4 = ResnetBlock(nf)
        self.RB_L1_5 = ResnetBlock(nf)
        self.RB_L2_1 = ResnetBlock(nf)
        self.RB_L2_2 = ResnetBlock(nf)
        self.RB_L3_1 = ResnetBlock(nf)
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out
