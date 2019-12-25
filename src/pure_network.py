import src.model as model
import torch.nn as nn
import torch
from models.ResBlock import GlobalGenerator


class StructureGen(nn.Module):

    def __init__(self, feature_level=3):
        super(StructureGen, self).__init__()

        self.feature_level = feature_level
        channel = 2 ** (6 + self.feature_level)
        self.extract_features = model.ExtractFeatures()
        self.dcn = model.DeformableConv(channel, dg=16)
        self.generator = GlobalGenerator(channel, 4, n_downsampling=self.feature_level)

    def forward(self, input):
        img0_e, img1_e= input

        ft_img0 = list(self.extract_features(img0_e))[self.feature_level]
        ft_img1 = list(self.extract_features(img1_e))[self.feature_level]

        pre_gen_ft_imgt, out_x, out_y = self.dcn(ft_img0, ft_img1)

        ref_imgt_e = self.generator(pre_gen_ft_imgt)

        # divide edge
        ref_imgt, edge_ref_imgt = ref_imgt_e[:, :3], ref_imgt_e[:, 3:]

        return ref_imgt, edge_ref_imgt



class DetailEnhance(nn.Module):


    def __init__(self):

        super(DetailEnhance, self).__init__()

        self.feature_level = 3

        self.extract_features = model.ValidationFeatures()
        self.extract_aligned_features = model.ExtractAlignedFeatures(n_res=5) # 4  5
        self.pcd_align = model.PCD_Align(groups=8) # 4  8
        self.tsa_fusion = model.TSA_Fusion(nframes=3, center=1)

        self.reconstruct = model.Reconstruct(n_res=20) # 5  40

    def forward(self, input):
        """
        Network forward tensor flow

        :param input: a tuple of input that will be unfolded
        :return: medium interpolation image
        """
        img0, img1, ref_imgt = input

        ref_align_ft = self.extract_aligned_features(ref_imgt)
        align_ft_0 = self.extract_aligned_features(img0)
        align_ft_1 = self.extract_aligned_features(img1)

        align_ft = [self.pcd_align(align_ft_0, ref_align_ft),
                    self.pcd_align(ref_align_ft, ref_align_ft),
                    self.pcd_align(align_ft_1, ref_align_ft)]
        align_ft = torch.stack(align_ft, dim=1)

        tsa_ft = self.tsa_fusion(align_ft)

        imgt = self.reconstruct(tsa_ft, ref_imgt)

        return imgt
