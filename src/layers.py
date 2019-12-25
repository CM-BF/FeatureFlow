import src.model as model
import src.loss as loss
import torch.nn as nn
import torch
from math import log10
from models.ResBlock import GlobalGenerator


class StructureGen(nn.Module):

    def __init__(self, feature_level=3):
        super(StructureGen, self).__init__()

        self.feature_level = feature_level
        channel = 2 ** (6 + self.feature_level)
        # self.structure_extractor = model.StructureExtractor()
        self.extract_features = model.ExtractFeatures()
        self.dcn = model.DeformableConv(channel, dg=16)
        self.generator = GlobalGenerator(channel, 4, n_downsampling=self.feature_level)

        # Loss calculate
        self.L1_lossFn = nn.L1Loss()
        self.sL1_lossFn = nn.SmoothL1Loss()
        self.cL1_lossFn = loss.CharbonnierLoss()
        self.MSE_LossFn = nn.MSELoss()

    def forward(self, input):
        img0_e, img1_e, IFrame_e = input

        ft_img0 = list(self.extract_features(img0_e))[self.feature_level]
        ft_img1 = list(self.extract_features(img1_e))[self.feature_level]

        pre_gen_ft_imgt, out_x, out_y = self.dcn(ft_img0, ft_img1)

        ref_imgt_e = self.generator(pre_gen_ft_imgt)

        # divide edge
        IFrame, edge_IFrame = IFrame_e[:, :3], IFrame_e[:, 3:]
        ref_imgt, edge_ref_imgt = ref_imgt_e[:, :3], ref_imgt_e[:, 3:]

        # extract for loss
        ft_IFrame = list(self.extract_features(IFrame_e))[self.feature_level]
        ft_ref_imgt = list(self.extract_features(ref_imgt_e))[self.feature_level]
        # st_IFrame = self.structure_extractor(IFrame)
        # st_ref_imgt = self.structure_extractor(ref_imgt)

        # Loss calculate

        feature_mix_loss = 500 * self.cL1_lossFn(pre_gen_ft_imgt, ft_IFrame)

        tri_loss = 20 * (self.cL1_lossFn(out_x, ft_IFrame) + self.L1_lossFn(out_y, ft_IFrame))

        # feature_gen_loss = 10 * self.MSE_LossFn(ft_ref_imgt, ft_IFrame)

        # structure_loss = 20 * (self.L1_lossFn(st_ref_imgt[0], st_IFrame[0]) + self.L1_lossFn(st_ref_imgt[1], st_IFrame[1]) +
        #                         self.L1_lossFn(st_ref_imgt[2], st_IFrame[2]) + self.L1_lossFn(st_ref_imgt[3], st_IFrame[3]))

        edge_loss = 5 * self.MSE_LossFn(edge_ref_imgt, edge_IFrame)

        gen_loss = 128 * self.cL1_lossFn(ref_imgt, IFrame)

        loss = gen_loss + feature_mix_loss + edge_loss + tri_loss# + structure_loss + feature_gen_loss

        MSE_val = self.MSE_LossFn(ref_imgt, IFrame)

        print('Loss:', loss.item(), 'feature_mix_loss:', feature_mix_loss.item(),
              'gen_loss:', gen_loss.item(),
              # 'feature_gen_loss:', feature_gen_loss.item(),
              # 'structure_loss:', structure_loss.item(),
              'edge_loss:', edge_loss.item(),
              'tri_loss:', tri_loss.item())

        return loss, MSE_val, ref_imgt



class DetailEnhance(nn.Module):


    def __init__(self):

        super(DetailEnhance, self).__init__()

        self.feature_level = 3

        self.extract_features = model.ValidationFeatures()
        self.extract_aligned_features = model.ExtractAlignedFeatures(n_res=5) # 4  5
        self.pcd_align = model.PCD_Align(groups=8) # 4  8
        self.tsa_fusion = model.TSA_Fusion(nframes=3, center=1)

        self.reconstruct = model.Reconstruct(n_res=20) # 5  40

        # Loss calculate
        self.L1_lossFn = nn.L1Loss()
        self.sL1_lossFn = nn.SmoothL1Loss()
        self.cL1_lossFn = loss.CharbonnierLoss()
        self.MSE_LossFn = nn.MSELoss()

    def forward(self, input):
        """
        Network forward tensor flow

        :param input: a tuple of input that will be unfolded
        :return: medium interpolation image
        """
        img0, img1, IFrame, ref_imgt = input

        ref_align_ft = self.extract_aligned_features(ref_imgt)
        align_ft_0 = self.extract_aligned_features(img0)
        align_ft_1 = self.extract_aligned_features(img1)

        align_ft = [self.pcd_align(align_ft_0, ref_align_ft),
                    self.pcd_align(ref_align_ft, ref_align_ft),
                    self.pcd_align(align_ft_1, ref_align_ft)]
        align_ft = torch.stack(align_ft, dim=1)

        tsa_ft = self.tsa_fusion(align_ft)

        imgt = self.reconstruct(tsa_ft, ref_imgt)


        # extract for loss
        ft_IFrame = list(self.extract_features(IFrame))[self.feature_level]
        ft_imgt = list(self.extract_features(imgt))[self.feature_level]


        """
        ----------------------------------------------------------------------
        ======================================================================
        """

        # Loss calculate
        # feature_recn_loss = 10 * self.MSE_LossFn(ft_imgt, ft_IFrame)

        recn_loss = 128 * self.cL1_lossFn(imgt, IFrame)
        ie = 128 * self.L1_lossFn(imgt, IFrame)

        loss = recn_loss #+ feature_recn_loss

        MSE_val = self.MSE_LossFn(imgt, IFrame)
        psnr = (10 * log10(4 / MSE_val.item()))

        print('Loss:', loss.item(), 'psnr:', psnr,
              'recn_loss:', recn_loss.item(),
              # 'feature_recn_loss:', feature_recn_loss.item()
              )

        return loss, MSE_val, ie, imgt

class DetailEnhance_last(nn.Module):


    def __init__(self):

        super(DetailEnhance_last, self).__init__()

        self.feature_level = 3

        self.extract_features = model.ValidationFeatures()
        self.extract_aligned_features = model.ExtractAlignedFeatures(n_res=5) # 4  5
        self.pcd_align = model.PCD_Align(groups=8) # 4  8
        self.tsa_fusion = model.TSA_Fusion(nframes=3, center=1)

        self.reconstruct = model.Reconstruct(n_res=20) # 5  40

        # Loss calculate
        self.L1_lossFn = nn.L1Loss()
        self.sL1_lossFn = nn.SmoothL1Loss()
        self.cL1_lossFn = loss.CharbonnierLoss()
        self.MSE_LossFn = nn.MSELoss()

    def forward(self, input):
        """
        Network forward tensor flow

        :param input: a tuple of input that will be unfolded
        :return: medium interpolation image
        """
        img0, img1, IFrame, ref_imgt = input

        ref_align_ft = self.extract_aligned_features(ref_imgt)
        align_ft_0 = self.extract_aligned_features(img0)
        align_ft_1 = self.extract_aligned_features(img1)

        align_ft = [self.pcd_align(align_ft_0, ref_align_ft),
                    self.pcd_align(ref_align_ft, ref_align_ft),
                    self.pcd_align(align_ft_1, ref_align_ft)]
        align_ft = torch.stack(align_ft, dim=1)

        tsa_ft = self.tsa_fusion(align_ft)

        imgt = self.reconstruct(tsa_ft, ref_imgt)


        # extract for loss
        # ft_IFrame = list(self.extract_features(IFrame))[self.feature_level]
        # ft_imgt = list(self.extract_features(imgt))[self.feature_level]


        """
        ----------------------------------------------------------------------
        ======================================================================
        """

        # Loss calculate
        # feature_recn_loss = 10 * self.MSE_LossFn(ft_imgt, ft_IFrame)

        recn_loss = 128 * self.L1_lossFn(imgt, IFrame)

        loss = recn_loss# + feature_recn_loss

        MSE_val = self.MSE_LossFn(imgt, IFrame)
        psnr = (10 * log10(4 / MSE_val.item()))

        print('Loss:', loss.item(), 'psnr:', psnr,
              'recn_loss:', recn_loss.item())
              # 'feature_recn_loss:', feature_recn_loss.item())

        return loss, MSE_val, imgt
