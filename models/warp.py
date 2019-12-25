import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1,
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, H, W):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda).
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.nn.Parameter(torch.tensor(gridX), requires_grad=False)
        self.gridY = torch.nn.Parameter(torch.tensor(gridY), requires_grad=False)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """


        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, padding_mode='border')
        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1.
class Coeff(nn.Module):

    def __init__(self):
        super(Coeff, self).__init__()
        self.t = torch.nn.Parameter(torch.FloatTensor(np.linspace(0.125, 0.875, 7)), requires_grad=False)

    def getFlowCoeff (self, indices):
        """
        Gets flow coefficients used for calculating intermediate optical
        flows from optical flows between I0 and I1: F_0_1 and F_1_0.

        F_t_0 = C00 x F_0_1 + C01 x F_1_0
        F_t_1 = C10 x F_0_1 + C11 x F_1_0

        where,
        C00 = -(1 - t) x t
        C01 = t x t
        C10 = (1 - t) x (1 - t)
        C11 = -t x (1 - t)

        Parameters
        ----------
            indices : tensor
                indices corresponding to the intermediate frame positions
                of all samples in the batch.
            device : device
                    computation device (cpu/cuda).

        Returns
        -------
            tensor
                coefficients C00, C01, C10, C11.
        """


        # Convert indices tensor to numpy array
        ind = indices.detach()
        C11 = C00 = - (1 - (self.t[ind])) * (self.t[ind])
        C01 = (self.t[ind]) * (self.t[ind])
        C10 = (1 - (self.t[ind])) * (1 - (self.t[ind]))
        return C00[None, None, None, :].permute(3, 0, 1, 2), C01[None, None, None, :].permute(3, 0, 1, 2), C10[None, None, None, :].permute(3, 0, 1, 2), C11[None, None, None, :].permute(3, 0, 1, 2)

    def getWarpCoeff (self, indices):
        """
        Gets coefficients used for calculating final intermediate
        frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

        It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

        where,
        C0 = 1 - t
        C1 = t

        V_t_0, V_t_1 --> visibility maps
        g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

        Parameters
        ----------
            indices : tensor
                indices corresponding to the intermediate frame positions
                of all samples in the batch.
            device : device
                    computation device (cpu/cuda).

        Returns
        -------
            tensor
                coefficients C0 and C1.
        """


        # Convert indices tensor to numpy array
        ind = indices.detach()
        C0 = 1 - self.t[ind]
        C1 = self.t[ind]
        return C0[None, None, None, :].permute(3, 0, 1, 2), C1[None, None, None, :].permute(3, 0, 1, 2)

    def set_t(self, factor):
        ti = 1 / factor
        self.t = torch.nn.Parameter(torch.FloatTensor(np.linspace(ti, 1 - ti, factor - 1)), requires_grad=False)