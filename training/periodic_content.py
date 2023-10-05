import math
import torch 
from torch import nn
import torch.nn.functional as F

from misc import AdaIN, Norm


class PeriodicityUnit(nn.Module):
    """Module used to add periodic information to a feature map.
    It is used at the begining of StyleBlocks
    Given frequencies of x coordinates fx (B, nfreq) and y coordinates 
    (B, nfreq)
    input x (B,nc_in,H,W)
    output (B,nc_out,H,W)
    """
    def __init__(
            self, in_channels:int, out_channels:int, wdim:int, nfreq:int=0):
        """
        Args:
            in_channels (int): number of channels in input of the StyleBlock
            out_channels (int): number of channels in output of the StyleBlock
            wdim (int): dimension of the latent space
            nfreq (int, optional): number of frequencies. 
                Defaults to 0.
        """
        super(PeriodicityUnit, self).__init__()

        self.nfreq = nfreq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ada = AdaIN()

        self.sine_maps_conv = nn.Conv2d(nfreq, out_channels, 1)
        self.to_style = nn.Linear(wdim, nfreq)

    def forward(
            self, 
            x:torch.Tensor, 
            w_mod:torch.Tensor, 
            w_map:torch.Tensor=None, 
            fx:torch.Tensor=None, 
            fy:torch.Tensor=None, 
            phase:torch.Tensor=None,
            freq_amp:str="2scales"
        ):
        """Forward pass that computes sine maps to be further used in 
        StyleBlock.

        Args:
            x (torch.Tensor): input feature maps
            w_mod (torch.Tensor): latent code
            w_map (torch.Tensor, optional): latent code map. 
                Defaults to None.
            fx (torch.Tensor, optional): x frequencies. 
                Defaults to None.
            fy (torch.Tensor, optional): y frequencies. 
                Defaults to None.
            phase (torch.Tensor, optional): phase to add in sine maps.
                If None, a random phase is added. 
                Defaults to None.
            freq_amp (str, optional): Way to compute sine maps amplitudes.
                Defaults to "2scales".

        Returns:
            Tuple(torch.Tensor, torch.Tensor): sine maps and absolute value
                of modulation component.
        """

        B, C, H, W = x.size()

        w_grid = (torch.arange(start=0, end=W).repeat(H, 1).view(1, H, W) 
                  - (W - 1) / 2)
        h_grid = (torch.arange(start=0, end=H).repeat(W, 1).T.view(1, H, W) 
                  - (H - 1) / 2)
        grid = torch.cat(
            [w_grid, h_grid]).view(2, -1).type(torch.cuda.FloatTensor)
        
        r = torch.sqrt(fx ** 2 + fy ** 2)

        # Default: a frequency is used in the 2 levels where it appears as a
        # not too high nor too low
        if freq_amp == "2scales": 
            amp = torch.maximum(
                torch.tensor(0.).cuda(), 1 - torch.abs(1 - torch.log2(8 * r)))  
        elif freq_amp == "trans_scales":
            amp = torch.minimum(
                torch.tensor(1.).cuda(), 
                torch.maximum(torch.tensor(0.).cuda(), -torch.log2(8 * r)))
            
        freq = torch.stack((fy, fx), dim=2)

        if phase is None :
            phase = 2 * math.pi * torch.rand(1).to("cuda")

        # Compute argument and modulation
        if w_map is None:
            # sine maps modulated accordingly to their magnitude
            arg = torch.matmul(2 * math.pi * freq, grid) + phase
            sines = amp.unsqueeze(-1) * torch.sin(arg)
                     
            # input-specific modulation
            modulation = self.to_style(w_mod).unsqueeze(-1).unsqueeze(-1)
        else:
            dh, dw = x.shape[2] - w_map.shape[2], x.shape[3] - w_map.shape[3]
            pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
            w_map = F.pad(w_map, pad, mode='replicate')
            fx = F.pad(fx, pad, mode='replicate')
            fy = F.pad(fy, pad, mode='replicate')
 
            arg = ((2 * math.pi * freq.view(B, self.nfreq, 2, -1) 
                    * grid.unsqueeze(0).unsqueeze(0)).sum(-2) 
                    + phase)
            sines = amp.view(B, self.nfreq, -1) * torch.sin(arg)

            modulation = self.to_style(
                w_map.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
        
        # Comppute sine maps
        sines = sines.view(B, self.nfreq, H, W) * modulation
        out = self.sine_maps_conv(sines)  
        
        return out, modulation.abs().mean()

class Pred(nn.Module):
    """Scale and rotation prediction ntework. Takes as input a batch of latent variables w 
    and outputs a scale and rotation parameter for each element
    """
    def __init__(self, wdim:int=128):
        """
        Args:
            wdim (int, optional): dimension of the latent space. 
                Defaults to 128.
        """
        super().__init__()
        self.fc = [
            nn.Linear(wdim, 512), Norm(), nn.LeakyReLU(),
            nn.Linear(512, 128), Norm(), nn.LeakyReLU(),
            nn.Linear(128, 3)
        ]
        self.fc = nn.Sequential(*self.fc)

    def forward(self, w:torch.Tensor) :
        """Predicts scale an rotation corresponding to the given latent code

        Args:
            w (torch.Tensor): latent code

        Returns:
            Tuple(torch.Tensor, torch.Tensor): scale, theta
        """
        out = self.fc(w)
        # predict a log scale proved to be more flexible
        logit_scale = out[...,0]
        scale = 2 ** (logit_scale - 1)
        # instead of predicting an angle directly, we avoid periodicity
        # complications by predicting a point in the 2D plane, and taking its
        # argument
        x, y = out[..., 1], out[..., 2]
        # divide by 2 to get pi-periodic result, as orientation of sine waves
        # is pi-periodic
        theta = torch.atan2(y,x) / 2 
        return scale, theta