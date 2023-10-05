from argparse import Namespace
from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from torch_utils import persistence

class Options :

    def __init__(self, cfg:Union[dict, Namespace, str], **kwargs) :
        self.is_none = False

        if isinstance(cfg, dict) :
            self.from_config_dict(cfg, **kwargs)
        elif isinstance(cfg, Namespace) :
            self.setattr_from_namespace(cfg, **kwargs)
        elif isinstance(cfg, str) :
            with open(cfg, "r") as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.CFullLoader)
            self.from_config_dict(cfg, **kwargs)
        elif cfg is None :
            self.from_config_dict({}, **kwargs)
    
    def from_config_dict(self, cfg:dict) :
        return

    def setattr_from_namespace(self, args:Namespace) :
        for key, value in args.__dict__.items() :
            setattr(self, key, value)

    @classmethod
    def from_attribute_dict(cls, cfg:Union[dict, None]) :
        _cls = cls({})
        if isinstance(cfg, dict) :
            for key, value in cfg.items() :
                setattr(_cls, key, value)

        return _cls
        
    def __repr__(self) :
        _repr = f"{self.__class__.__name__}(\n"
        key_len = max([len(key) for key in self.__dict__.keys()])
        for key, value in self.__dict__.items() :
            spaced_key = key + (key_len - len(key))*" "
            _repr += f"\t{spaced_key} = {value},\n"
        _repr = _repr[:-2] + ")\n"
        return _repr
    
    def __eq__(self, __value: object) -> bool:
        if __value==None :
            return False
        else :
            return self.__dict__==__value.__dict__
        
class GeneratorOptions(Options) :

    def from_config_dict(self, cfg:dict) :
        self.nlevels           = cfg.get("NLEVELS", 7)
        self.nbands            = cfg.get("NBANDS", 11)
        self.nfreq             = cfg.get("NFREQ", 0)
        self.nc_w              = cfg.get("NC_W", 128)
        self.max_depth         = cfg.get("MAX_DEPTH", 128)
        self.local_stats_width = cfg.get("LOCAL_STATS_WIDTH", .2)

@persistence.persistent_class
class Conv(nn.Module):
    """"convolution layer to use across layers, useful to control padding mode
    """
    def __init__(self, n_ch_in:int, n_ch_out:int, k:int=3):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_ch_in, n_ch_out, k, padding=0)  
        
    def forward(self, x:torch.Tensor):
        return self.conv(x)
    
@persistence.persistent_class
class Norm(nn.Module):
    """Normalizes each channel independently of the others and of the other elements of the batch.
    Is used in the encoder due to small batch size suring training
    """
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x:torch.Tensor):
        m, s = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        return (x - m) / (s + 1e-8)
    
@persistence.persistent_class
class AdaIN(nn.Module):
    def __init__(self):
        """code modified from 
        https://github.com/CellEight/Pytorch-Adaptive-Instance-Normalization
        """
        super().__init__()
        self.width = 4

    def mu(self, x:torch.Tensor):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w)"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x:torch.Tensor):
        """ Takes a (n,c,h,w) tensor as input and returns the standard 
        deviation across it's spatial dimensions as (h,w) tensor
        """
        dx = (x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1])
        return torch.sqrt(
            (torch.sum(dx ** 2, (2, 3)) + 0.000000023) 
            / (x.shape[2] * x.shape[3]))

    def forward(self, x:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style."""

        dx = x.permute([2, 3, 0, 1]) - self.mu(x)
        return (sigma * (dx / self.sigma(x)) + mu).permute([2, 3, 0, 1])

    def forward_map(self, x:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor):
        """Performs the forward pass with mu and sigma being a 2D map instead 
        of scalar.
        Each location is normalized by local statistics, and denormalized by 
        mu and sigma evaluated at the given location.
        The local statistics are computed with a gaussian kernel of relative 
        width controlled by the variable local_stats_width defined at the top 
        of this file"""
        _, C, H0, W0 = x.shape
        # the pooling operation is used to make the computation of local mu 
        # and sigma less expensive using the spacial smoothness of locally 
        # computed statistics
        pool = nn.AdaptiveAvgPool2d((min(H0, 128), min(W0, 128))) 
        x_pooled = pool(x) # x_pooled of maximum spacial size 128*128

        # now create the gaussian kernel for loacl stats computation
        B, C, H, W = x_pooled.shape
        rx, ry = H0 / H, W0 / W
        width = self.width/min(rx,ry)
        kernel_size = [
            (max(int(2 * width), 5) // 2) * 2 + 1, 
            (max(int(2 * width), 5) // 2) * 2 + 1] # kernel size is odd
        width = [width, width]
        kernel = 1
        mgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, width, mgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel_1d = kernel.view(1, 1, *kernel.size())
        kernel = kernel_1d.repeat(C, *[1] * (kernel_1d.dim() - 1)).cuda()

        # create a weight map by convolution of a constant map with the 
        # gaussian kernel. It used to correctly compute the local statistics 
        # at the border of the image, accounting for zero padding
        ones = torch.ones(1,1,H,W).cuda()
        weight = F.conv2d(ones,kernel_1d.cuda(),bias=None,padding='same')

        # define channel-wise gaussian convolution module conv
        conv = nn.Conv2d(
            C, 
            C, 
            kernel_size, 
            groups=C, 
            bias=False, 
            stride=1, 
            padding=int((kernel_size[0] - 1) / 2), 
            padding_mode='zeros'
        )
        conv.weight.data = kernel
        conv.weight.requires_grad = False
        
        # pooling already performs local averaging, so it does not perturb the 
        # computation of the local mean
        local_mu = conv(x_pooled)/weight 
        local_mu = F.interpolate(
            local_mu, size=(H0,W0), mode='bilinear', align_corners=False) # now upsample the local mean map to the original shape

        local_sigma = torch.sqrt(
            conv(pool(((x-local_mu)**2)) /weight) + 10 ** -8) # perform (x-local_mu)**2 at the high resolution, THEN pool and finally smooth to get the local standard deviation.
        
        local_sigma = F.interpolate(
            local_sigma, size=(H0,W0), mode='bilinear', align_corners=False) # upsample the local std map to the original shape
    
        #finally perform the local AdaIN operation using these maps of local_mu and local_sigma to normalize, then denormalize with the given maps mu and sigma.
        x_norm = (x - local_mu) / local_sigma
        return (sigma * x_norm + mu)
        