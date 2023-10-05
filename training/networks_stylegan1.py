import math
from typing import Callable, Tuple, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch_utils import persistence

from misc import Conv, AdaIN, GeneratorOptions
from periodic_content import Pred, PeriodicityUnit
from torch_utils import misc


@persistence.persistent_class
class StyleConv(nn.Module):
    """Basic Style convolution operation: convolution followed by AdaIN.
    Used in StyleBlocks and at the end of the generator to project in the 
    wanted color space.
    Uses the latent variable w to predict statistics mu and sigma used in 
    AdaIN, then performs convolution 3*3 and add noise.
    """
    def __init__(self, nc_in:int, nc_out:int, w_dim:int, k=3, use_noise=True):
        """
        Args:
            nc_in (int): number of channels in input of the layer.
            nc_out (int): number of channels in output of the layer.
            w_dim (int): dimension of the latent space.
            k (int, optional): size of the convolutionnal kernel. 
                Defaults to 3.
            use_noise (bool, optional): whethere to ads noise or not. 
                Defaults to True.
        """
        
        super().__init__()
        self.conv1 = Conv(nc_in, nc_out, k)
        self.ada = AdaIN()
        self.nc_out = nc_out
        self.to_style = nn.Linear(w_dim, nc_out * 2)

        self.use_noise = use_noise
        if self.use_noise :
            # Noise maps to save for inference experiments with fixed spatial 
            # realization, such as interpolation in the latent space
            self.noise = None
            # Learnable modulation of the noise
            self.noise_modulation = nn.Parameter(
                .01 * torch.randn(1, nc_out, 1, 1).cuda(), requires_grad=True)

    def forward(
            self, 
            x:torch.Tensor, 
            w:torch.Tensor, 
            w_map:Union[torch.Tensor, None]=None,
            op_add:Callable=lambda x: x,
            save_noise:bool=False
        ):
        """Forward pass.
        Either takes a single w vector or a map of latent vectors w_map to 
        allow spatial texture interpolation : w_map is a small map of
        different textures style in different locations. It needs to be
        smoothly upsampled to the size of the current feature map.

        Args:
            x (torch.Tensor): input feature maps.
            w (torch.Tensor): latent vector.
            w_map (Union[torch.Tensor, None], optional): latent vector map. 
                If None, this parameter is not taken into account and the 
                StyleConv layer only uses the latent vector w.
                Defaults to None.
            op_add (Callable): operation to perform after convolution and
                before AdaIn (e.g. sine maps for periodic content).
                Defaults to None.
            save_noise (bool, optional): whether to save noise or not.

        Returns:
            torch.Tensor: output feature maps
        """
        # Conv
        x = self.conv1(x)

        # Additionnal operation feature maps
        x = op_add(x)

        # Noise
        if self.use_noise :
            if save_noise and self.n2 is None:
                self.noise = torch.randn(1,*x.shape[1:]).cuda()
                noise = self.noise
            elif save_noise:
                noise = self.noise
            else:
                self.noise = None
                noise = torch.randn(x.shape).cuda()
            x += self.noise_modulation * noise

        # AdaIN
        if w_map is None:
            style = self.to_style(w)  # w is B,nc_w
            mu, sigma = self.get_statistics(style)
            x = self.ada(x, mu, sigma)
        else:
            dh,dw = x.shape[2] - w_map.shape[2], x.shape[3] - w_map.shape[3]
            w_map = F.pad(
                w_map, 
                (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), 
                mode='replicate')
            style_map = self.to_style(
                w_map.permute((0, 2, 3, 1))).permute((0, 3, 1, 2))
            mu = style_map[:, :self.nc_out]
            sigma = style_map[:, self.nc_out:2 * self.nc_out]
            x = self.ada.forward_map(x, mu, sigma)
        return x
    
    def get_statistics(self, style:torch.Tensor) :
        """Extracts statistics (mean and standard deviation) from style vector
        or style map.

        Args:
            style (torch.Tensor): style vector or map.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): mean and std.
        """
        mu = style[:, :self.nc_out]
        sigma = style[:, self.nc_out:self.nc_out * 2]
        return mu, sigma
    
@persistence.persistent_class
class StyleBlock(nn.Module):
    """Basic StyleBlock made of :
        - Periodic content addition : add or concatenatze sine maps modulated
        by latent code
        - Upsampling operator (x2)
        - 2 StyleConv layers 
        - Non linear operator
    """

    def __init__(
            self, 
            nc_in:int, 
            nc_out:int, 
            w_dim:int, 
            n_freq:int=0, 
            sine_maps:bool=False, 
            sine_maps_merge:str="add"
        ):
        """
        Args:
            nc_in (int): number of channels in input of the layer.
            nc_out (int): number of channels in output of the layer.
            w_dim (int): dimension of the latent space.
            n_freq (int, optional): number of learned frequencies. 
                Defaults to 0.
            sine_maps (bool, optional): if sine maps are used or not. 
                Defaults to False.
            sine_maps_merge (str, optional): Way to treat sine maps. 
                Either add or concatenate them to the input feature maps.
                Defaults to "add".
        """
        super().__init__()
        self.n_freq = n_freq
        self.nc_in=nc_in
        self.nc_out=nc_out
        self.sine_maps = sine_maps
        self.sine_maps_merge = sine_maps_merge
        
        if sine_maps:
            if sine_maps_merge == 'add':
                self.periodic_content = PeriodicityUnit(
                    nc_in, nc_out, w_dim, n_freq)
                self.style_conv1 = StyleConv(nc_in, nc_out, w_dim)
            else: 
                self.periodic_content = PeriodicityUnit(
                    nc_in, nc_out // 2, w_dim, n_freq)
                self.style_conv1 = StyleConv(nc_in, nc_out // 2, w_dim)
        else: 
            self.periodic_content = None
            self.style_conv1 = StyleConv(nc_in, nc_out, w_dim)

        self.style_conv2 = StyleConv(nc_out, nc_out, w_dim)

        self.nl = nn.LeakyReLU(.1)
    
    def forward(
            self, 
            x:torch.Tensor, 
            w:torch.Tensor, 
            fx, 
            fy, 
            w_map=None, 
            save_noise=False, 
            phase=None):
        
        # Upsample x2
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Retrieve periodic content and merge it to the current feature maps 
        # accordingly to sine_maps_merge mode
        if self.periodic_content is not None : 
            x_f_conv, l1_loss = self.periodic_content(
                x, w, w_map, fx, fy, phase=phase)  
            x_f_conv = x_f_conv[..., 1:-1, 1:-1]
            self.l1_loss = l1_loss
            if self.sine_maps_merge == 'add':
                op_add = lambda x_conv : x_conv + x_f_conv
            else :
                op_add = lambda x_conv : torch.cat((x_conv, x_f_conv), dim=1)
        else :
            self.l1_loss = 0. * w.mean()
            op_add = lambda x : x

        # StyleConv
        x = self.style_conv1(
            x, w, w_map=w_map, op_add=op_add, save_noise=save_noise)
        x = self.style_conv2(
            x, w, w_map=w_map, save_noise=save_noise)
        
        # Non-linearity (was found essential experimentally)
        x = self.nl(x) 

        return x
    
@persistence.persistent_class
class SynthesisNetwork(nn.Module):
    """Modified Style based generator, inspired from StyleGAN architecture.
    It allows one to generate a texture from a latent code or a map of
    different latent code to perform spatial texture interpolation.
    """
    def __init__(self, generator_options:GeneratorOptions):
        """
        Args:
            generator_options (GeneratorOptions): options to consider.
        """
        super(SynthesisNetwork, self).__init__()

        self.generator_options = generator_options
        self.zoom = (1, 1)

        # Not 4*4 like in StyleGAN, but a constant replicated along spatial
        # axes at the start of the network
        self.input_tensor = nn.Parameter(
            torch.randn(1, self.generator_options.max_depth, 1, 1).cuda(), 
            requires_grad=True) 

        # The input tensor is expanded to a greater spacial size to account
        # for the fact that no padding is performed in any convolution in the
        # generator
        self.pad = 4  
        
        # for inference purposes
        self.save_noise = False
        self.offset= None 

        # Scale and rotation predictor network
        self.pred = Pred(self.generator_options.nc_w) 

        # For plotting histograms of predicted scale and rotation parameters
        self.scale, self.theta = None, None 

        # learnable frequency are learned with a greater learning rate with
        # a factor 100
        self.grad_boost = 100 
        # initialization of learnable frequencies
        if self.generator_options.nfreq != 0:
            linespace_radius = torch.linspace(
                1, 
                self.generator_options.nlevels, 
                self.generator_options.nfreq)
            linespace_phase = torch.linspace(
                0,
                np.pi, 
                self.generator_options.nfreq
            )[torch.randperm(self.generator_options.nfreq)]

            # log magnitude of the frequencies, will be exponentiated to get
            # the magnitude
            self.r = nn.Parameter(
                (linespace_radius / self.grad_boost).cuda(), 
                requires_grad=True) 
            self.phase = nn.Parameter(
                (linespace_phase / self.grad_boost).cuda(), 
                requires_grad=True)
        
        # Main body = cascade of StyleBlock modules
        l = []
        for i in range(self.generator_options.nlevels): 
            ch_in  = 2 ** (5 + self.generator_options.nlevels - i)
            ch_out = 2 ** (5 + self.generator_options.nlevels - i - 1)
            l.append(StyleBlock(
                nc_in  = min(self.generator_options.max_depth, ch_in), 
                nc_out = min(self.generator_options.max_depth, ch_out), 
                w_dim  = self.generator_options.nc_w, 
                n_freq = self.generator_options.nfreq
            ))
        self.body_modules = nn.ModuleList(l)

        def body_forward(
                x:torch.Tensor, w:torch.Tensor, w_map:torch.Tensor=None):
            if w_map is None:
                # infer scale and rotation
                s, t = self.pred(w) 
                # get magnitude of each frequency
                mod = 2 ** (-self.r * self.grad_boost) 
                fx = s.unsqueeze(1) * mod.unsqueeze(0) * torch.cos(
                    self.phase.unsqueeze(0) * self.grad_boost + t.unsqueeze(1)
                )
                fy = s.unsqueeze(1) * mod.unsqueeze(0) * torch.sin(
                    self.phase.unsqueeze(0) * self.grad_boost + t.unsqueeze(1)
                )  
                # In this operation, we retrieve the cartesian coordonates of
                # each transformed frequency f'_i the first two dimensions of
                # fx are (batch_size,n_freq,...) each frequency is transformed
                # differently according to the element of the batch through
                # the latent variable w, that yields exemplar-specific scale
                # and rotation prediction.
                # For each element in the batch b, all the learned frequencies
                # are scaled and rotated with the same predicted parameters
                # s_b and t_b.

                # for logging
                self.scale, self.theta = s, t 
            
            # for inference experiments, phase is random during training
            if self.save_noise and self.offset is None: 
                self.offset = 2 * math.pi * torch.rand(1, 1, 1).to("cuda")
                offset = self.offset
            elif self.save_noise:
                offset = self.offset
            else:
                offset = (2**self.generator_options.nlevels 
                          * math.pi 
                          * torch.rand(x.shape[0], 1, 1).to("cuda"))

            for i, m in enumerate(self.body_modules):
                if w_map is not None:
                    w_map_curr = F.interpolate(
                        w_map, 
                        (2**(i+2) * self.zoom[0], 2**(i+2) * self.zoom[1]), 
                        mode='bilinear', 
                        align_corners=True)
                    # Important detail: local_stats_width controls how local
                    # stats are computed you may try from .1 to .8
                    m.ada.width = (2**(i+2) 
                                   * self.generator_options.local_stats_width 
                                   * min(self.zoom[0],self.zoom[1])) 
                    s, t = self.pred(w_map_curr.permute((0, 2, 3, 1)))
                    if self.generator_options.sine_maps:
                        mod = 2 ** (-self.r * self.grad_boost)
                        pow2 = 2 ** (self.generator_options.nlevels - i - 1)
                        a = (
                            self.phase.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
                            * self.grad_boost 
                            + t.unsqueeze(1)
                        )
                        fx = (s.unsqueeze(1) 
                              * mod.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
                              * torch.cos(a) 
                              * pow2)
                        fy = (s.unsqueeze(1) 
                              * mod.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
                              * torch.sin(a) 
                              * pow2) 
                    else:
                        fx, fy = 0, 0

                    x = m(
                        x, w, fx, fy, 
                        w_map=w_map_curr, 
                        save_noise=self.save_noise, 
                        phase=offset * 2 ** - i
                    )
                else :
                    # Phases are scaled in order for the sine waves to be
                    # aligned in the two consectutive leveks in which a
                    # frequency is used
                    x = m(
                        x, w, fx, fy, 
                        w_map=w_map, 
                        save_noise=self.save_noise, 
                        phase=offset * 2 ** - i
                    )

            return x

        self.body = body_forward
        self.rgb = StyleConv(
            nc_in     = 32, 
            nc_out    = self.generator_options.nbands, 
            w_dim     = self.generator_options.nc_w, 
            k         = 1,
            use_noise = False)

    def forward(
            self, 
            w:torch.Tensor, 
            w_map:torch.Tensor=None, 
            zoom:Tuple[int, int]=(1,1)) :
        """
        Args:
            w (torch.Tensor): latent code
            w_map (torch.Tensor, optional): latent codes map. 
                Defaults to None.
            zoom (Tuple[int, int], optional): Zoom factor to use when croping. 
                Defaults to (1,1).

        Returns:
            (torch.Tensor): synthetised textured
        """
        
        self.zoom = zoom
        self.pad = 4 if zoom==(1,1) else 5
        
        k = 2**(8 - self.generator_options.nlevels) * zoom[0] + self.pad
        x = self.body(self.input_tensor.repeat(w.shape[0], 1, k, k), w, w_map)

        if w_map is not None:
            self.rgb.ada.width = (2**(self.generator_options.nlevels+1) 
                                  * self.generator_options.local_stats_width 
                                  * min(self.zoom[0], self.zoom[1]))
            size = (
                2**(self.generator_options.nlevels+1) * zoom[0], 
                2**(self.generator_options.nlevels+1)*zoom[1]
            )
            w_map = F.interpolate(
                w_map, size=size, mode='bilinear', align_corners=True)

        x = self.rgb(x, w, w_map)
        
        if self.training:
            x = transforms.RandomCrop((256 * zoom[0], 256 * zoom[1]))(x)
        else:
            x = TF.center_crop(x, (256 * zoom[0], 256 * zoom[1]))
        
        x = torch.tanh(x)  
        return x
    
    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.generator_options.nc_w:d},',
            f'img_resolution=256, img_channels={self.generator_options.nbands:d},',
            f'num_layers={self.generator_options.nlevels:d}'])
    
@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = torch.matmul(x, w.t())
        if self.activation=='lrelu' :
            x = torch.nn.functional.leaky_relu(x, 0.2)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'
    
@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

    def forward(self, z, c):
        misc.assert_shape(z, [None, self.z_dim])

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}'

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        nlevels = np.log2(img_resolution // 4) - 1
        assert nlevels.is_integer()
        self.nlevels = nlevels
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(self._get_generator_options(**synthesis_kwargs))
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, **mapping_kwargs)

    def _get_generator_options(self, synthesis_kwargs:dict) :
        synthesis_args = dict([
            ("nlevels", self.nlevels),
            ("nbands", 3),
            ("nfreq", synthesis_kwargs.get("nfreq", 0)),
            ("nc_w", self.w_dim),
            ("max_depth", synthesis_kwargs.get("max_depth", 128)),
            ("local_stats_width", synthesis_kwargs.get("local_stats_width", .2))
        ])
        return GeneratorOptions.from_attribute_dict(synthesis_args)
    

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img