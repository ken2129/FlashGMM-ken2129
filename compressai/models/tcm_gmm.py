'''
TCM Model from the paper "Learned Image Compression with Mixed Transformer-CNN Architectures"
(Liu et al., CVPR 2023)
adapted from https://github.com/jmliu206/LIC_TCM/
'''

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.entropy_models import GaussianMixtureConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}", 
            state_dict,
            policy,
            dtype,
        )

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = ResidualBlock(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class SWAtten(AttentionBlock):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(input_dim, input_dim, head_dim, window_size, drop_path)
        if inter_dim is not None:
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='W')
        self.block_2 = Block(input_dim, output_dim, head_dim, window_size, drop_path, type='SW')
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        trans_x = self.block_1(trans_x)
        trans_x =  self.block_2(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x

class TCM_GMM_Light(CompressionModel):
    """TCM with Gaussian Mixture Model - Light version.
    
    Uses unified h_s, atten, and cc_transforms to reduce parameters.
    Suitable for resource-constrained scenarios.
    """
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], 
    drop_path_rate=0, N=128,  M=320, num_slices=5, max_support_slices=5, K = 4, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        

        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [conv3x3(2*N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 2*N, 2)] + \
            self.ha_down1
        )

        # Single h_s outputs 320*3 channels, then chunk into mean, scale, weight
        self.hs_up = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320 * 3, 2)]

        self.h_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up
        )
        # h_s: 192 -> 320*3, then chunk into (mean: 320, scale: 320, weight: 320)

        # Attention modules: same input size as TCM (320 + support_slices)
        self.atten = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), 
                (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size,0, inter_dim=128)
            ) for i in range(self.num_slices)
            )
        # cc_transforms: output 3*K times the channels for GMM (mean, scale, weight combined)
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices) * K * 3, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_mixture_conditional = GaussianMixtureConditional(K)
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        # Single h_s -> chunk into mean, scale, weight (each 320 channels)
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales, latent_weights = latent_params.chunk(3, dim=1)
        # each of size (B, 320, H, W)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        
        mu_list = []
        scale_list = []
        weight_list = []

        K = self.gaussian_mixture_conditional.K
        slice_ch = self.M // self.num_slices  # 64

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            # of size [(B, 64, H, W) ...]
            
            # Concatenate latent params with support slices
            support = torch.cat([latent_means, latent_scales, latent_weights] + support_slices, dim=1)
            # of size (B, 320*3 + 64 * min(slice_index, 5), H, W)
            # But we need to match the atten input size (320 + 64 * min(i, 5))
            # So we use just latent_means for the support (same as TCM)
            support = torch.cat([latent_means] + support_slices, dim=1)
            # of size (B, 320 + 64 * min(slice_index, 5), H, W)
            support = self.atten[slice_index](support)
            
            # cc_transforms outputs (mean, scale, weight) combined: (B, 64*K*3, H, W)
            gmm_params = self.cc_transforms[slice_index](support)
            gmm_params = gmm_params[:, :, :y_shape[0], :y_shape[1]]
            
            # Split into mu, scale, weight (each 64*K channels)
            mu, scale, weight = gmm_params.chunk(3, dim=1)
            # mu, scale, weight shapes: (B, 64*K, H, W)
            
            mu_list.append(mu)
            scale_list.append(scale)

            # Apply softmax to weights
            weight = self.gaussian_mixture_conditional._reshape_gmm_weight(weight)
            weight_list.append(weight)

            # Compute weighted mean for quantization (use dominant component mean)
            B, CK, H, W = mu.shape
            C = CK // K
            mu_expanded = mu.view(B, K, C, H, W)
            weight_expanded = weight.view(B, K, C, H, W)
            # Use weighted mean of GMM components for quantization
            weighted_mu = torch.sum(mu_expanded * weight_expanded, dim=1)  # (B, C, H, W)

            _, y_slice_likelihood = self.gaussian_mixture_conditional(y_slice, scale, mu, weight)
            y_likelihood.append(y_slice_likelihood)
            
            # Quantize using weighted mean
            y_hat_slice = ste_round(y_slice - weighted_mu) + weighted_mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Single h_s -> chunk into mean, scale, weight (each 320 channels)
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales, latent_weights = latent_params.chunk(3, dim=1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []
        y_weights = []
        y_symbols = []
        y_strings = []

        K = self.gaussian_mixture_conditional.K

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            support = torch.cat([latent_means] + support_slices, dim=1)
            support = self.atten[slice_index](support)
            
            # cc_transforms outputs (mean, scale, weight) combined: (B, 64*K*3, H, W)
            gmm_params = self.cc_transforms[slice_index](support)
            gmm_params = gmm_params[:, :, :y_shape[0], :y_shape[1]]
            
            # Split into mu, scale, weight (each 64*K channels)
            mu, scale, weight = gmm_params.chunk(3, dim=1)
            
            # Apply softmax to weights
            weight_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(weight)

            # Compute weighted mean for quantization
            B, CK, H, W = mu.shape
            C = CK // K
            mu_expanded = mu.view(B, K, C, H, W)
            weight_expanded = weight_normalized.view(B, K, C, H, W)
            weighted_mu = torch.sum(mu_expanded * weight_expanded, dim=1)  # (B, C, H, W)

            # Quantize: round(y - weighted_mu)
            y_q_slice = torch.round(y_slice - weighted_mu)
            y_hat_slice = y_q_slice + weighted_mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)
            y_weights.append(weight)  # Store un-normalized weights
            y_symbols.append(y_q_slice)

        # Concatenate all slices for GMM encoding
        y_scales_tensor = torch.cat(y_scales, dim=1)  # (B, M*K, H, W)
        y_means_tensor = torch.cat(y_means, dim=1)    # (B, M*K, H, W)
        y_weights_tensor = torch.cat(y_weights, dim=1)  # (B, M*K, H, W)
        y_symbols_tensor = torch.cat(y_symbols, dim=1)  # (B, M, H, W)

        # Reshape for GMM encoding
        B, MK, H, W = y_scales_tensor.shape
        M = MK // K
        
        # Reshape scales/means from (B, M*K, H, W) to (B*M*H*W, K)
        # Layout: (B, K, M, H, W) -> (B, M, H, W, K) -> (B*M*H*W, K)
        y_scales_gmm = y_scales_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K).clamp(0.11, 256)
        y_means_gmm = y_means_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)
        
        # Apply softmax to weights and reshape
        y_weights_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(y_weights_tensor)
        y_weights_gmm = y_weights_normalized.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)
        
        y_symbols_flat = y_symbols_tensor.reshape(-1).int()  # (B*M*H*W,)

        encoder = BufferedRansEncoder()
        # Move to CPU for encoding
        encoder.encode_with_indexes_gmm(
            y_symbols_flat.cpu(), 
            y_scales_gmm.cpu(), 
            y_means_gmm.cpu(), 
            y_weights_gmm.cpu(), 
            0
        )
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        # Single h_s -> chunk into mean, scale, weight (each 320 channels)
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales, latent_weights = latent_params.chunk(3, dim=1)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        K = self.gaussian_mixture_conditional.K

        # First pass: compute all GMM parameters with placeholder y_hat
        # (matching the autoregressive structure used in compress)
        all_scales = []
        all_means = []
        all_weights = []
        all_supports = []
        temp_y_hat_slices = []
        
        for slice_index in range(self.num_slices):
            support_slices = (temp_y_hat_slices if self.max_support_slices < 0 else temp_y_hat_slices[:self.max_support_slices])
            
            support = torch.cat([latent_means] + support_slices, dim=1)
            support = self.atten[slice_index](support)
            
            # cc_transforms outputs (mean, scale, weight) combined: (B, 64*K*3, H, W)
            gmm_params = self.cc_transforms[slice_index](support)
            gmm_params = gmm_params[:, :, :y_shape[0], :y_shape[1]]
            
            # Split into mu, scale, weight (each 64*K channels)
            mu, scale, weight = gmm_params.chunk(3, dim=1)

            all_scales.append(scale)
            all_means.append(mu)
            all_weights.append(weight)
            all_supports.append(support)
            
            # Placeholder for next iteration
            B, CK, H, W = mu.shape
            C = CK // K
            temp_y_hat_slices.append(torch.zeros(B, C, H, W, device=mu.device))

        # Concatenate all parameters for batch decoding
        y_scales_tensor = torch.cat(all_scales, dim=1)
        y_means_tensor = torch.cat(all_means, dim=1)
        y_weights_tensor = torch.cat(all_weights, dim=1)

        B, MK, H, W = y_scales_tensor.shape
        M = MK // K
        
        # Reshape for GMM decoding
        y_scales_gmm = y_scales_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K).clamp(0.11, 256)
        y_means_gmm = y_means_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)
        y_weights_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(y_weights_tensor)
        y_weights_gmm = y_weights_normalized.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)

        # Decode all symbols at once (move to CPU for decoding)
        device = y_scales_tensor.device
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        rv_flat = decoder.decode_stream_gmm(
            y_scales_gmm.cpu(), 
            y_means_gmm.cpu(), 
            y_weights_gmm.cpu(), 
            255
        )
        rv = rv_flat.reshape(B, M, H, W).to(device)

        # Split decoded symbols and apply LRP for each slice
        y_hat_slices = []
        slice_size = M // self.num_slices
        
        for slice_index in range(self.num_slices):
            start_ch = slice_index * slice_size
            end_ch = (slice_index + 1) * slice_size
            
            rv_slice = rv[:, start_ch:end_ch, :, :]
            mu = all_means[slice_index]
            weight = all_weights[slice_index]
            support = all_supports[slice_index]
            
            C = slice_size
            weight_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(weight)
            mu_expanded = mu.view(B, K, C, H, W)
            weight_expanded = weight_normalized.view(B, K, C, H, W)
            weighted_mu = torch.sum(mu_expanded * weight_expanded, dim=1)

            y_hat_slice = rv_slice.float() + weighted_mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


class TCM_GMM(CompressionModel):
    """TCM with Gaussian Mixture Model - Full version.
    
    Uses unified h_s but separate atten and cc_transforms for mean/scale/weight.
    Provides richer context modeling for better compression performance.
    """
    def __init__(self, config=[2, 2, 2, 2, 2, 2], head_dim=[8, 16, 32, 32, 16, 8], 
    drop_path_rate=0, N=128,  M=320, num_slices=5, max_support_slices=5, K=4, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        dim = N
        self.M = M
        self.K = K
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0

        self.m_down1 = [ConvTransBlock(dim, dim, self.head_dim[0], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim[1], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride(2*N, 2*N, stride=2)]
        self.m_down3 = [ConvTransBlock(dim, dim, self.head_dim[2], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW')
                      for i in range(config[2])] + \
                      [conv3x3(2*N, M, stride=2)]

        self.m_up1 = [ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up2 = [ConvTransBlock(dim, dim, self.head_dim[4], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[4])] + \
                      [ResidualBlockUpsample(2*N, 2*N, 2)]
        self.m_up3 = [ConvTransBlock(dim, dim, self.head_dim[5], self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[5])] + \
                      [subpel_conv3x3(2*N, 3, 2)]
        
        self.g_a = nn.Sequential(*[ResidualBlockWithStride(3, 2*N, 2)] + self.m_down1 + self.m_down2 + self.m_down3)
        self.g_s = nn.Sequential(*[ResidualBlockUpsample(M, 2*N, 2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down1 = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [conv3x3(2*N, 192, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBlockWithStride(320, 2*N, 2)] + \
            self.ha_down1
        )

        # Single h_s outputs 320*3 channels, then chunk into mean, scale, weight
        self.hs_up = [ConvTransBlock(N, N, 32, 4, 0, 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [subpel_conv3x3(2*N, 320 * 3, 2)]

        self.h_s = nn.Sequential(
            *[ResidualBlockUpsample(192, 2*N, 2)] + \
            self.hs_up
        )
        # h_s: 192 -> 320*3, then chunk into (mean: 320, scale: 320, weight: 320)

        # Separate attention modules for mean, scale, weight (rich context modeling)
        self.atten_mean = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), 
                (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.atten_scale = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), 
                (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )
        self.atten_weight = nn.ModuleList(
            nn.Sequential(
                SWAtten((320 + (320//self.num_slices)*min(i, 5)), 
                (320 + (320//self.num_slices)*min(i, 5)), 16, self.window_size, 0, inter_dim=128)
            ) for i in range(self.num_slices)
        )

        # Separate cc_transforms for mean, scale, weight (output K times channels)
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices) * K, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices) * K, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_weight_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices) * K, stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + (320//self.num_slices)*min(i+1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_mixture_conditional = GaussianMixtureConditional(K)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_mixture_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        # Single h_s -> chunk into mean, scale, weight (each 320 channels)
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales, latent_weights = latent_params.chunk(3, dim=1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        
        mu_list = []
        scale_list = []

        K = self.gaussian_mixture_conditional.K
        slice_ch = self.M // self.num_slices  # 64

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            
            # Separate processing for mean, scale, weight
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]
            scale_list.append(scale)

            weight_support = torch.cat([latent_weights] + support_slices, dim=1)
            weight_support = self.atten_weight[slice_index](weight_support)
            weight = self.cc_weight_transforms[slice_index](weight_support)
            weight = weight[:, :, :y_shape[0], :y_shape[1]]
            weight = self.gaussian_mixture_conditional._reshape_gmm_weight(weight)

            # Compute weighted mean for quantization
            B, CK, H, W = mu.shape
            C = CK // K
            mu_expanded = mu.view(B, K, C, H, W)
            weight_expanded = weight.view(B, K, C, H, W)
            weighted_mu = torch.sum(mu_expanded * weight_expanded, dim=1)

            _, y_slice_likelihood = self.gaussian_mixture_conditional(y_slice, scale, mu, weight)
            y_likelihood.append(y_slice_likelihood)
            
            y_hat_slice = ste_round(y_slice - weighted_mu) + weighted_mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Single h_s -> chunk into mean, scale, weight
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales, latent_weights = latent_params.chunk(3, dim=1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []
        y_weights = []
        y_symbols = []

        K = self.gaussian_mixture_conditional.K

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            weight_support = torch.cat([latent_weights] + support_slices, dim=1)
            weight_support = self.atten_weight[slice_index](weight_support)
            weight = self.cc_weight_transforms[slice_index](weight_support)
            weight = weight[:, :, :y_shape[0], :y_shape[1]]
            
            weight_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(weight)

            B, CK, H, W = mu.shape
            C = CK // K
            mu_expanded = mu.view(B, K, C, H, W)
            weight_expanded = weight_normalized.view(B, K, C, H, W)
            weighted_mu = torch.sum(mu_expanded * weight_expanded, dim=1)

            y_q_slice = torch.round(y_slice - weighted_mu)
            y_hat_slice = y_q_slice + weighted_mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)
            y_weights.append(weight)
            y_symbols.append(y_q_slice)

        # Concatenate all slices for GMM encoding
        y_scales_tensor = torch.cat(y_scales, dim=1)
        y_means_tensor = torch.cat(y_means, dim=1)
        y_weights_tensor = torch.cat(y_weights, dim=1)
        y_symbols_tensor = torch.cat(y_symbols, dim=1)

        B, MK, H, W = y_scales_tensor.shape
        M = MK // K
        
        y_scales_gmm = y_scales_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K).clamp(0.11, 256)
        y_means_gmm = y_means_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)
        y_weights_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(y_weights_tensor)
        y_weights_gmm = y_weights_normalized.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)
        y_symbols_flat = y_symbols_tensor.permute(0, 2, 3, 1).reshape(-1).int()

        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes_gmm(
            y_symbols_flat.cpu(),
            y_scales_gmm.cpu(),
            y_means_gmm.cpu(),
            y_weights_gmm.cpu(),
            255
        )
        y_string = encoder.flush()

        return {"strings": [[y_string], z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        latent_params = self.h_s(z_hat)
        latent_means, latent_scales, latent_weights = latent_params.chunk(3, dim=1)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]

        K = self.gaussian_mixture_conditional.K

        # First pass: compute all GMM parameters
        all_scales = []
        all_means = []
        all_weights = []
        all_mean_supports = []
        temp_y_hat_slices = []
        
        for slice_index in range(self.num_slices):
            support_slices = (temp_y_hat_slices if self.max_support_slices < 0 else temp_y_hat_slices[:self.max_support_slices])
            
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support = self.atten_mean[slice_index](mean_support)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support = self.atten_scale[slice_index](scale_support)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            weight_support = torch.cat([latent_weights] + support_slices, dim=1)
            weight_support = self.atten_weight[slice_index](weight_support)
            weight = self.cc_weight_transforms[slice_index](weight_support)
            weight = weight[:, :, :y_shape[0], :y_shape[1]]

            all_scales.append(scale)
            all_means.append(mu)
            all_weights.append(weight)
            all_mean_supports.append(mean_support)
            
            B, CK, H, W = mu.shape
            C = CK // K
            temp_y_hat_slices.append(torch.zeros(B, C, H, W, device=mu.device))

        # Concatenate for batch decoding
        y_scales_tensor = torch.cat(all_scales, dim=1)
        y_means_tensor = torch.cat(all_means, dim=1)
        y_weights_tensor = torch.cat(all_weights, dim=1)

        B, MK, H, W = y_scales_tensor.shape
        M = MK // K
        
        y_scales_gmm = y_scales_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K).clamp(0.11, 256)
        y_means_gmm = y_means_tensor.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)
        y_weights_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(y_weights_tensor)
        y_weights_gmm = y_weights_normalized.view(B, K, M, H, W).permute(0, 2, 3, 4, 1).reshape(-1, K)

        device = y_scales_tensor.device
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        rv_flat = decoder.decode_stream_gmm(
            y_scales_gmm.cpu(), 
            y_means_gmm.cpu(), 
            y_weights_gmm.cpu(), 
            255
        )
        rv = rv_flat.reshape(B, M, H, W).to(device)

        # Apply LRP for each slice
        y_hat_slices = []
        slice_size = M // self.num_slices
        
        for slice_index in range(self.num_slices):
            start_ch = slice_index * slice_size
            end_ch = (slice_index + 1) * slice_size
            
            rv_slice = rv[:, start_ch:end_ch, :, :]
            mu = all_means[slice_index]
            weight = all_weights[slice_index]
            mean_support = all_mean_supports[slice_index]
            
            C = slice_size
            weight_normalized = self.gaussian_mixture_conditional._reshape_gmm_weight(weight)
            mu_expanded = mu.view(B, K, C, H, W)
            weight_expanded = weight_normalized.view(B, K, C, H, W)
            weighted_mu = torch.sum(mu_expanded * weight_expanded, dim=1)

            y_hat_slice = rv_slice.float() + weighted_mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}