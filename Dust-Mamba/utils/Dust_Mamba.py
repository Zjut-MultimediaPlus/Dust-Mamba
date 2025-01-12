from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from einops import rearrange

# ----------------------------------------------------------------------------------------------
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# ----------------------------------------------------------------------
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch, bilinear):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1, bilinear=False):
        super(UNet, self).__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.bilinear = bilinear
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        # 0  1   2   3   4
        # 64 128 256 512 1024

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3], bilinear)
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], bilinear)
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], bilinear)
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], bilinear)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        T = 0
        if len(x.shape) == 5:
            B, T, H, W, C = x.shape
            x = rearrange(x, "B T H W C -> (B T) C H W")
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)
        if T:
            out = rearrange(out, "(B T) C H W -> B T H W C", B=B, T=T)

        return out


class ChannelModule(nn.Module):
    def __init__(self, in_channels, ratio=3):
        super(ChannelModule, self).__init__()
        c = in_channels
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.share_liner = nn.Sequential(
            nn.Linear(c, c // ratio),
            nn.ReLU(),
            nn.Linear(c // ratio, c)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.maxpool(inputs).view(inputs.size(0), -1)  # nc
        maxout = self.share_liner(x).unsqueeze(2).unsqueeze(3)  # nchw
        y = self.avgpool(inputs).view(inputs.size(0), -1)
        avgout = self.share_liner(y).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(maxout + avgout)


class SpatialModule(nn.Module):
    def __init__(self):
        super(SpatialModule, self).__init__()
        self.maxpool = torch.max
        self.avgpool = torch.mean
        self.concat = torch.cat
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        maxout, _ = self.maxpool(inputs, dim=1, keepdim=True)  # n1hw
        avgout = self.avgpool(inputs, dim=1, keepdim=True)  # n1hw
        outs = self.concat([maxout, avgout], dim=1)  # n2hw
        outs = self.conv(outs)  # n1hw
        # print(outs.shape)
        return self.sigmoid(outs)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_out = ChannelModule(in_channels)
        self.spatial_out = SpatialModule()

    def forward(self, inputs):
        inputs = inputs.permute(0, 3, 1, 2)
        outs = self.channel_out(inputs) * inputs
        outs = self.spatial_out(outs) * outs
        outs = outs.permute(0, 2, 3, 1)
        return outs


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # x_proj_weight_device = self.x_proj_weight.to(xs.device)  # 正确地移动到设备上
        # x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), x_proj_weight_device)
        # dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # dt_projs_weight_device = self.dt_projs_weight.to(dts.device)  # 正确地移动到设备上
        # dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), dt_projs_weight_device)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)

        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.CBAM = CBAM(hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        input = input.permute(0, 2, 3, 1)
        x = self.CBAM(input) + self.drop_path(self.self_attention(self.ln_1(input)))
        x = x.permute(0, 3, 1, 2)
        return x


class VSSBlock_no_CBAM(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        # self.CBAM = CBAM(hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        input = input.permute(0, 2, 3, 1)
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        x = x.permute(0, 3, 1, 2)
        return x


class Dust_Mamba(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, bilinear=False):
        super(Dust_Mamba, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.bilinear = bilinear
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        n2 = 32
        filters_new = [n2, n2 * 2, n2 * 4, n2 * 8, n2 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.y_Conv1 = conv_block_nested(12, filters_new[0], filters_new[0])
        self.y_Conv2 = conv_block_nested(filters_new[0], filters_new[1], filters_new[1])
        self.y_Conv3 = conv_block_nested(filters_new[1], filters_new[2], filters_new[2])
        self.y_Conv4 = conv_block_nested(filters_new[2], filters_new[3], filters_new[3])
        self.y_Conv5 = conv_block_nested(filters_new[3], filters_new[4], filters_new[4])

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0] + filters_new[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1] + filters_new[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2] + filters_new[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3] + filters_new[3], filters[4], filters[4])
        # ----------------------------------------------------------------
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

        # VSS BLOCK 1
        self.layer_1 = VSSBlock(
            hidden_dim=in_ch,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.1,
            d_state=16,
        )
        # VSS BLOCK 2
        self.layer_2 = VSSBlock(
            hidden_dim=filters[0] + filters_new[0],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.1,
            d_state=16,
        )

    def forward(self, x):
        y = x[:, :12, :, :]

        y1 = self.y_Conv1(y)
        y2 = self.pool(y1)
        y2 = self.y_Conv2(y2)

        y3 = self.pool(y2)
        y3 = self.y_Conv3(y3)

        y4 = self.pool(y3)
        y4 = self.y_Conv4(y4)

        x0_0 = self.conv0_0(self.layer_1(x))
        x1_0 = self.conv1_0(self.layer_2(self.pool(torch.cat((y1, x0_0), dim=1))))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(torch.cat((y2, x1_0), dim=1)))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(torch.cat((y3, x2_0), dim=1)))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(torch.cat((y4, x3_0), dim=1)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class Add_MRDF(nn.Module):
    # Use only MRDF without VSS BLOCK
    def __init__(self, in_ch=3, out_ch=1, bilinear=False):
        super(Add_MRDF, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.bilinear = bilinear
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        n2 = 32
        filters_new = [n2, n2 * 2, n2 * 4, n2 * 8, n2 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ----------------------------------------------------------------
        self.y_Conv1 = conv_block_nested(12, filters_new[0], filters_new[0])
        self.y_Conv2 = conv_block_nested(filters_new[0], filters_new[1], filters_new[1])
        self.y_Conv3 = conv_block_nested(filters_new[1], filters_new[2], filters_new[2])
        self.y_Conv4 = conv_block_nested(filters_new[2], filters_new[3], filters_new[3])
        self.y_Conv5 = conv_block_nested(filters_new[3], filters_new[4], filters_new[4])

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0] + filters_new[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1] + filters_new[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2] + filters_new[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3] + filters_new[3], filters[4], filters[4])

        # ----------------------------------------------------------------
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        y = x[:, :12, :, :]

        y1 = self.y_Conv1(y)
        y2 = self.pool(y1)
        y2 = self.y_Conv2(y2)

        y3 = self.pool(y2)
        y3 = self.y_Conv3(y3)

        y4 = self.pool(y3)
        y4 = self.y_Conv4(y4)

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(torch.cat((y1, x0_0), dim=1)))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(torch.cat((y2, x1_0), dim=1)))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(torch.cat((y3, x2_0), dim=1)))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(torch.cat((y4, x3_0), dim=1)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class Add_VSB(nn.Module):
    # Introduce VSSBLOCK but do not use CBAM
    def __init__(self, in_ch=3, out_ch=1, bilinear=False):
        super(Add_VSB, self).__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.bilinear = bilinear
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        n2 = 32
        filters_new = [n2, n2 * 2, n2 * 4, n2 * 8, n2 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ----------------------------------------------------------------
        self.y_Conv1 = conv_block_nested(12, filters_new[0], filters_new[0])
        self.y_Conv2 = conv_block_nested(filters_new[0], filters_new[1], filters_new[1])
        self.y_Conv3 = conv_block_nested(filters_new[1], filters_new[2], filters_new[2])
        self.y_Conv4 = conv_block_nested(filters_new[2], filters_new[3], filters_new[3])
        self.y_Conv5 = conv_block_nested(filters_new[3], filters_new[4], filters_new[4])

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0] + filters_new[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1] + filters_new[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2] + filters_new[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3] + filters_new[3], filters[4], filters[4])

        # ----------------------------------------------------------------
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

        # VSS BLOCK 1
        self.layer_1 = VSSBlock_no_CBAM(
            hidden_dim=in_ch,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.1,
            d_state=16,
        )
        # VSS BLOCK 2
        self.layer_2 = VSSBlock_no_CBAM(
            hidden_dim=filters[0] + filters_new[0],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.1,
            d_state=16,
        )

    def forward(self, x):
        y = x[:, :12, :, :]

        y1 = self.y_Conv1(y)
        y2 = self.pool(y1)
        y2 = self.y_Conv2(y2)

        y3 = self.pool(y2)
        y3 = self.y_Conv3(y3)

        y4 = self.pool(y3)
        y4 = self.y_Conv4(y4)

        x0_0 = self.conv0_0(self.layer_1(x))
        x1_0 = self.conv1_0(self.layer_2(self.pool(torch.cat((y1, x0_0), dim=1))))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(torch.cat((y2, x1_0), dim=1)))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(torch.cat((y3, x2_0), dim=1)))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(torch.cat((y4, x3_0), dim=1)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class Dust_Mamba_joint_training(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, bilinear=False):
        super(Dust_Mamba_joint_training, self).__init__()
        self.in_channels = in_ch
        # self.out_channels = out_ch
        self.out_channels_1 = 2
        self.out_channels_2 = 7
        self.bilinear = bilinear
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        n2 = 32
        filters_new = [n2, n2 * 2, n2 * 4, n2 * 8, n2 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        # self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        # self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        # self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        # self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        # ----------------------------------------------------------------
        self.y_Conv1 = conv_block_nested(12, filters_new[0], filters_new[0])
        self.y_Conv2 = conv_block_nested(filters_new[0], filters_new[1], filters_new[1])
        self.y_Conv3 = conv_block_nested(filters_new[1], filters_new[2], filters_new[2])
        self.y_Conv4 = conv_block_nested(filters_new[2], filters_new[3], filters_new[3])
        self.y_Conv5 = conv_block_nested(filters_new[3], filters_new[4], filters_new[4])

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0] + filters_new[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1] + filters_new[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2] + filters_new[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3] + filters_new[3], filters[4], filters[4])
        # ----------------------------------------------------------------
        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final_1 = nn.Conv2d(filters[0], self.out_channels_1, kernel_size=1)
        self.final_2 = nn.Conv2d(filters[0], self.out_channels_2, kernel_size=1)

        self.layer_1 = VSSBlock(
            hidden_dim=in_ch,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.1,
            d_state=16,
        )

        self.layer_2 = VSSBlock(
            hidden_dim=filters[0] + filters_new[0],
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0.1,
            d_state=16,
        )

        # self.layer_3 = VSSBlock(
        #     hidden_dim=filters[1] + filters_new[1],
        #     norm_layer=nn.LayerNorm,
        #     attn_drop_rate=0.1,
        #     d_state=16,
        # )

    def forward(self, x):
        y = x[:, :12, :, :]

        y1 = self.y_Conv1(y)
        y2 = self.pool(y1)
        y2 = self.y_Conv2(y2)

        y3 = self.pool(y2)
        y3 = self.y_Conv3(y3)

        y4 = self.pool(y3)
        y4 = self.y_Conv4(y4)

        x0_0 = self.conv0_0(self.layer_1(x))
        x1_0 = self.conv1_0(self.layer_2(self.pool(torch.cat((y1, x0_0), dim=1))))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(torch.cat((y2, x1_0), dim=1)))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(torch.cat((y3, x2_0), dim=1)))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(torch.cat((y4, x3_0), dim=1)))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output_1 = self.final_1(x0_4)
        output_2 = self.final_2(x0_4)
        # print(output_1.shape,output_2.shape)
        return output_1, output_2