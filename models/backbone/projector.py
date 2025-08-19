# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Projector
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        #if x.dtype == torch.half:
        #    x = x / (x.max() + self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """ get activation """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", 'leakyrelu', 'lrelu']:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ConvX(nn.Module):
    """ Conv-bn module"""
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act='relu', layer_norm=False, rms_norm=False):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel,
                              stride=stride, padding=padding, groups=groups,
                              dilation=dilation, bias=False)
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm('LN', out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """ forward """
        out = self.act(self.bn(self.conv(x)))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu', layer_norm=False, rms_norm=False):
        """ ch_in, ch_out, shortcut, groups, kernels, expand """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu', layer_norm=False, rms_norm=False):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        #print(f"[C2f INIT] c1: {c1}, c2: {c2}, n: {n}, c: {self.c}, shortcut: {shortcut}, g: {g}, e: {e}")
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x):
        #print(f"[C2f DEBUG] Input shape: {x.shape}")
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for i, m in enumerate(self.m):
            y.append(m(y[-1]))
            #print(f"[C2f DEBUG] After Bottleneck {i}: {y[-1].shape}")
        out = self.cv2(torch.cat(y, 1))
        #print(f"[C2f DEBUG] Output shape: {out.shape}")
        return out


class MultiScaleProjector(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        layer_norm=False,
        rms_norm=False,
        survival_prob=1.0,
        force_drop_last_n_features=0,
    ):
        super().__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        stages_sampling = []
        stages = []

        #print(f"\n[MultiScaleProjector INIT]")
        #print(f"Input channels: {in_channels}")
        #print(f"Output channels: {out_channels}")
        #print(f"Scale factors: {scale_factors}")

        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1),
                get_norm("LN", out_channels),
                nn.GELU()
            ) for in_ch in in_channels
        ])

        self.use_extra_pool = False

        for scale in scale_factors:
            #print(f"[SCALE {scale}] Building sampling layers...")
            stages_sampling.append([])

            for i, in_dim in enumerate(in_channels):
                input_dim = out_channels
                out_dim = input_dim
                layers = []

                if scale == 4.0:
                    layers.extend([
                        nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=2, stride=2),
                        get_norm('LN', input_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(input_dim // 2, input_dim // 4, kernel_size=2, stride=2),
                    ])
                    out_dim = input_dim // 4
                elif scale == 2.0:
                    layers.append(nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=2, stride=2))
                    out_dim = input_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.append(ConvX(input_dim, input_dim, 3, 2, layer_norm=layer_norm))
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError(f"Unsupported scale_factor: {scale}")

                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)

            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            fused_input_channels = out_channels * len(in_channels)
            #print(f"[STAGE DEBUG] Fused input channels for C2f: {fused_input_channels}")

            layers = [
                C2f(c1=fused_input_channels, c2=out_channels, n=num_blocks, layer_norm=layer_norm),
                get_norm('LN', out_channels),
            ]
            stages.append(nn.Sequential(*layers))

        self.stages_sampling = nn.ModuleList(stages_sampling)
        #print(f"[STAGES SAMPLING] {len(self.stages_sampling)} stages")
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        num_features = len(x)

        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                x[-(i+1)] = torch.zeros_like(x[-(i+1)])

        results = []

        for i, stage in enumerate(self.stages):
            feat_fuse = []
            target_size = None

            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                projected_feat = self.proj_layers[j](x[j])
                #print(f"[FORWARD DEBUG] Scale {i}, Feat {j} projected shape: {projected_feat.shape}")
                # sampled_feat = stage_sampling(projected_feat)
                sampled_feat=projected_feat 
                #print(f"[FORWARD DEBUG] Scale {i}, Feat {j} shape: {sampled_feat.shape}")

                if target_size is None:
                    target_size = sampled_feat.shape[-2:]
                elif sampled_feat.shape[-2:] != target_size:
                    sampled_feat = F.interpolate(sampled_feat, size=target_size, mode='bilinear', align_corners=False)
                    #print(f" - Resized feat {j} to {target_size}")

                feat_fuse.append(sampled_feat)

            feat_fuse = torch.cat(feat_fuse, dim=1) if len(feat_fuse) > 1 else feat_fuse[0]
            #print(f"[FORWARD DEBUG] Fused feature shape (before C2f): {feat_fuse.shape}")
            fused_out = stage(feat_fuse)
            #print(f"[FORWARD DEBUG] Fused output shape: {fused_out.shape}")
            results.append(fused_out)

            if self.use_extra_pool:
                pooled = F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
                results.append(pooled)

        return results


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim*2, layer_norm=True, act='silu')
            self.convx2 = ConvX(in_dim*2, out_dim, layer_norm=True, act='silu')
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act='silu')
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act='silu')
        self.ln = get_norm('LN', out_dim)

    def forward(self, x):
        """ forward """
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]
