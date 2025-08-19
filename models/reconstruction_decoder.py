import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class ConvBNAct(nn.Module):
    """
    A tiny conv block: 3x3 -> BN -> GELU (twice).
    Keeps things lightweight but expressive for reconstruction.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)

"""
    Reconstruction decoder with FPN-like top-down pathway + U-Net-style skip connections.

    Expected inputs:
      - A list or tuple (or dict values) of feature maps from shallow->deep OR deep->shallow.
        You can pass any length >= 2; commonly 3–4 scales from a ConvNeXt encoder.
        Example shapes (B, C_i, H_i, W_i) where spatial strides typically double each deeper level.

    What it does:
      1) Projects ea -ch incoming feature to a common 'feat_ch' via 1x1 conv (lateral convs).
      2) Starts from the deepest map, upsample x2, fuse with the skip from the next shallower stage
         (either add or concat), then refine with a small conv block.
      3) Repeats until the shallowest resolution is reached.
      4) Optionally upsamples extra times to reach the original input resolution (if needed).
      5) Outputs a 3-channel reconstruction with configurable activation:
         - 'identity'  : raw logits in the same normalized space as your inputs
         - 'tanh'      : (-1, 1) range (use if your inputs are normalized to [-1, 1])
         - 'sigmoid'   : (0, 1) range (use if you compute MSE in unnormalized pixel space)

    Notes:
      - Set `fuse='concat'` if you prefer concatenation before refinement (more capacity).
        Set `fuse='add'` if you want lighter compute (requires equal channels).
      - This module does NOT denormalize; compute your recon loss consistently with how you normalized inputs.
      - Use `debug=True` to print shapes for quick verification.

    Args:
      in_channels: list[int] channel sizes for each encoder stage you will pass in (ordered shallow->deep OR deep->shallow).
      feat_ch:     int, internal feature width after lateral projection.
      out_activation: 'identity' | 'tanh' | 'sigmoid'
      fuse: 'concat' | 'add'
      extra_upsamples: int, how many times to upsample x2 after finishing the top-down (to hit input size if needed).
      debug: bool, prints shapes.
    """

# class ConvNeXtReconstructionDecoder(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         feat_ch=256,
#         out_activation="identity",
#         fuse="concat",
#         extra_upsamples=0,
#         debug=False,
#         **kwargs
#     ):
#         super().__init__()
#         assert len(in_channels) >= 2, "Pass at least two feature levels"
#         assert out_activation in ("identity", "tanh", "sigmoid")
#         assert fuse in ("concat", "add")
#         self.out_activation = out_activation
#         self.fuse = fuse
#         self.debug = debug
#         self.extra_upsamples = int(extra_upsamples)
#         self.feat_ch = feat_ch

#         # Lateral projections — now will be rebuilt dynamically if channels differ
#         self.laterals = nn.ModuleList([nn.Conv2d(c, feat_ch, kernel_size=1) for c in in_channels])

#         # Refinement blocks after fusion
#         self.refines = nn.ModuleList()
#         for _ in range(len(in_channels) - 1):
#             in_ch = feat_ch * 2 if fuse == "concat" else feat_ch
#             self.refines.append(ConvBNAct(in_ch, feat_ch))

#         # Final head
#         self.head = nn.Sequential(
#             nn.Conv2d(feat_ch, feat_ch // 2, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(feat_ch // 2),
#             nn.GELU(),
#             nn.Conv2d(feat_ch // 2, 3, kernel_size=1, bias=True),
#         )

#     @staticmethod
#     def _to_tensor(x):
#         if hasattr(x, "tensors"):
#             return x.tensors
#         if hasattr(x, "decompose"):
#             t, _ = x.decompose()
#             return t
#         return x

#     def _order_features_by_resolution(self, feats):
#         def unwrap_feature(f):
#             if hasattr(f, "tensors"):
#                 return f.tensors
#             return f

#         feats_unwrapped = []
#         for f in feats:
#             if isinstance(f, (list, tuple)):
#                 feats_unwrapped.extend([unwrap_feature(ff) for ff in f])
#             else:
#                 feats_unwrapped.append(unwrap_feature(f))

#         for i, f in enumerate(feats_unwrapped):
#             if not torch.is_tensor(f):
#                 raise TypeError(f"Feature at index {i} is not a torch.Tensor: {type(f)}")

#         sizes = [f.shape[-2] * f.shape[-1] for f in feats_unwrapped]
#         idx_sorted = sorted(range(len(sizes)), key=lambda k: sizes[k], reverse=True)
#         feats_sorted = [feats_unwrapped[i] for i in idx_sorted]

#         if self.debug:
#             print("[Decoder] Feature resolutions sorted (H*W):", [sizes[i] for i in idx_sorted])

#         return feats_sorted, idx_sorted

#     def forward(self, features):
#         if isinstance(features, dict):
#             feats = list(features.values())
#         else:
#             feats = list(features)

#         if self.debug:
#             print("[ReconDec DEBUG] Incoming feature shapes:")
#             for i, f in enumerate(feats):
#                 tf = self._to_tensor(f)
#                 print(f"  L{i}: {tuple(tf.shape)}")

#         feats_sorted, _ = self._order_features_by_resolution(feats)

#         # Dynamically match channels to feat_ch
#         feats_lat = []
#         for i, f in enumerate(feats_sorted):
#             f_t = self._to_tensor(f)
#             in_c = f_t.shape[1]
#             if in_c != self.feat_ch:
#                 adapter = nn.Conv2d(in_c, self.feat_ch, kernel_size=1).to(f_t.device)
#                 f_t = adapter(f_t)
#             feats_lat.append(f_t)

#         x = feats_lat[0]
#         if self.debug:
#             print(f"[ReconDec DEBUG] Start (deepest) after lateral: {tuple(x.shape)}")

#         for k in range(1, len(feats_lat)):
#             target_h, target_w = feats_lat[k].shape[-2:]
#             x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

#             if self.fuse == "concat":
#                 x = torch.cat([x, feats_lat[k]], dim=1)
#             else:
#                 x = x + feats_lat[k]

#             if (k - 1) < len(self.refines):
#                 x = self.refines[k - 1](x)

#             if self.debug:
#                 print(f"[ReconDec DEBUG] After fuse+refine stage {k}: {tuple(x.shape)}")

#         for t in range(self.extra_upsamples):
#             x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
#             if self.debug:
#                 print(f"[ReconDec DEBUG] After extra upsample {t+1}: {tuple(x.shape)}")

#         recon = self.head(x)

#         if self.out_activation == "tanh":
#             recon = torch.tanh(recon)
#         elif self.out_activation == "sigmoid":
#             recon = torch.sigmoid(recon)

#         if self.debug:
#             print(f"[ReconDec DEBUG] Output reconstruction: {tuple(recon.shape)}")

#         return recon


class ConvNeXtReconstructionDecoder(nn.Module):
    def __init__(
        self,
        feat_ch=256,
        out_activation="identity",
        fuse="concat",
        extra_upsamples=0,
        debug=False,
        use_checkpoint=True,   # <-- new flag
        **kwargs
    ):
        super().__init__()
        assert out_activation in ("identity", "tanh", "sigmoid")
        assert fuse in ("concat", "add")
        self.out_activation = out_activation
        self.fuse = fuse
        self.debug = debug
        self.extra_upsamples = int(extra_upsamples)
        self.feat_ch = feat_ch
        self.use_checkpoint = use_checkpoint

        # Lazy init placeholders
        self.laterals = None
        self.refines = None

        # Head remains fixed
        self.head = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch // 2),
            nn.GELU(),
            nn.Conv2d(feat_ch // 2, 3, kernel_size=1, bias=True),
        )

    @staticmethod
    def _to_tensor(x):
        if hasattr(x, "tensors"):
            return x.tensors
        if hasattr(x, "decompose"):
            t, _ = x.decompose()
            return t
        return x

    def _order_features_by_resolution(self, feats):
        feats_unwrapped = [self._to_tensor(f) if not isinstance(f, (list, tuple)) else self._to_tensor(f[0]) for f in feats]
        sizes = [f.shape[-2] * f.shape[-1] for f in feats_unwrapped]
        idx_sorted = sorted(range(len(sizes)), key=lambda k: sizes[k], reverse=True)
        feats_sorted = [feats[i] for i in idx_sorted]
        if self.debug:
            print("[Decoder] Feature resolutions sorted (H*W):", [sizes[i] for i in idx_sorted])
        return feats_sorted, idx_sorted

    def forward(self, features):
        if isinstance(features, dict):
            feats = list(features.values())
        else:
            feats = list(features)

        feats_sorted, idx_sorted = self._order_features_by_resolution(feats)

        # ---- Lazy init ----
        if self.laterals is None:
            self.laterals = nn.ModuleList([
                nn.Conv2d(self._to_tensor(f).shape[1], self.feat_ch, kernel_size=1).to(f.device)
                for f in feats_sorted
            ])
            self.refines = nn.ModuleList()
            for k in range(1, len(feats_sorted)):
                in_ch = self.feat_ch * 2 if self.fuse == "concat" else self.feat_ch
                self.refines.append(ConvBNAct(in_ch, self.feat_ch).to(self._to_tensor(feats_sorted[k]).device))

        # Laterals
        feats_lat = [self.laterals[i](self._to_tensor(f)) for i, f in enumerate(feats_sorted)]

        x = feats_lat[0]

        # Fuse + refine stages with checkpoint
        for k in range(1, len(feats_lat)):
            target_h, target_w = feats_lat[k].shape[-2:]
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

            if self.fuse == "concat":
                x = torch.cat([x, feats_lat[k]], dim=1)
            else:
                x = x + feats_lat[k]

            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.refines[k - 1], x)
            else:
                x = self.refines[k - 1](x)

        # Extra upsampling
        for t in range(self.extra_upsamples):
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)

        # Final head
        recon = self.head(x)
        if self.out_activation == "tanh":
            recon = torch.tanh(recon)
        elif self.out_activation == "sigmoid":
            recon = torch.sigmoid(recon)

        return recon