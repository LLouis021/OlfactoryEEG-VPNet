import torch
import torch.nn as nn
from CWTLayer import vp_layer


# =========================================================
# Orthogonality Loss
# =========================================================
def orthogonality_loss(W):
    if W.dim() == 1:
        return torch.norm(W, p=2)

    WT_W = W.T @ W
    I = torch.eye(WT_W.size(0), device=W.device, dtype=W.dtype)
    return torch.norm(WT_W - I, p="fro")


# =========================================================
# v7.0 New: SE Channel Attention Module
# =========================================================
class SEChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention.
    Global information aggregation and adaptive weighting for channel dimension of 1D feature maps.
    
    Input:  [B, C, T]
    Output: [B, C, T] (channel weighted)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # [B, C, 1]
            nn.Flatten(),                       # [B, C]
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        w = self.se(x).unsqueeze(-1)  # [B, C, 1]
        return x * w


# =========================================================
# CBAM Attention Module (Channel + Spatial)
# =========================================================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) for 1D feature maps.
    Combines channel attention and spatial (temporal) attention.
    
    Input:  [B, C, T]
    Output: [B, C, T] (channel and spatial weighted)
    """
    def __init__(self, channels, reduction=4, kernel_size=7):
        super().__init__()
        mid = max(channels // reduction, 4)
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )
        # Also use max pooling for channel attention
        self.channel_att_max = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        
        # Spatial (temporal) attention
        padding = kernel_size // 2
        self.spatial_att = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, T]
        # Channel attention: combine avg and max pooling
        avg_out = self.channel_att(x)  # [B, C]
        max_out = self.channel_att_max(x)  # [B, C]
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(-1)  # [B, C, 1]
        x = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, T]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, T]
        spatial_in = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, T]
        spatial_att = self.spatial_att(spatial_in)  # [B, 1, T]
        x = x * spatial_att
        
        return x


# =========================================================
# Multi-Scale Residual Branch
# =========================================================
class MultiScaleResidualBranch(nn.Module):
    """
    Multi-scale residual branch with parallel convolutions of different kernel sizes.
    Captures both local and global temporal patterns.
    
    Input:  [B, 30, T] (30 channels, T timepoints)
    Output: [B, out_dim]
    """
    def __init__(self, in_channels=30, out_dim=64, kernel_sizes=[3, 7, 15]):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_dim, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.branches.append(branch)
        
        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * len(kernel_sizes), out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x):
        # x: [B, 30, T]
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)  # [B, out_dim, 1]
            out = out.squeeze(-1)  # [B, out_dim]
            branch_outputs.append(out)
        
        # Concatenate and fuse
        concat = torch.cat(branch_outputs, dim=1)  # [B, out_dim * num_branches]
        output = self.fusion(concat)  # [B, out_dim]
        return output


# =========================================================
# Spatial Branch: Cross-Channel Convolution
# =========================================================
class SpatialBranch(nn.Module):
    """
    Models spatial relationships between EEG channels.

    Input:  [B, 30, T] (30 electrodes, T timepoints)
    Output: [B, out_dim]

    Method: Treat the 30 channels as a spatial dimension.
            1) Reshape to [B, 1, 30, T]
            2) 2D conv (kernel_h=30, kernel_w=15) → covers all channels at once
            3) Temporal pooling → spatial feature
    """

    def __init__(self, in_channels=30, time_length=512, out_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            # Layer 1: Cover all channels + local temporal patterns
            nn.Conv2d(1, 32, kernel_size=(in_channels, 15), padding=(0, 7)),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Layer 2: Further temporal abstraction
            nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.proj = nn.Linear(64, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, 30, T]
        x = x.unsqueeze(1)         # [B, 1, 30, T]
        x = self.conv(x)           # [B, 64, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, 64]
        x = self.proj(x)           # [B, out_dim]
        x = self.norm(x)
        return x


# =========================================================
# 🧪 VPNet EEG (v7.0 Optimized)
# =========================================================
class VPCWTNN(nn.Module):

    def __init__(self,
                 wavegenfun,
                 nparams,
                 input_length,
                 vp_latent_dim,
                 vp_target,
                 p, r, b_min, a, b,
                 neuron_n=[128, 64],
                 penalty=0.01,
                 init_vp=None,
                 device=None,
                 ortho_lambda=1e-3,

                # =========================
                # ABLATION FLAGS
                # =========================
                use_vp=True,
                use_residual=True,
                use_bn=True,
                use_fusion=True,
                use_ortho=True,
                use_spatial=False,
                # v7.0 new parameters
                use_se=True,
                se_reduction=4,
                j_vp_weight=0.01,
                # v8.0 new: CBAM attention (channel + spatial)
                use_cbam=False,
                # v8.0 new: multi-scale residual branch
                use_multiscale_residual=False):

        super().__init__()

        self.use_vp = use_vp
        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_fusion = use_fusion
        self.use_ortho = use_ortho
        self.use_spatial = use_spatial
        self.use_se = use_se
        self.use_cbam = use_cbam
        self.use_multiscale_residual = use_multiscale_residual





        self.device = device
        self.ortho_lambda = ortho_lambda
        self.vp_latent_dim = vp_latent_dim
        self.j_vp_weight = j_vp_weight  # v7.0: reduced from 0.1 to 0.01 to avoid over-constraining VP branch

        self.flat_dim = 30 * input_length
        self.vp_coeff_flat_dim = 30 * vp_latent_dim

        # =========================
        # VP layer
        # =========================
        if self.use_vp:
            self.vp_layer = vp_layer(
                wavegenfun,
                n_in=input_length,
                p=p, r=r, b_min=b_min,
                a=a, b=b,
                target=vp_target,
                n_out=vp_latent_dim,
                nparams=nparams,
                device=device,
                penalty=penalty,
                init=init_vp
            )
            self.vp_proj = nn.Linear(self.vp_coeff_flat_dim, vp_latent_dim)

        # =========================
        # BatchNorm
        # =========================
        if self.use_bn:
            self.ln_vp = nn.LayerNorm(vp_latent_dim)
            self.ln_res = nn.LayerNorm(vp_latent_dim)

        # =========================
        # Residual branch + v7.0 SE Channel Attention
        # =========================
        if self.use_residual:
            if self.use_multiscale_residual:
                # v8.0 new: multi-scale residual branch
                self.res_branch = MultiScaleResidualBranch(
                    in_channels=30,
                    out_dim=vp_latent_dim,
                    kernel_sizes=[3, 7, 15]
                )
            else:
                # Original residual branch
                self.res_cnn = nn.Sequential(
                    nn.Conv1d(in_channels=30, out_channels=32, kernel_size=15, padding=7),
                    nn.BatchNorm1d(32),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2),

                    nn.Conv1d(in_channels=32, out_channels=vp_latent_dim, kernel_size=7, padding=3),
                    nn.BatchNorm1d(vp_latent_dim),
                    nn.GELU(),

                    # Global pooling to compress temporal dimension
                    nn.AdaptiveAvgPool1d(1)
                )
            # v7.0 new: Add SE channel attention after the second CNN layer
            # v8.0: CBAM attention (channel + spatial) replaces SE if use_cbam=True
            # Note: SE/CBAM is applied before GlobalPool, need to call manually in forward
            if not self.use_multiscale_residual:  # Only apply attention to original residual branch
                if self.use_cbam:
                    self.attention_module = CBAM(vp_latent_dim, reduction=se_reduction)
                elif self.use_se:
                    self.attention_module = SEChannelAttention(vp_latent_dim, reduction=se_reduction)

        # =========================
        # Spatial branch
        # =========================
        if self.use_spatial:
            self.spatial_branch = SpatialBranch(
                in_channels=30,
                time_length=input_length,
                out_dim=vp_latent_dim
            )

        # =========================
        # Classifier (MLP head)
        # =========================
        layers = []
        dim = vp_latent_dim

        for h in neuron_n:
            layers += [
                nn.Linear(dim, h),
                nn.GELU(),
                nn.Dropout(0.5)
            ]
            dim = h

        layers.append(nn.Linear(dim, 13))
        self.classifier = nn.Sequential(*layers)

        # Baseline projection (for ablation)
        if not self.use_vp and not self.use_residual and not self.use_spatial:
            self.baseline_proj = nn.Linear(self.flat_dim, vp_latent_dim)

        # SE-Style Cross Attention fusion gate
        if self.use_fusion:
            n_branches = int(self.use_vp) + int(self.use_residual) + int(self.use_spatial)
            concat_dim = vp_latent_dim * max(n_branches, 2)

            self.attention_net = nn.Sequential(
                nn.Linear(concat_dim, vp_latent_dim),
                nn.GELU(),
                nn.Linear(vp_latent_dim, vp_latent_dim)
            )
            # Second gate for 3-way fusion
            if self.use_spatial:
                self.attention_net2 = nn.Sequential(
                    nn.Linear(concat_dim, vp_latent_dim),
                    nn.GELU(),
                    nn.Linear(vp_latent_dim, vp_latent_dim)
                )

    # =========================================================
    # Forward
    # =========================================================
    def forward(self, x, return_embedding=False):

        # ---------------- VP branch ----------------
        if self.use_vp:
            vp_coeffs, y_est = self.vp_layer(x)
            residual_signal = x - y_est

            # J_VP reconstruction loss (v7.0: weight reduced to 0.01)
            self.j_vp_loss = torch.mean(
                torch.norm(residual_signal, dim=(1,2))**2 /
                (torch.norm(x, dim=(1,2))**2 + 1e-8)
            )

            vp_feat = vp_coeffs.reshape(vp_coeffs.size(0), -1)
            vp_feat = self.vp_proj(vp_feat)
            if self.use_bn:
                vp_feat = self.ln_vp(vp_feat)
        else:
            vp_feat = torch.zeros(x.size(0), self.vp_latent_dim, device=x.device)

        # ---------------- Residual branch + SE Attention ----------------
        if self.use_residual:
            if self.use_multiscale_residual:
                # Multi-scale residual branch (already pooled)
                residual = self.res_branch(x)  # [B, vp_latent_dim]
            else:
                # Extract intermediate features for attention (SE or CBAM)
                res_feat = self.res_cnn[:5](x)  # Conv1d→BN→GELU→MaxPool→Conv1d→BN→GELU
                # Apply channel/spatial attention if enabled
                if self.use_cbam or self.use_se:
                    res_feat = self.attention_module(res_feat)
                # Global pooling
                residual = nn.functional.adaptive_avg_pool1d(res_feat, 1).squeeze(-1)
            if self.use_bn:
                residual = self.ln_res(residual)
        else:
            residual = torch.zeros(x.size(0), self.vp_latent_dim, device=x.device)

        # ---------------- Spatial branch ----------------
        if self.use_spatial:
            spatial = self.spatial_branch(x)
        else:
            spatial = torch.zeros(x.size(0), self.vp_latent_dim, device=x.device)

        # ---------------- Fusion ----------------
        if self.use_fusion:
            branches = []
            if self.use_vp:
                branches.append(vp_feat)
            if self.use_residual:
                branches.append(residual)
            if self.use_spatial:
                branches.append(spatial)

            if len(branches) >= 2:
                concat_feat = torch.cat(branches, dim=1)
                gate = torch.sigmoid(self.attention_net(concat_feat))
                if len(branches) == 2:
                    x = (gate * branches[0]) + ((1 - gate) * branches[1])
                else:
                    gate2 = torch.sigmoid(self.attention_net2(concat_feat))
                    x = gate * branches[0] + (1 - gate) * gate2 * branches[1] + (1 - gate) * (1 - gate2) * branches[2]
            elif len(branches) == 1:
                x = branches[0]
            else:
                x_flat = x.reshape(x.size(0), -1)
                x = self.baseline_proj(x_flat)
        else:
            active = []
            if self.use_vp:
                active.append(vp_feat)
            if self.use_residual:
                active.append(residual)
            if self.use_spatial:
                active.append(spatial)

            if len(active) == 0:
                x_flat = x.reshape(x.size(0), -1)
                x = self.baseline_proj(x_flat)
            elif len(active) == 1:
                x = active[0]
            else:
                x = sum(active)

        if return_embedding:
            return x

        return self.classifier(x)

    def orthogonality_regularization(self):
        if not self.use_ortho or not self.use_vp or not hasattr(self, 'vp_layer'):
            device = next(self.classifier.parameters()).device
            return torch.tensor(0.0, device=device)
        return orthogonality_loss(self.vp_layer.weight)

    def compute_total_loss(self, pred, target=None, base_loss_fn=None, base_loss_val=None):
        if base_loss_val is not None:
            loss = base_loss_val
        else:
            loss = base_loss_fn(pred, target)

        loss += self.ortho_lambda * self.orthogonality_regularization()

        # v7.0: J_VP weight reduced from 0.1 to configurable 0.01
        if self.use_vp and hasattr(self, 'j_vp_loss'):
            loss += self.j_vp_weight * self.j_vp_loss

        return loss
