import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
        print(f"k: {k}")
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        print(f"p: {p}")
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        print("Conv start")
        print(f"Input conv Shape: {x.shape}")
        y = self.conv(x)
        print(f"self.conv Shape: {y.shape}")
        b = self.bn(y)
        print(f"self.bn Shape: {b.shape}")
        act = self.act(b)
        print(f"self.act/output Shape: {act.shape}")
        print("Conv end")
        return act

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class EMA(nn.Module):
    def __init__(self, in_channels, groups=4):
        """
        Efficient Multi-scale Attention block with cross-spatial learning.

        Args:
            in_channels (int): Number of input channels.
            groups (int): Number of groups for grouped attention.
        """
        super(EMA, self).__init__()
        self.groups = groups
        self.group_channels = in_channels // groups

        # Convolution layer for re-weighting
        self.conv_1_1 = Conv(in_channels, in_channels, k=1, s=1, p=0)
        self.conv_3_3 = Conv(in_channels, in_channels, k=3, s=1, p=1, g=groups)

        # Group normalization
        self.group_norm = nn.GroupNorm(groups, in_channels)

    def forward(self, x):
        """
        Forward pass of EMA.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        print("EMA start")
        print(f"EMA input shape: {x.shape}")
        B, C, H, W = x.shape # 1, 256, 8, 8
        g = self.groups # 4
        Cg = self.group_channels # 64

        # Conv 3X3
        conv_3_3 = self.conv_3_3(x)
        print(f"EMA conv_3_3 shape: {conv_3_3.shape}")

        x_avg_pool = F.adaptive_avg_pool2d(x, output_size=(1, W))  # (B, C, 1, W)
        print(f"EMA x_avg_pool shape: {x_avg_pool.shape}")
        y_avg_pool = F.adaptive_avg_pool2d(x, output_size=(H, 1))  # (B, C, H, 1)
        print(f"EMA y_avg_pool shape: {y_avg_pool.shape}")
        y_avg_pool = y_avg_pool.permute(0, 1, 3, 2)  # (B, C, 1, H)

        # x_groups = x.view(B, g, Cg, H, W)  # (B, g, Cg, H, W)
        # print(f"EMA x_groups shape: {x_groups.shape}")

        #contating x1 and y
        concat_pool = torch.cat([x_avg_pool, y_avg_pool], dim=3)  # (B, C, H, W+H)
        print(f"EMA concat_pool shape: {concat_pool.shape}")

        #conv layer
        conv1x1 = self.conv_1_1(concat_pool)  # (B, g, 1, W+H)
        print(f"EMA conv1x1 shape: {conv1x1.shape}")

        #Split the conv1x1 into two equal parts
        x_attention, y_attention = torch.split(conv1x1, [W, H], dim=-1)
        # x_attention, y_attention = torch.split(conv1x1, conv1x1.shape[3] // 2, dim=3)
        print(f"EMA x_attention shape: {x_attention.shape}") # (B, C, 1, W)
        print(f"EMA y_attention shape: {y_attention.shape}") # (B, C, 1, H)

        #Sigmoid
        x_attention = torch.sigmoid(x_attention) # (B, C, 1, W)
        y_attention = torch.sigmoid(y_attention) # (B, C, 1, H)

        x_attention = x_attention.unsqueeze(2) # (B, C, 1, 1, W)
        y_attention = y_attention.permute(0, 1, 3, 2).unsqueeze(3) # (B, C, H, 1, 1)
        print(f"EMA x_attention unsqueeze shape: {x_attention.shape}")
        print(f"EMA y_attention unsqueeze shape: {y_attention.shape}")

        #Reweight
        x_reweighted = x * x_attention.squeeze(2) # (B, C, H, W)
        x_reweighted = x * y_attention.squeeze(3) # (B, C, H, W)
        print(f"EMA x shape: {x_reweighted.shape}")

        #Group norm
        x_GN_reweighted = self.group_norm(x_reweighted) # (B, C, H, W)
        print(f"EMA x_GN_reweighted shape: {x_GN_reweighted.shape}")

        #Group formation
        x_groups = x_GN_reweighted.view(B, g, Cg, H, W)  # (B, g, Cg, H, W)
        print(f"EMA x_groups shape: {x_groups.shape}")
        conv_3_3_groups = conv_3_3.view(B, g, Cg, H, W)
        print(f"EMA conv_3_3_view shape: {conv_3_3_groups.shape}")

        #Global pool
        x_global_pool = F.adaptive_avg_pool2d(x_groups, output_size=(1, 1))  # (B, g, Cg, 1, 1)
        print(f"EMA x_global_pool shape: {x_global_pool.shape}")
        conv_3_3_pool = F.adaptive_avg_pool2d(conv_3_3_groups, output_size=(1, 1))  # (B, g, Cg, 1, 1)
        print(f"EMA conv_3_3_pool shape: {conv_3_3_pool.shape}")

        #Softmax
        x_g_pool_softmax = F.softmax(x_global_pool, dim=1)  # (B, g, Cg, 1, 1)
        print(f"EMA x_g_pool_softmax shape: {x_g_pool_softmax.shape}")
        conv_3_3_pool_softmax = F.softmax(conv_3_3_pool, dim=1)  # (B, g, Cg, 1, 1)
        print(f"EMA conv_3_3_pool_softmax shape: {conv_3_3_pool_softmax.shape}")

        #Matmul
        x_g_pool_softmax_reshaped = x_g_pool_softmax.expand_as(x_groups)  # (B, g, Cg, H, W)
        print(f"EMA x_g_pool_softmax_reshaped shape: {x_g_pool_softmax_reshaped.shape}")
        matmul_1 = x_g_pool_softmax_reshaped * conv_3_3_groups
        print(f"EMA matmul_1 shape: {matmul_1.shape}")
        conv_3_3_pool_reshaped = conv_3_3_pool_softmax.expand_as(conv_3_3_groups)  # (B, g, Cg, H, W)
        print(f"EMA conv_3_3_pool_reshaped shape: {conv_3_3_pool_reshaped.shape}")
        matmul_2 = conv_3_3_pool_reshaped * x_groups
        print(f"EMA matmul_2 shape: {matmul_2.shape}")

        #Concat matmul and sigmoid
        spatial_attention = matmul_1 + matmul_2  # (B, g, Cg, H, W)
        print(f"EMA spatial_attention shape: {spatial_attention.shape}")
        spatial_attention = torch.sigmoid(spatial_attention)
        print(f"EMA spatial_attention shape: {spatial_attention.shape}")

        # Adjust sigmoid shape
        spatial_attention = spatial_attention.view(B, g * Cg, H, W)
        print(f"EMA spatial_attention shape: {spatial_attention.shape}")

        # Reweight sigmoid matmul
        x_reweighted = x * spatial_attention  # Element-wise multiplication (broadcasting across channels)
        print(f"EMA Reweighted x shape: {x_reweighted.shape}")

        # Return the reweighted tensor
        return x_reweighted

if __name__ == "__main__":
    x = torch.randn(32, 256, 12, 21)
    model = EMA(256, 32)
    out = model(x)
