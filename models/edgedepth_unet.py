
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with GroupNorm and residual learning."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        groups = 8 if out_channels % 8 == 0 else 1

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)

        self.activation = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        return self.activation(x + residual)


class DownBlock(nn.Module):
    """Feature extraction followed by spatial downsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.block(x)
        downsampled = self.pool(features)
        return features, downsampled


class UpBlock(nn.Module):
    """Upsample, concatenate skip features, and refine."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.block = ConvBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class EdgeDepthUNet(nn.Module):
    """
    RGB-guided residual U-Net for depth upsampling.

    Input channels:
        0-2: RGB image, normalized to [0, 1]
        3:   Low-resolution depth
        4:   Validity mask

    Input shape:
        [B, 5, H, W]

    Output:
        {
            "depth":      [B, 1, H*scale, W*scale],
            "boundary":   [B, 1, H*scale, W*scale],
            "residual":   [B, 1, H*scale, W*scale]
        }

    Depth values should use consistent units during training and inference.
    """

    def __init__(
        self,
        in_channels=5,
        base_channels=32,
        scale_factor=2
    ):
        super().__init__()

        if in_channels != 5:
            raise ValueError(
                "EdgeDepthUNet expects exactly 5 input channels."
            )

        if scale_factor not in (2, 4, 8):
            raise ValueError("Scale factor must be 2, 4, or 8.")

        self.scale_factor = scale_factor

        # Encoder.
        self.enc1 = DownBlock(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)

        # Bottleneck.
        self.bottleneck = ConvBlock(
            base_channels * 4,
            base_channels * 8
        )

        # Decoder.
        self.up3 = UpBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4
        )

        self.up2 = UpBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )

        self.up1 = UpBlock(
            base_channels * 2,
            base_channels,
            base_channels
        )

        # Shared feature refinement.
        self.shared_refine = ConvBlock(
            base_channels,
            base_channels
        )

        # Predicts a residual correction to interpolated depth.
        self.depth_head = nn.Conv2d(
            base_channels, 1, kernel_size=3, padding=1
        )

        # Predicts boundary probability.
        self.boundary_head = nn.Conv2d(
            base_channels, 1, kernel_size=3, padding=1
        )

        # Start with a small residual correction.
        nn.init.zeros_(self.depth_head.weight)
        nn.init.zeros_(self.depth_head.bias)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Input must have shape [B, 5, H, W].")

        if x.shape[1] != 5:
            raise ValueError("Input must contain exactly 5 channels.")

        if x.shape[-2] < 8 or x.shape[-1] < 8:
            raise ValueError(
                "Input height and width must each be at least 8."
            )

        # Separate low-resolution depth and validity mask.
        low_res_depth = x[:, 3:4]
        validity = x[:, 4:5].clamp(0.0, 1.0)

        # Encoder.
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)

        # Bottleneck.
        x = self.bottleneck(x)

        # Decoder.
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        x = self.shared_refine(x)

        # Upscale features to the requested output resolution.
        output_size = (
            low_res_depth.shape[-2] * self.scale_factor,
            low_res_depth.shape[-1] * self.scale_factor
        )

        high_res_features = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        # Baseline depth reconstruction.
        base_depth = F.interpolate(
            low_res_depth,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )

        # Upsample validity as a soft confidence map.
        high_res_validity = F.interpolate(
            validity,
            size=output_size,
            mode="nearest"
        )

        # Learned residual.
        residual = self.depth_head(high_res_features)

        # Boundary probabilities.
        boundary = torch.sigmoid(
            self.boundary_head(high_res_features)
        )

        # The model learns corrections while retaining the baseline.
        refined_depth = base_depth + residual

        # Preserve output tensor shape and finite values.
        refined_depth = torch.nan_to_num(
            refined_depth,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return {
            "depth": refined_depth,
            "boundary": boundary,
            "residual": residual,
            "validity": high_res_validity
        }


def build_model(scale_factor=2, base_channels=32):
    """Convenience function for creating an EdgeDepth model."""

    return EdgeDepthUNet(
        in_channels=5,
        base_channels=base_channels,
        scale_factor=scale_factor
    )
