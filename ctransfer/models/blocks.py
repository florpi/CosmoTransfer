from typing import Callable, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module


def get_conv(
    input_channels: int,
    output_channels: int,
    dim: int = 3,
    kernel_size: int = 3,
    padding: int = 1,
    padding_mode: str = "circular",
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    weight_init: Callable = None,
    transposed: bool = False,
) -> nn.Module:
    """
    Create a convolutional layer (2D or 3D) with specified parameters.

    Parameters:
    - input_channels: Number of input channels.
    - output_channels: Number of output channels.
    - dim: Dimension of the convolution (2 or 3).
    - kernel_size: Size of the convolutional kernel.
    - padding: Padding size.
    - padding_mode: Type of padding.
    - stride: Stride size.
    - dilation: Dilation rate.
    - groups: Number of blocked connections.
    - weight_init: Weight initialization function.
    - transposed: Whether to use a transposed convolution.

    Returns:
    - A convolutional layer with specified parameters.
    """
    if weight_init is None:
        weight_init = lambda x: x

    if dim == 2:
        conv = nn.ConvTranspose2d if transposed else nn.Conv2d
    elif dim == 3:
        conv = nn.ConvTranspose3d if transposed else nn.Conv3d
    else:
        raise ValueError("Invalid dimension, only 2D and 3D are supported.")

    out = conv(
        input_channels,
        output_channels,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode=padding_mode,
        stride=stride,
        dilation=dilation,
        # groups=groups,
    )
    if weight_init is not None:
        out = weight_init(out)
    return out


def get_down_block(
    num_resnet_blocks: int,
    input_channels: int,
    output_channels: int,
    activation_fn,
    kernel_size,
    padding,
    padding_mode,
    conditioning_dim: int,
    dropout_prob: float,
    num_groups_norm: int,
    interpolate_down: bool = True,
    add_downsample: bool = True,
):
    return ResNetDownsampleBlock(
        num_layers=num_resnet_blocks,
        input_channels=input_channels,
        output_channels=output_channels,
        activation_fn=activation_fn,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode=padding_mode,
        conditioning_dim=conditioning_dim,
        dropout_prob=dropout_prob,
        num_groups_norm=num_groups_norm,
        add_downsample=add_downsample,
        use_conv=not interpolate_down,
    )


def get_up_block(
    num_resnet_blocks: int,
    input_channels: int,
    output_channels: int,
    prev_output_channels: int,
    activation_fn,
    kernel_size,
    padding,
    padding_mode,
    conditioning_dim: int,
    dropout_prob: float,
    num_groups_norm: int,
    add_upsample: bool,
    interpolate_up: bool = True,
    transpose_conv: bool = False,
):
    return ResNetUpsampleBlock(
        num_layers=num_resnet_blocks,
        input_channels=input_channels,
        output_channels=output_channels,
        prev_output_channels=prev_output_channels,
        conditioning_dim=conditioning_dim,
        dropout_prob=dropout_prob,
        num_groups_norm=num_groups_norm,
        activation_fn=activation_fn,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode=padding_mode,
        add_upsample=add_upsample,
        use_conv=(not transpose_conv) & (not interpolate_up),
        use_conv_transpose=(transpose_conv) & (not interpolate_up),
    )


def get_mid_block(
    num_resnet_blocks: int,
    input_channels: int,
    output_channels: int,
    conditioning_dim: int,
    dropout_prob: float,
    num_groups_norm: int,
    activation_fn,
    kernel_size,
    padding,
    padding_mode,
):
    return MidBlock(
        num_resnet_blocks=num_resnet_blocks,
        input_channels=input_channels,
        output_channels=output_channels,
        conditioning_dim=conditioning_dim,
        dropout_prob=dropout_prob,
        num_groups_norm=num_groups_norm,
        activation_fn=activation_fn,
        kernel_size=kernel_size,
        padding=padding,
        padding_mode=padding_mode,
    )


class MidBlock(nn.Module):
    def __init__(
        self,
        num_resnet_blocks,
        input_channels,
        output_channels,
        conditioning_dim,
        dropout_prob,
        num_groups_norm,
        activation_fn,
        kernel_size,
        padding,
        padding_mode,
    ):
        super(MidBlock, self).__init__()
        self.layers = nn.Sequential(
            *[
                ResNetBlock(
                    input_channels=input_channels if i == 0 else output_channels,
                    output_channels=output_channels,
                    conditioning_dim=conditioning_dim,
                    dropout_prob=dropout_prob,
                    num_groups_norm=num_groups_norm,
                    activation_fn=activation_fn,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                )
                for i in range(num_resnet_blocks)
            ]
        )

    def forward(self, x, conditioning):
        for layer in self.layers:
            x = layer(x, conditioning)
        return x


class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        padding_mode: str = "circular",
        name: str = "conv",
        kernel_size=3,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv3d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=stride,
                stride=stride,
            )
        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Upsample3D(nn.Module):
    """A 3D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        padding_mode: str = "circular",
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose3d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv3d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
                padding_mode=padding_mode,
            )
        self.conv = conv

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: Optional[int] = None,
    ) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class ResNetBlock(nn.Module):
    """
    Residual Block with optional conditioning for 2D or 3D convolutions.

    Parameters:
    - input_channels: Number of input channels.
    - output_channels: Number of output channels.
    - activation_fn: Activation function to use.
    - kernel_size: Size of the convolutional kernel.
    - padding: Padding size.
    - padding_mode: Type of padding for convolutions.
    - dim: Dimension of the convolution (default is 2).
    - conditioning_dim: Dimension of the conditioning vector (default is None).
    - dropout_prob: Dropout probability (default is 0.0).
    - num_groups: Number of groups for GroupNorm (default is 8).
    - norm_eps: Epsilon for GroupNorm (default is 1e-5).
    - norm_affine: Affine parameter for GroupNorm (default is True).
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        activation_fn: Callable,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        conditioning_dim: Optional[int] = None,
        dropout_prob: float = 0.0,
        num_groups_norm: int = 8,
        norm_eps: float = 1e-6,
        norm_affine: bool = True,
        down: bool = False,
        up: bool = False,
        use_conv_down: bool = False,
        use_conv_up: bool = False,
        use_conv_transpose_up: bool = False,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conditioning_dim = conditioning_dim
        self.padding_mode = padding_mode
        self.up = up
        self.down = down

        self.norm1 = nn.GroupNorm(
            num_channels=input_channels,
            num_groups=num_groups_norm,
            eps=norm_eps,
            affine=norm_affine,
        )
        self.conv1 = get_conv(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )  # groups=num_groups_cnn,),

        if conditioning_dim is not None:
            self.conditioning_embedding = nn.Sequential(
                nn.Linear(conditioning_dim, output_channels),
                activation_fn,
                nn.Linear(output_channels, output_channels),
            )

        self.norm2 = nn.GroupNorm(
            num_channels=output_channels,
            num_groups=num_groups_norm,
            eps=norm_eps,
            affine=norm_affine,
        )
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv2 = get_conv(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )  # groups=num_groups_cnn,)
        self.nonlinearity = activation_fn
        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample3D(
                input_channels,
                use_conv=use_conv_up,
                use_conv_transpose=use_conv_transpose_up,
            )
        elif self.down:
            self.downsample = Downsample3D(
                input_channels, use_conv=use_conv_down, padding=1, name="op"
            )
        self.use_in_shortcut = input_channels != output_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode=padding_mode,
                bias=True,
            )

    def forward(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the ResNet blocks and downsampling layer.


        Parameters:
        - x: Input tensor.
        - conditioning: Optional conditioning tensor.

        Returns:
        - Downsampled tensor and tensor before downsampling (skip connection).
        """
        hidden_states = x
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        if self.upsample is not None:
            x = self.upsample(x)
            hidden_states = self.upsample(hidden_states)
        if self.downsample is not None:
            x = self.downsample(x)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        if conditioning is not None:
            conditioning = self.conditioning_embedding(conditioning)
            conditioning = conditioning.view(
                conditioning.shape + (1,) * (hidden_states.dim() - conditioning.dim())
            )
            hidden_states = hidden_states + conditioning

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + hidden_states


class ResNetDownsampleBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        activation_fn: Callable,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        conditioning_dim: int,
        dropout_prob: float = 0.0,
        num_layers: int = 1,
        norm_eps: float = 1e-6,
        num_groups_norm: int = 32,
        add_downsample: bool = True,
        use_conv: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = input_channels if i == 0 else output_channels
            resnets.append(
                ResNetBlock(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    activation_fn=activation_fn,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    conditioning_dim=conditioning_dim,
                    dropout_prob=dropout_prob,
                    num_groups_norm=num_groups_norm,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    ResNetBlock(
                        input_channels=output_channels,
                        output_channels=output_channels,
                        activation_fn=activation_fn,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                        conditioning_dim=conditioning_dim,
                        dropout_prob=dropout_prob,
                        num_groups_norm=num_groups_norm,
                        norm_eps=norm_eps,
                        down=True,
                        use_conv_down=use_conv,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, conditioning)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for i, downsampler in enumerate(self.downsamplers):
                hidden_states = downsampler(hidden_states, conditioning)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class ResNetUpsampleBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        prev_output_channels: int,
        activation_fn: Callable,
        kernel_size: int,
        padding: int,
        padding_mode: str,
        conditioning_dim: int,
        dropout_prob: float = 0.0,
        num_layers: int = 1,
        norm_eps: float = 1e-6,
        num_groups_norm: int = 32,
        add_upsample: bool = True,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = (
                input_channels if (i == num_layers - 1) else output_channels
            )
            resnet_input_channels = prev_output_channels if i == 0 else output_channels

            resnets.append(
                ResNetBlock(
                    input_channels=resnet_input_channels + res_skip_channels,
                    output_channels=output_channels,
                    activation_fn=activation_fn,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    conditioning_dim=conditioning_dim,
                    dropout_prob=dropout_prob,
                    num_groups_norm=num_groups_norm,
                    norm_eps=norm_eps,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResNetBlock(
                        input_channels=output_channels,
                        output_channels=output_channels,
                        activation_fn=activation_fn,
                        kernel_size=kernel_size,
                        padding=padding,
                        padding_mode=padding_mode,
                        conditioning_dim=conditioning_dim,
                        dropout_prob=dropout_prob,
                        num_groups_norm=num_groups_norm,
                        norm_eps=norm_eps,
                        up=True,
                        use_conv_up=use_conv,
                        use_conv_transpose_up=use_conv_transpose,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, conditioning)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, conditioning)

        return hidden_states
