from typing import Tuple
import torch
from torch import nn
from ctransfer.models.blocks import (
    get_conv,
    get_down_block,
    get_mid_block,
)


class ResNet(nn.Module):
    def __init__(
        self,
        input_image_resolution: int = 256,
        in_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (32, 64, 64, 16, 16, 4, 4, 4),
        layers_per_block: int = 2,
        act_fn: str = "SiLU",
        conv_kernel_size: int = 3,
        conv_padding_mode: str = "circular",
        dropout_prob: float = 0.05,
        num_groups_norm: int = 2,
        downsample_padding: int = 1,
        interpolate_down: bool = True,
        summary_dim: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.conv_kernel_size = conv_kernel_size
        self.conv_padding_mode = conv_padding_mode
        self.dropout_prob = dropout_prob
        self.num_groups_norm = num_groups_norm
        self.downsample_padding = downsample_padding
        self.interpolate_down = interpolate_down
        self.summary_dim = summary_dim

        activation_fn = getattr(nn, act_fn)()

        # input convolution:
        padding = (conv_kernel_size - 1) // 2
        self.conv_in = get_conv(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        output_channel = block_out_channels[0]
        current_size = input_image_resolution

        self.down_blocks = nn.ModuleList([])
        for i, _ in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                num_resnet_blocks=layers_per_block,
                input_channels=input_channel,
                output_channels=output_channel,
                conditioning_dim=None,
                dropout_prob=dropout_prob,
                num_groups_norm=num_groups_norm,
                activation_fn=activation_fn,
                kernel_size=conv_kernel_size,
                padding=downsample_padding,
                padding_mode=conv_padding_mode,
                add_downsample=not is_final_block,
                interpolate_down=interpolate_down,
            )
            self.down_blocks.append(down_block)
            if not is_final_block:
                current_size = current_size // 2

        # mid blocks
        self.mid_block = get_mid_block(
            num_resnet_blocks=layers_per_block,
            input_channels=block_out_channels[-1],
            output_channels=output_channel,
            conditioning_dim=None,
            dropout_prob=dropout_prob,
            num_groups_norm=num_groups_norm,
            activation_fn=activation_fn,
            kernel_size=conv_kernel_size,
            padding=1,
            padding_mode=conv_padding_mode,
        )
        self.fc = nn.Linear(
            output_channel * current_size * current_size * current_size, summary_dim
        )  # Adjust the dimensions based on your output shape

    @property
    def hparams(
        self,
    ):
        return {
            "in_channels": self.in_channels,
            "block_out_channels": self.block_out_channels,
            "layers_per_block": self.layers_per_block,
            "act_fn": self.act_fn,
            "num_groups_norm": self.num_groups_norm,
            "interpolate_down": self.interpolate_down,
            "dropout_prob": self.dropout_prob,
            "conv_kernel_size": self.conv_kernel_size,
            "conv_padding_mode": self.conv_padding_mode,
            "summary_dim": self.summary_dim,
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 1. pre-process x
        x = self.conv_in(x)

        # 2. down
        for i, downsample_block in enumerate(self.down_blocks):
            x, res_samples = downsample_block(hidden_states=x, conditioning=None)

        # 3. mid
        if self.mid_block is not None:
            x = self.mid_block(x, conditioning=None)
        x = torch.flatten(x, start_dim=1)
        # Pass through the fully connected layer for classification
        x = self.fc(x)
        return x
