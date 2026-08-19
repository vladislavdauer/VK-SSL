from typing import List, Tuple

import torch
import torch.nn as nn


def conv_output_length(length, kernel_size, stride):
    return torch.div(length - kernel_size, stride, rounding_mode="floor") + 1


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.conv_layers_cfg = list(conv_layers)
        self.downsampling_factor = 1
        for _, _, stride in conv_layers:
            self.downsampling_factor *= stride

        in_d = 1
        blocks = []
        for i, (dim, kernel, stride) in enumerate(conv_layers):
            conv = nn.Conv1d(in_d, dim, kernel, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            layers = [conv, nn.Dropout(p=dropout)]
            if i == 0:
                layers.append(nn.GroupNorm(dim, dim, affine=True))

            layers.append(nn.GELU())
            blocks.append(nn.Sequential(*layers))
            in_d = dim

        self.conv_layers = nn.ModuleList(blocks)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(dtype=torch.long)
        for _, kernel, stride in self.conv_layers_cfg:
            lengths = conv_output_length(lengths, kernel, stride)
            lengths = lengths.clamp_min(0)

        return lengths

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        x = source.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)

        return x
