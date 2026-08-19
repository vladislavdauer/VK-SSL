import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_bert_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        if module.q_proj_weight is not None:
            module.q_proj_weight.data.normal_(mean=0.0, std=0.02)
        if module.k_proj_weight is not None:
            module.k_proj_weight.data.normal_(mean=0.0, std=0.02)
        if module.v_proj_weight is not None:
            module.v_proj_weight.data.normal_(mean=0.0, std=0.02)
        if module.out_proj.weight is not None:
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        if module.in_proj_bias is not None:
            module.in_proj_bias.data.zero_()
        if module.out_proj.bias is not None:
            module.out_proj.bias.data.zero_()


class SamePad(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            return x[:, :, : -self.remove]
        return x


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        layer_norm_first: bool,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_first = layer_norm_first
        self.self_attn = nn.MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                x, x, x, key_padding_mask=padding_mask, need_weights=False
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
        else:
            x, _ = self.self_attn(
                x, x, x, key_padding_mask=padding_mask, need_weights=False
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            residual = x
            x = F.gelu(self.fc1(x))
            x = F.dropout(x, p=self.activation_dropout, training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.final_layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_layers: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        layerdrop: float,
        layer_norm_first: bool,
        conv_pos: int,
        conv_pos_groups: int,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop

        pos_conv = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        std = math.sqrt(4.0 / (conv_pos * embedding_dim))
        nn.init.normal_(pos_conv.weight, mean=0.0, std=std)
        nn.init.constant_(pos_conv.bias, 0.0)
        pos_conv = nn.utils.parametrizations.weight_norm(pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(pos_conv, SamePad(conv_pos), nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    layer_norm_first=layer_norm_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.apply(init_bert_params)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
        tgt_layer: int = None,
    ):
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        pos = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + pos

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        hidden_states = []
        for i, layer in enumerate(self.layers):
            dropout_probability = torch.rand((), device=x.device).item()
            skip = self.training and self.layerdrop > 0 and dropout_probability < self.layerdrop
            if not skip:
                x = layer(x, padding_mask=padding_mask)

            hidden_states.append(x)
            if tgt_layer is not None and i == tgt_layer:
                break

        if self.layer_norm_first and tgt_layer is None:
            x = self.layer_norm(x)

        return x, hidden_states
