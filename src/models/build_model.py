import torch

from src.models.rnnt import _ConformerEncoder, RNNT, _Predictor, _Joiner
from src.models.conformer_v2 import ConformerEncoder as ConformerEncoderV2


class _ConformerV2EncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        time_reduction_stride: int,
        conformer_input_dim: int,
        conformer_ffn_dim: int,
        conformer_num_layers: int,
        conformer_num_heads: int,
        conformer_depthwise_conv_kernel_size: int,
        conformer_dropout: float,
    ):
        super().__init__()

        assert conformer_ffn_dim % conformer_input_dim == 0

        self.output_dim = output_dim
        self.time_reduction = _TimeReduction(time_reduction_stride)

        self.input_linear = torch.nn.Linear(
            input_dim * time_reduction_stride,
            conformer_input_dim,
        )

        self.encoder = ConformerEncoderV2(
            feat_in=conformer_input_dim,
            n_layers=conformer_num_layers,
            d_model=conformer_input_dim,
            subsampling="conv1d",
            subs_kernel_size=3,
            subsampling_factor=2,
            ff_expansion_factor=conformer_ffn_dim // conformer_input_dim,
            self_attention_model="rotary",
            n_heads=conformer_num_heads,
            pos_emb_max_len=5000,
            conv_norm_type="batch_norm",
            conv_kernel_size=conformer_depthwise_conv_kernel_size,
            flash_attn=False,
            activation_checkpointing=False,
        )

        for layer in self.encoder.layers:
            if hasattr(layer.self_attn, "torch_sdpa_attn"):
                layer.self_attn.torch_sdpa_attn = False
            if hasattr(layer.self_attn, "flash_attn"):
                layer.self_attn.flash_attn = False

        self.output_linear = torch.nn.Linear(
            conformer_input_dim,
            output_dim,
        )

        self.layer_norm = torch.nn.LayerNorm(output_dim)

        _ = conformer_dropout

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        x, lengths = self.time_reduction(input, lengths)
        x = self.input_linear(x)

        lengths = lengths.to(device=x.device, dtype=torch.long)
        lengths = lengths.clamp_min(0).clamp_max(x.size(1))

        pad_mask = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if not hasattr(self.encoder.pos_enc, "pe"):
            self.encoder.pos_enc.extend_pe(
                self.encoder.pos_emb_max_len,
                x.device,
            )

        max_len = x.size(1)
        x, pos_emb = self.encoder.pos_enc(x=x)

        valid_mask = (
            torch.arange(max_len, device=x.device).expand(lengths.size(0), -1)
            < lengths.unsqueeze(-1)
        )

        att_mask = None
        if x.size(0) > 1:
            att_mask = valid_mask.unsqueeze(1).repeat(1, max_len, 1)
            att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
            att_mask = ~att_mask

        pad_mask = ~valid_mask

        for layer in self.encoder.layers:
            x = layer(
                x=x,
                pos_emb=pos_emb,
                att_mask=att_mask,
                pad_mask=pad_mask,
            )

        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x = self.output_linear(x)
        x = self.layer_norm(x)

        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return x, lengths

    def infer(self, input: torch.Tensor, lengths: torch.Tensor, states=None):
        raise RuntimeError("ConformerV2 CTC wrapper does not support streaming inference.")

def conformer_v2_ctc_model(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
) -> _ConformerV2EncoderWrapper:
    return _ConformerV2EncoderWrapper(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )


def conformer_v2_ctc_base() -> _ConformerV2EncoderWrapper:
    return conformer_v2_ctc_model(
        input_dim=80,
        encoding_dim=256,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
    )

def conformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
) -> RNNT:
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols, activation=joiner_activation)
    return RNNT(encoder, predictor, joiner)


def conformer_rnnt_base() -> RNNT:
    r"""Builds basic version of Conformer RNN-T model.

    Returns:
        RNNT:
            Conformer RNN-T model.
    """
    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=256,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=1,
        lstm_hidden_dim=640,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )
    # return conformer_rnnt_model(
    #     input_dim=80,
    #     encoding_dim=1024,
    #     time_reduction_stride=4,
    #     conformer_input_dim=256,
    #     conformer_ffn_dim=1024,
    #     conformer_num_layers=16,
    #     conformer_num_heads=4,
    #     conformer_depthwise_conv_kernel_size=31,
    #     conformer_dropout=0.1,
    #     num_symbols=1024,
    #     symbol_embedding_dim=256,
    #     num_lstm_layers=2,
    #     lstm_hidden_dim=512,
    #     lstm_layer_norm=True,
    #     lstm_layer_norm_epsilon=1e-5,
    #     lstm_dropout=0.3,
    #     joiner_activation="tanh",
    # )