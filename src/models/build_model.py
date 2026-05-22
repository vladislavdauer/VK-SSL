from src.models.rnnt import _ConformerEncoder, RNNT, _Predictor, _Joiner
from src.models.conformer_v2 import ConformerEncoder as ConformerEncoderV2

# TODO: Add ConformerEncoderV2

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


# TODO: make custom pass args
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
        conformer_depthwise_conv_kernel_size=32,
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