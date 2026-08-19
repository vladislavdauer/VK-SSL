from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HubertConfig:
    sample_rate: int = 16000
    conv_layers: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
    )
    conv_bias: bool = False
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    layerdrop: float = 0.05
    dropout_input: float = 0.1
    layer_norm_first: bool = False
    conv_pos: int = 128
    conv_pos_groups: int = 16
    final_dim: int = 256
    logit_temp: float = 0.1
    mask_prob: float = 0.08
    mask_length: int = 10
    min_masks: int = 2
    num_classes: List[int] = field(default_factory=lambda: [100])
    label_rate: float = 50.0
    mask_alpha: float = 1.0
    feature_grad_mult: float = 1.0


def hubert_tiny(num_classes=None, label_rate=50.0, mask_alpha=1.0):
    if num_classes is None:
        num_classes = [16]

    return HubertConfig(
        conv_layers=[
            (32, 10, 5),
            (32, 3, 2),
            (32, 3, 2),
            (32, 3, 2),
            (32, 3, 2),
            (32, 2, 2),
            (32, 2, 2),
        ],
        encoder_layers=2,
        encoder_embed_dim=32,
        encoder_ffn_embed_dim=64,
        encoder_attention_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        dropout_input=0.0,
        conv_pos=16,
        conv_pos_groups=4,
        final_dim=16,
        num_classes=list(num_classes),
        label_rate=label_rate,
        mask_alpha=mask_alpha,
        mask_prob=0.08,
        mask_length=10,
        min_masks=1,
    )


def hubert_base(num_classes=None, label_rate=50.0, mask_alpha=1.0):
    if num_classes is None:
        num_classes = [100]

    return HubertConfig(
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=3072,
        encoder_attention_heads=8,
        layerdrop=0.05,
        final_dim=256,
        num_classes=list(num_classes),
        label_rate=label_rate,
        mask_alpha=mask_alpha,
    )


def hubert_large(num_classes=None, label_rate=50.0, mask_alpha=1.0):
    if num_classes is None:
        num_classes = [500]

    return HubertConfig(
        encoder_layers=24,
        encoder_embed_dim=1024,
        encoder_ffn_embed_dim=4096,
        encoder_attention_heads=16,
        layerdrop=0.0,
        final_dim=768,
        num_classes=list(num_classes),
        label_rate=label_rate,
        mask_alpha=mask_alpha,
    )


def hubert_xlarge(num_classes=None, label_rate=50.0, mask_alpha=1.0):
    if num_classes is None:
        num_classes = [500]

    return HubertConfig(
        encoder_layers=48,
        encoder_embed_dim=1280,
        encoder_ffn_embed_dim=5120,
        encoder_attention_heads=16,
        layerdrop=0.0,
        final_dim=1024,
        num_classes=list(num_classes),
        label_rate=label_rate,
        mask_alpha=mask_alpha,
    )


def get_hubert_config(name, num_classes=None, label_rate=50.0, mask_alpha=1.0):
    factories = {
        "tiny": hubert_tiny,
        "base": hubert_base,
        "large": hubert_large,
        "xlarge": hubert_xlarge,
    }
    if name not in factories:
        raise ValueError(f"Unknown HuBERT config '{name}'. Expected one of {list(factories)}")

    return factories[name](num_classes=num_classes, label_rate=label_rate, mask_alpha=mask_alpha)
