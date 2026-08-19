from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.models.hubert.cnn_encoder import ConvFeatureExtractionModel
from src.models.hubert.config import HubertConfig
from src.models.hubert.masking import apply_mask, compute_span_mask
from src.models.hubert.prediction_head import HubertPredictionHead
from src.models.hubert.transformer import TransformerEncoder


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class HubertModel(nn.Module):
    def __init__(self, cfg: HubertConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=cfg.conv_layers,
            dropout=0.0,
            conv_bias=cfg.conv_bias,
        )
        conv_dim = cfg.conv_layers[-1][0]
        self.post_extract_proj = (
            nn.Linear(conv_dim, cfg.encoder_embed_dim)
            if conv_dim != cfg.encoder_embed_dim
            else None
        )
        self.layer_norm = nn.LayerNorm(conv_dim)
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.mask_emb = nn.Parameter(torch.empty(cfg.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(
            embedding_dim=cfg.encoder_embed_dim,
            ffn_embedding_dim=cfg.encoder_ffn_embed_dim,
            num_layers=cfg.encoder_layers,
            num_attention_heads=cfg.encoder_attention_heads,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.activation_dropout,
            layerdrop=cfg.layerdrop,
            layer_norm_first=cfg.layer_norm_first,
            conv_pos=cfg.conv_pos,
            conv_pos_groups=cfg.conv_pos_groups,
        )
        self.pred_head = HubertPredictionHead(
            embed_dim=cfg.encoder_embed_dim,
            final_dim=cfg.final_dim,
            num_classes=cfg.num_classes,
            logit_temp=cfg.logit_temp,
        )
        downsample = 1
        for _, _, stride in cfg.conv_layers:
            downsample *= stride

        self.feat2tar_ratio = cfg.label_rate * downsample / float(cfg.sample_rate)
        self.mask_alpha = cfg.mask_alpha
        self.feature_grad_mult = cfg.feature_grad_mult

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
            return features
        with torch.no_grad():
            return self.feature_extractor(source)

    def align_targets(self, features: torch.Tensor, target_list: List[torch.Tensor]):
        feat_tsz = features.size(2)
        targ_tsz = min(t.size(1) for t in target_list)
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]

        target_inds = (torch.arange(feat_tsz, device=features.device).float() * self.feat2tar_ratio).long()
        target_list = [t.index_select(1, target_inds) for t in target_list]
        return features, target_list

    def padding_mask_from_lengths(self, lengths: torch.Tensor, time: int) -> torch.Tensor:
        return torch.arange(time, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    def extract_features(
        self,
        source: torch.Tensor,
        lengths: torch.Tensor,
        tgt_layer: Optional[int] = None,
    ):
        features = self.forward_features(source)
        feat_lengths = self.feature_extractor.output_lengths(lengths)
        feat_lengths = feat_lengths.clamp(min=0, max=features.size(2))
        x = features.transpose(1, 2)
        x = self.layer_norm(x)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)

        padding_mask = self.padding_mask_from_lengths(feat_lengths, x.size(1))
        x, hidden_states = self.encoder(x, padding_mask=padding_mask, tgt_layer=tgt_layer)
        if tgt_layer is not None:
            x = hidden_states[tgt_layer]

        return x, feat_lengths, padding_mask

    def forward(
        self,
        source: torch.Tensor,
        lengths: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        mask: bool = True,
        features_only: bool = False,
        tgt_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.forward_features(source)
        feat_lengths = self.feature_extractor.output_lengths(lengths)
        feat_lengths = feat_lengths.clamp(min=0, max=features.size(2))

        if target_list is not None:
            features, target_list = self.align_targets(features, target_list)
            feat_lengths = torch.minimum(
                feat_lengths,
                torch.full_like(feat_lengths, features.size(2)),
            )

        x = features.transpose(1, 2)
        x = self.layer_norm(x)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)

        x = self.dropout_input(x)

        feat_lengths = feat_lengths.clamp(min=0, max=x.size(1))
        padding_mask = self.padding_mask_from_lengths(feat_lengths, x.size(1))

        if mask:
            span_mask = compute_span_mask(
                feat_lengths,
                mask_prob=self.cfg.mask_prob,
                mask_length=self.cfg.mask_length,
                min_masks=self.cfg.min_masks,
            )
            span_mask = span_mask & ~padding_mask
            x = apply_mask(x, span_mask, self.mask_emb)
        else:
            span_mask = torch.zeros_like(padding_mask)

        encoded, hidden_states = self.encoder(
            x, padding_mask=padding_mask, tgt_layer=tgt_layer
        )
        if tgt_layer is not None:
            encoded = hidden_states[tgt_layer]

        if features_only:
            return {
                "x": encoded,
                "padding_mask": padding_mask,
                "feat_lengths": feat_lengths,
                "hidden_states": hidden_states,
            }

        valid = ~padding_mask
        masked = span_mask & valid
        unmasked = (~span_mask) & valid

        logit_m_list = []
        logit_u_list = []
        target_m_list = []
        target_u_list = []
        for k, targets in enumerate(target_list):
            logits = self.pred_head.logits(encoded, k)
            logit_m_list.append(logits[masked])
            logit_u_list.append(logits[unmasked])
            target_m_list.append(targets[masked])
            target_u_list.append(targets[unmasked])

        return {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "target_m_list": target_m_list,
            "target_u_list": target_u_list,
            "padding_mask": padding_mask,
            "mask_indices": span_mask,
            "feat_lengths": feat_lengths,
            "x": encoded,
        }

    def masked_prediction_loss(self, net_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        alpha = self.mask_alpha
        losses = []
        for logit_m, target_m, logit_u, target_u in zip(
            net_output["logit_m_list"],
            net_output["target_m_list"],
            net_output["logit_u_list"],
            net_output["target_u_list"],
        ):
            loss = logit_m.new_zeros(())
            if alpha > 0 and logit_m.numel() > 0:
                loss = loss + alpha * nn.functional.cross_entropy(logit_m.float(), target_m.long())
            if alpha < 1 and logit_u.numel() > 0:
                loss = loss + (1.0 - alpha) * nn.functional.cross_entropy(logit_u.float(), target_u.long())

            losses.append(loss)

        stacked = torch.stack(losses) if losses else torch.zeros((), device=self.mask_emb.device)
        return stacked.mean()

    def remove_pretraining_modules(self):
        self.pred_head = None

    def freeze_feature_extractor(self):
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.feature_grad_mult = 0.0
