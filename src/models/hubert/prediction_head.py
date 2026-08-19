from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class HubertPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, final_dim: int, num_classes: List[int], logit_temp: float = 0.1):
        super().__init__()
        self.logit_temp = logit_temp
        self.num_classes = list(num_classes)
        self.projections = nn.ModuleList(
            [nn.Linear(embed_dim, final_dim) for _ in self.num_classes]
        )
        embeddings = []
        for n_class in self.num_classes:
            emb = nn.Parameter(torch.empty(n_class, final_dim))
            nn.init.uniform_(emb)
            embeddings.append(emb)

        self.label_embs = nn.ParameterList(embeddings)

    def logits(self, hidden: torch.Tensor, codebook_index: int = 0) -> torch.Tensor:
        projected = self.projections[codebook_index](hidden)
        codes = self.label_embs[codebook_index]
        sim = F.cosine_similarity(
            projected.unsqueeze(-2).float(),
            codes.unsqueeze(0).float(),
            dim=-1,
        )
        return sim / self.logit_temp

    def forward(self, hidden: torch.Tensor) -> List[torch.Tensor]:
        return [self.logits(hidden, i) for i in range(len(self.num_classes))]
