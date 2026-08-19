import torch


def compute_span_mask(
    lengths: torch.Tensor,
    mask_prob: float,
    mask_length: int,
    min_masks: int = 2,
):
    batch_size = lengths.size(0)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    device = lengths.device
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    if max_len == 0 or mask_length < 1:
        return mask

    for i in range(batch_size):
        sz = int(lengths[i].item())
        if sz <= 1:
            continue

        span = min(mask_length, sz)
        max_start = max(sz - span + 1, 1)
        num_starts = int(mask_prob * sz + torch.rand((), device=device).item())
        num_starts = max(min_masks, num_starts)
        num_starts = min(num_starts, max_start)
        starts = torch.randperm(max_start, device=device)[:num_starts]

        for start in starts.tolist():
            end = min(start + span, sz)
            mask[i, start:end] = True

        if not mask[i, :sz].any():
            start = int(torch.randint(0, max_start, (1,), device=device).item())
            mask[i, start : min(start + span, sz)] = True

        if sz > 1 and bool(mask[i, :sz].all()):
            unmask_at = int(torch.randint(0, sz, (1,), device=device).item())
            mask[i, unmask_at] = False

    return mask


def apply_mask(features: torch.Tensor, mask: torch.Tensor, mask_emb: torch.Tensor):
    x = features.clone()
    x[mask] = mask_emb.to(dtype=x.dtype, device=x.device)
    return x
