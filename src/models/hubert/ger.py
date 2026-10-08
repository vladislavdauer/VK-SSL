import torch

def effective_rank_from_gram(gram: torch.Tensor, eps: float = 1e-12) -> float:
    eigvals = torch.linalg.eigvalsh(gram.detach().double().cpu()).clamp_min(0.0)
    singular = torch.sqrt(eigvals)
    total = singular.sum()
    if float(total) <= eps:
        return 0.0

    p = singular / total
    p = p[p > eps]
    entropy = -(p * torch.log(p)).sum()
    return float(torch.exp(entropy).item())

def effective_rank(matrix: torch.Tensor, eps: float = 1e-12) -> float:

    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got {tuple(matrix.shape)}")

    if matrix.numel() == 0:
        return 0.0

    x = matrix.double()
    if x.size(0) < x.size(1):
        gram = x @ x.T
    else:
        gram = x.T @ x

    return effective_rank_from_gram(gram, eps=eps)
