import torch
import math
from einops import einsum, reduce


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device, dtype: torch.dtype):
        super().__init__()

        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(
            w,
            mean=0,
            std=math.sqrt(2 / (in_features + out_features)),
            a=-math.sqrt(2 / (in_features + out_features)) * 3,
            b=math.sqrt(2 / (in_features + out_features)) * 3,
        )
        self.w = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: None, dtype: None, **kwargs):
        super().__init__()
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3)
        self.w = torch.nn.Parameter(w)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        w = torch.ones((d_model,), device=device, dtype=dtype)
        self.w = torch.nn.Parameter(w)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        result = self.w * x / rms

        return result.to(in_dtype)
