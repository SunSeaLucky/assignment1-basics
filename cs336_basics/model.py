import torch
import math
from einops import einsum, reduce, rearrange


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
        """_summary_

        Args:
            num_embeddings (int): = vocab_size (int): The number of embeddings in the vocabulary
            embedding_dim (int): = d_model (int): The size of the embedding dimension
            device (None): _description_
            dtype (None): _description_
        """
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


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = 8 * d_model / 3
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor):  # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        _x = self.w1(x)
        x_silu = _x * torch.sigmoid(_x)
        return self.w2(self.w3(x) * x_silu)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """_summary_

        Args:
            theta (float): RoPE parameter.
            d_k (int): Embedding dimension size for the query or key tensor.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            device (torch.device | None, optional): _description_. Defaults to None.
        """
        super().__init__()
        position = torch.arange(max_seq_len, device=device)  # [seq_len]
        freq = torch.arange(0, d_k, 2, device=device) / d_k  # [d_k/2]
        freq_inv = 1.0 / (theta**freq)  # [d_k/2]
        angles = einsum(position, freq_inv, "seq_len, d_k_half -> seq_len d_k_half")
        self.register_buffer("cos", torch.cos(angles), persistent=False)  # [seq_len, d_k/2]
        self.register_buffer("sin", torch.sin(angles), persistent=False)  # [seq_len, d_k/2]

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:  # ([..., seq_len, d_k], [..., seq_len]) -> [..., seq_len, d_k]
        pos_sin = self.sin[token_positions]  # [..., seq_len, d_k/2]
        pos_cos = self.cos[token_positions]  # [..., seq_len, d_k/2]

        x_even = x[..., 0::2]  # [..., seq_len, d_k/2]
        x_old = x[..., 1::2]  # [..., seq_len, d_k/2]

        x_even_rot = x_even * pos_cos - x_old * pos_sin  # [..., seq_len, d_k/2]
        x_old_rot = x_even * pos_sin + x_old * pos_cos  # [..., seq_len, d_k/2]

        x_rot = rearrange([x_even_rot, x_old_rot], "two ... -> ... two")
        x_rot = rearrange(x_rot, "... d1 d2 -> ... (d1 d2)")

        return x_rot


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)
