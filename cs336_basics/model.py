import torch
import math
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device, 
        dtype: torch.dtype
    ):
        super().__init__()
        
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(
            w, 
            mean=0, 
            std=math.sqrt(2 / (in_features + out_features)), 
            a=-math.sqrt(2 / (in_features + out_features))*3,
            b=math.sqrt(2 / (in_features + out_features))*3
        )
        self.w = torch.nn.Parameter(w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")
    
class Embedding(torch.nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: None, 
        dtype: None, 
        **kwargs
    ):
        super().__init__()
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(
            w, 
            mean=0, 
            std=1, 
            a=-3, 
            b=3
        )
        self.w = torch.nn.Parameter(w)
        
    def forward(
        self, 
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.w[token_ids]