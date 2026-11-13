import torch
import torch.nn as nn
from einops import rearrange
import math


"""
Patchify:
    - Input: (batch_size, channels, height, width)
    - Output: (batch_size, num_patches, patch_dim)
    - patch_dim = patch_size * patch_size * channels
    - num_patches = (H/patch_size) * (W/patch_size)
"""
class Patchify(nn.Module):
    def __init__(self, patch_size=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, C, H, W)
        # Descomponemos H = nh*p, W = nw*p y reorganizamos
        p = self.patch_size
        out = rearrange(x, 'b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)', p1=p, p2=p)
        return out


"""
Unpatchify:
    - Input: (batch_size, num_patches, patch_dim)
    - Output: (batch_size, channels, height, width)
    - Infiere C = patch_dim / (p*p), nh = nw = sqrt(num_patches)
"""
class Unpatchify(nn.Module):
    def __init__(self, patch_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def forward(self, x):
        # x: (B, num_patches, patch_dim)
        p = self.patch_size
        B, num_patches, patch_dim = x.shape
        nh = nw = int(num_patches ** 0.5)
        C = patch_dim // (p * p)
        out = rearrange(x, 'b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)',
                        nh=nh, nw=nw, p1=p, p2=p, c=C)
        return out


'''
FeedForward:
    - hidden_dim -> inner_dim -> hidden_dim
    - ReLU en el medio
'''
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


'''
SelfAttention:
    - Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    - Input: (B, N, hidden_dim) → Output: (B, N, inner_dim)
    - 3 proyecciones lineales: W_q, W_k, W_v (hidden_dim -> inner_dim)
'''
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, inner_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inner_dim = inner_dim
        self.W_q = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, inner_dim, bias=False)

    def forward(self, x):
        Q = self.W_q(x)  # (B, N, inner_dim)
        K = self.W_k(x)
        V = self.W_v(x)
        d_k = self.inner_dim
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # (B, N, N)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)  # (B, N, inner_dim)
        return out


'''
MultiHeadSelfAttn:
    - num_heads cabezas, cada una con inner_dim = hidden_dim // num_heads
    - Concatena las salidas y proyecta de vuelta a hidden_dim
'''
class MultiHeadSelfAttn(nn.Module):
    def __init__(self, hidden_dim, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hidden_dim % num_heads == 0
        self.inner_dim = hidden_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttention(hidden_dim, self.inner_dim) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # Cada cabeza: (B, N, inner_dim), concatenamos -> (B, N, hidden_dim)
        concat = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(concat)


"""
DiTBlock con adaLN-Zero:
    - MLP(cond) -> 6 params (γ1, β1, α1, γ2, β2, α2), todos zero-init
    - Scale & Shift: (γ+1)*LN(x) + β   (LN sin affine)
    - Scale: α * attn(x)
    - Dos residual connections (attn y ff)
    - Zero-init garantiza que al inicio el bloque sea la identidad
"""
class DiTBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, ff_dim, time_emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # MLP: time_emb_dim -> time_emb_dim//4 -> hidden_dim*6 (zero-init ambos linear)
        mid_dim = time_emb_dim // 4
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, mid_dim),
            nn.SiLU(),
            nn.Linear(mid_dim, hidden_dim * 6),
        )
        nn.init.zeros_(self.adaLN_mlp[1].weight)
        nn.init.zeros_(self.adaLN_mlp[1].bias)
        nn.init.zeros_(self.adaLN_mlp[3].weight)
        nn.init.zeros_(self.adaLN_mlp[3].bias)

        # LayerNorm sin parámetros propios (la modulación la hacen γ y β)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.attn = MultiHeadSelfAttn(hidden_dim, num_heads)
        self.ff = FeedForward(hidden_dim, ff_dim)

    def forward(self, x, cond):
        # Step 1: obtener los 6 parámetros de modulación
        params = self.adaLN_mlp(cond)           # (B, hidden_dim*6)
        params = params.unsqueeze(1)             # (B, 1, hidden_dim*6) para broadcast
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)

        # ---- Rama atención ----
        h = self.norm1(x)
        h = (gamma1 + 1) * h + beta1            # Scale & Shift
        h = self.attn(h)
        h = alpha1 * h                           # Scale
        x = x + h                               # Residual 1

        # ---- Rama feed-forward ----
        h = self.norm2(x)
        h = (gamma2 + 1) * h + beta2            # Scale & Shift
        h = self.ff(h)
        h = alpha2 * h                           # Scale
        x = x + h                               # Residual 2

        return x


## (FOR FLOW MATCHING EC ONLY) Given for free!
class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, emb_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(t.size(0))
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT(nn.Module):

    ## Given for free!
    def get_position_embedding(self, num_patches, patch_size, hidden_dim):
        grid_size = int(patch_size ** 0.5)
        dim = hidden_dim // 2
        
        pos = torch.arange(num_patches, dtype=torch.float32)
        r = pos // grid_size
        c = pos % grid_size
        
        omega = torch.arange(0, dim, 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / dim))
        
        out_r = r[:, None] @ omega[None, :]
        out_c = c[:, None] @ omega[None, :]
        
        row_emb = torch.zeros((1, num_patches, dim))
        col_emb = torch.zeros((1, num_patches, dim))
        
        row_emb[:, :, 0::2] = torch.sin(out_r)
        row_emb[:, :, 1::2] = torch.cos(out_r)
        col_emb[:, :, 0::2] = torch.sin(out_c)
        col_emb[:, :, 1::2] = torch.cos(out_c)
        
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        pos_emb = nn.Parameter(pos_emb, requires_grad=False)
        return pos_emb

    def __init__(self, patch_size, num_blocks, num_heads, ff_dim, time_emb_dim, num_timesteps,
                 hidden_dim, num_patches, num_channels=3, training_type="ddpm", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        patch_dim = patch_size * patch_size * num_channels

        # Step 1: Patchify
        self.patchify = Patchify(patch_size)

        # Step 2: Proyección patch_dim -> hidden_dim (NO zero-init)
        self.patch_proj = nn.Linear(patch_dim, hidden_dim)

        # Step 3: Positional embedding (no trainable)
        self.pos_emb = self.get_position_embedding(num_patches, num_patches, hidden_dim)

        # Step 4: Time embedding: índice entero -> time_emb_dim
        self.time_emb = nn.Embedding(num_timesteps, time_emb_dim)

        # Step 5: Bloques DiT
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, ff_dim, time_emb_dim)
            for _ in range(num_blocks)
        ])

        # Step 6: LayerNorm final
        self.norm = nn.LayerNorm(hidden_dim)

        # Step 7: Proyección hidden_dim -> patch_dim (zero-init)
        self.out_proj = nn.Linear(hidden_dim, patch_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Step 8: Unpatchify
        self.unpatchify = Unpatchify(patch_size)

    def forward(self, image, timestep):
        # Step 1: (B, C, H, W) -> (B, N, patch_dim)
        x = self.patchify(image)

        # Step 2: (B, N, patch_dim) -> (B, N, hidden_dim)
        x = self.patch_proj(x)

        # Step 3: Sumar positional encoding
        x = x + self.pos_emb.to(x.device)

        # Step 4: Time embedding -> (B, time_emb_dim)
        t_emb = self.time_emb(timestep)

        # Step 5: Pasar por los bloques DiT
        for block in self.blocks:
            x = block(x, t_emb)

        # Step 6: LayerNorm
        x = self.norm(x)

        # Step 7: (B, N, hidden_dim) -> (B, N, patch_dim)
        x = self.out_proj(x)

        # Step 8: (B, N, patch_dim) -> (B, C, H, W)
        x = self.unpatchify(x)

        return x