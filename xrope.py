import torch


def duplicate_interleave(m):
    return m.view(-1, 1).repeat(1, 2).view(m.shape[0], -1)


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class XposRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int = 10_000,
        scale_base: int = 512,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scale_base = scale_base
        self.seq_len_cached = None
        self.batch_size_cached = None
        self.cos_cached = None
        self.sin_cached = None
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() /  head_dim))
        self.register_buffer(
            "inv_freq", inv_freq, persistent=False
        )
        self.register_buffer(
            "scale", (torch.arange(0, head_dim, 2) + 0.4 * head_dim) / (1.4 * head_dim)
        )

    def cos_sin(
        self,
        seq_len: int,
        device: str = "cpu",
        dtype=torch.bfloat16,
    ):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.einsum("i, j -> ij", t, self.inv_freq)
            emb = duplicate_interleave(freqs).to(device)

            if dtype in [torch.float16, torch.bfloat16]:
                emb = emb.float()

            self.cos_cached = emb.cos()[None, ...].type(dtype)
            self.sin_cached = emb.sin()[None, ...].type(dtype)

        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        """
        Args:
            x (torch.Tensor): input tensor with shape [batch x num_head, seq_len, embed_dim]
            offset (int): position offset
            downscale (bool):  apply scale
        """
        batch, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)
        scale = self.scale ** torch.arange(seq_len).to(self.scale).div(self.scale_base)[:, None]
        scale = duplicate_interleave(scale)
        return (q * cos * scale) + (rotate_every_two(q) * sin * scale), (k * cos * scale) + (rotate_every_two(k) * sin * scale)
