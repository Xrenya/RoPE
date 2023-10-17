# Rotary Position Embedding (RoPE)
#### Rotary matrix multiplication
$$ R^d_{\theta, m}x = 
\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
. \\
. \\
. \\
x_{d-1} \\
x_d
\end{pmatrix} \otimes 
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_1 \\
\cos m\theta_2 \\
\cos m\theta_2 \\
. \\
. \\
. \\
\cos m\theta_{d/2} \\
\cos m\theta_{d/2}
\end{pmatrix}
+ 
\begin{pmatrix}
-x_2 \\
x_1 \\
-x_4 \\
x_3 \\
. \\
. \\
. \\
-x_{d-1} \\
x_d
\end{pmatrix} \otimes 
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_1 \\
\cos m\theta_2 \\
\cos m\theta_2 \\
. \\
. \\
. \\
\cos m\theta_{d/2} \\
\cos m\theta_{d/2}
\end{pmatrix}
$$

```python
class RotaryEmbedding(nn.Module):
    ...
    def forward(self, q, k):
        batch, num_heads, seq_len, head_dim = q.shape
        cos, sin = self.cos_sin(seq_len, q.device, q.dtype)

        return (q * cos) + (rotate_every_two(q) * sin), (k * cos) + (rotate_every_two(k) * sin)
```
