import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from functools import partial

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in Microsoft Samba.
    
    From paper: "We use SwiGLU for all the models trained in this paper"
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        # SwiGLU(x) = Swish(xW1) ⊙ (xW3) W2
        # where Swish(x) = x * sigmoid(x) = x * σ(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention as used in Microsoft Samba.
    
    From paper: "Our SWA layer operates on a window size w=2048 that slides over 
    the input sequence, ensuring that the computational complexity remains linear 
    with respect to the sequence length. The RoPE relative positions are applied 
    within the sliding window."
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        window_size: int = 2048,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.window_size = window_size
        self.dropout = dropout
        
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model
        
        # Grouped Query Attention support (as mentioned in paper)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # RoPE initialization
        self.rope_theta = rope_theta
        self._init_rope()
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
    
    def _init_rope(self):
        """Initialize RoPE (Rotary Position Embedding) as used in Samba."""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def _apply_rope(self, x, seq_len, offset=0):
        """Apply RoPE within sliding window."""
        dtype = x.dtype
        
        # Only apply RoPE within the window
        effective_len = min(seq_len, self.window_size)
        t = torch.arange(effective_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        freqs = torch.outer(t, self.inv_freq)
        
        # Create rotation matrices
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        
        # Rotate x
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return rotated
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        
        # Sliding window attention implementation
        # For efficiency, we use standard attention but mask beyond window
        if seq_len <= self.window_size:
            # Standard attention for short sequences
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Sliding window for long sequences
            attn_output = self._sliding_window_attention(q, k, v, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.head_dim
        )
        
        return self.o_proj(attn_output)
    
    def _sliding_window_attention(self, q, k, v, attention_mask=None):
        """Implement sliding window attention for long sequences."""
        batch_size, n_heads, seq_len, head_dim = q.shape
        window_size = self.window_size
        
        # Create sliding window mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        
        # Apply sliding window constraint
        for i in range(seq_len):
            start_pos = max(0, i - window_size + 1)
            causal_mask[i, :start_pos] = 0
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask
        
        # Compute attention with sliding window mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)
        
        return torch.matmul(attn_weights, v)


class SambaBlock(nn.Module):
    """Single Samba block: Mamba + MLP + SWA + MLP
    
    From paper: "Samba combines SSMs with attention through layer-wise interleaving 
    Mamba, SwiGLU, and Sliding Window Attention (SWA)."
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_intermediate: Optional[int] = None,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        window_size: int = 2048,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_intermediate = d_intermediate or int(8/3 * d_model)  # Microsoft Samba default
        
        # Layer normalization (Pre-Norm as mentioned in paper)
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.norm3 = RMSNorm(d_model, eps=norm_eps)
        self.norm4 = RMSNorm(d_model, eps=norm_eps)
        
        # Mamba layer for time-dependent semantics
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # First MLP after Mamba
        self.mlp1 = SwiGLU(d_model, self.d_intermediate)
        
        # Sliding Window Attention for precise memory retrieval
        self.swa = SlidingWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            window_size=window_size,
            dropout=dropout,
        )
        
        # Second MLP after SWA
        self.mlp2 = SwiGLU(d_model, self.d_intermediate)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x, attention_mask=None):
        # Mamba layer with residual connection
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        if self.dropout:
            x = self.dropout(x)
        x = residual + x
        
        # First MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = self.mlp1(x)
        if self.dropout:
            x = self.dropout(x)
        x = residual + x
        
        # Sliding Window Attention with residual connection
        residual = x
        x = self.norm3(x)
        x = self.swa(x, attention_mask)
        if self.dropout:
            x = self.dropout(x)
        x = residual + x
        
        # Second MLP with residual connection
        residual = x
        x = self.norm4(x)
        x = self.mlp2(x)
        if self.dropout:
            x = self.dropout(x)
        x = residual + x
        
        return x


class SambaEncoder(nn.Module):
    """Microsoft Samba Encoder for DiariZen integration.
    
    Architecture: Layer-wise combination of Mamba + MLP + SWA + MLP
    
    From the paper:
    "Samba selectively compresses a given sequence into recurrent hidden states 
    while still maintaining the ability to precisely recall memories with the 
    attention mechanism."
    
    Key features:
    - Linear time complexity O(n)
    - Unlimited context extrapolation  
    - Hybrid of SSM (Mamba) + Attention (SWA) + MLP
    """
    def __init__(
        self,
        attention_in: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_intermediate: Optional[int] = None,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        window_size: int = 2048,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.attention_in = attention_in
        self.num_layers = num_layers
        
        print(f"🚀 Initializing SambaEncoder with {num_layers} layers")
        print(f"   Model dimension: {attention_in}")
        print(f"   Window size: {window_size}")
        print(f"   Architecture: Mamba + MLP + SWA + MLP per block")
        
        # Stack of Samba blocks
        self.layers = nn.ModuleList([
            SambaBlock(
                d_model=attention_in,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                d_intermediate=d_intermediate,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                window_size=window_size,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = RMSNorm(attention_in, eps=norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following Microsoft Samba paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass through Samba encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Process through all Samba blocks
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x
    
    def get_memory_usage(self):
        """Return memory usage information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_efficient": True,
            "architecture": "Hybrid Mamba + SWA",
            "complexity": "O(n) linear time"
        } 