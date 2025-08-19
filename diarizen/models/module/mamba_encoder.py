"""
Mamba2-based Encoder for Speaker Diarization

This module implements a state-space model encoder using Mamba2 architecture as a drop-in
replacement for ConformerEncoder in speaker diarization tasks. The implementation provides
significant computational efficiency improvements while maintaining or improving accuracy.

Key Features:
- Linear O(n) complexity
- Bidirectional processing for complete temporal context
- Hardware-optimized Mamba2 implementation
- Full compatibility with ConformerEncoder interface
- Dynamic parameter detection for cross-version compatibility

Architecture:
The encoder consists of stacked Mamba2 blocks with RMSNorm, supporting both unidirectional
and bidirectional processing. For bidirectional mode, separate forward and backward block
stacks process the sequence in both temporal directions, with flexible merging strategies.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import inspect

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm


class MambaEncoder(nn.Module):
    """
    Mamba2-based encoder as a drop-in replacement for ConformerEncoder.
    
    This implementation leverages the Mamba2 state-space model architecture for efficient
    sequence modeling in speaker diarization tasks. Key features include:
    
    1. **Linear Complexity**: O(n) complexity vs O(n²) for Transformers, enabling
       efficient processing of long audio sequences.
    
    2. **Bidirectional Processing**: Processes sequences in both forward and backward
       directions to capture complete temporal context, crucial for accurate speaker
       boundary detection.
    
    3. **ConformerEncoder Compatibility**: Drop-in replacement with identical interface:
       - Input: (batch, time, features)
       - Output: (batch, time, features) or (batch, time, features*2) if bidirectional
    
    4. **Mamba2 Enhancements**: Uses latest Mamba2 implementation with improved
       hardware optimization and performance compared to original Mamba.
    
    Architecture Details:
    - Each layer consists of a Mamba2 block with RMSNorm normalization
    - Bidirectional processing uses separate forward/backward block stacks
    - Flexible merging strategies: concatenation, addition, or multiplication
    - Dynamic parameter detection for cross-version compatibility
    """

    def __init__(
        self,
        attention_in: int = 256,        # Input/output feature dimension
        num_layer: int = 4,             # Number of Mamba2 layers in each direction
        d_state: int = 128,             # State space dimension (memory size)
        d_conv: int = 4,                # Convolution kernel size for temporal modeling
        expand: int = 2,                # Hidden dimension expansion factor
        headdim: int = 64,              # Head dimension for Mamba2 (new in v2)
        ngroups: int = 1,               # Number of groups for Mamba2 (new in v2)
        rmsnorm_eps: float = 1e-5,      # RMSNorm epsilon for numerical stability
        bidirectional: bool = True,     # Enable bidirectional processing
        bidirectional_merging: str = "add",  # Merge strategy: "concat", "add", "mul"
        output_activate_function: str = None,  # Optional output activation
    ):
        super().__init__()
        
        # Store configuration
        self.bidirectional = bidirectional
        self.bidirectional_merging = bidirectional_merging
        self.d_model = attention_in
        
        print(f"🔧 Initializing Mamba2Encoder: d_model={attention_in}, layers={num_layer}, bidirectional={bidirectional}")
        
        # Dynamic compatibility checking for different mamba_ssm versions
        # This ensures the encoder works across different library versions
        mamba_params = inspect.signature(Mamba2).parameters
        supports_headdim = 'headdim' in mamba_params  # Mamba2 feature
        supports_ngroups = 'ngroups' in mamba_params  # Mamba2 feature
        
        block_params = inspect.signature(Block).parameters
        needs_mlp_cls = 'mlp_cls' in block_params     # Some versions require MLP
        
        print(f"   Block needs mlp_cls: {needs_mlp_cls}")
        
        # Create Mamba2 mixer factory function
        # Each layer gets a unique layer_idx for proper parameter initialization
        def create_mamba_partial(layer_idx):
            """Factory function to create Mamba2 instances with layer-specific parameters."""
            kwargs = {
                'd_state': d_state,      # State space dimension
                'd_conv': d_conv,        # Convolution kernel size
                'expand': expand,        # Hidden dimension expansion
                'layer_idx': layer_idx   # Unique layer identifier
            }
            # Add Mamba2-specific parameters if supported by the library version
            if supports_headdim:
                kwargs['headdim'] = headdim
            if supports_ngroups:
                kwargs['ngroups'] = ngroups
            return partial(Mamba2, **kwargs)
        
        # Build forward direction Mamba2 blocks
        # These process the sequence from left to right (past to future)
        self.forward_blocks = nn.ModuleList([])
        for i in range(num_layer):
            block_kwargs = {
                'dim': attention_in,                          # Input/output dimension
                'mixer_cls': create_mamba_partial(i),         # Mamba2 state-space mixer
                'norm_cls': partial(RMSNorm, eps=rmsnorm_eps), # RMS normalization
                'fused_add_norm': False,                      # Separate norm and residual
            }
            
            # Add MLP component if required by the Block implementation
            if needs_mlp_cls:
                def simple_mlp_fn(dim):
                    return nn.Identity()  # No-op MLP for pure Mamba processing
                block_kwargs['mlp_cls'] = simple_mlp_fn
            
            self.forward_blocks.append(Block(**block_kwargs))
            
        # Build backward direction Mamba2 blocks (if bidirectional enabled)
        # These process the time-reversed sequence (future to past)
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(num_layer):
                block_kwargs = {
                    'dim': attention_in,
                    'mixer_cls': create_mamba_partial(i + num_layer),  # Unique layer_idx
                    'norm_cls': partial(RMSNorm, eps=rmsnorm_eps),
                    'fused_add_norm': False,
                }
                
                if needs_mlp_cls:
                    def simple_mlp_fn(dim):
                        return nn.Identity()
                    block_kwargs['mlp_cls'] = simple_mlp_fn
                
                self.backward_blocks.append(Block(**block_kwargs))

        # Output projection layer for bidirectional concatenation
        # When using "concat" merging, the output dimension doubles (forward + backward)
        # This projection brings it back to the original dimension
        if bidirectional and bidirectional_merging == "concat":
            self.output_proj = nn.Linear(attention_in * 2, attention_in)
        else:
            self.output_proj = None

        # === Output Activation Function (ConformerEncoder Compatibility) ===
        # Support the same activation functions as ConformerEncoder for drop-in replacement
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            elif output_activate_function == "Sigmoid":
                self.activate_function = nn.Sigmoid()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {output_activate_function}. "
                    f"Supported: Tanh, ReLU, ReLU6, LeakyReLU, PReLU, Sigmoid"
                )
        self.output_activate_function = output_activate_function

        # === Weight Initialization ===
        # Use Mamba's official weight initialization scheme
        self.apply(partial(_init_weights, n_layer=num_layer))
        
        print(f"✅ Mamba2Encoder initialized successfully with {sum(p.numel() for p in self.parameters()):,} parameters")
        print(f"   Architecture: {'Bidirectional' if bidirectional else 'Unidirectional'} "
              f"Mamba2 with {num_layer} layers per direction")
        print(f"   State space dimension: {d_state}, Expansion factor: {expand}")
        print(f"   Bidirectional merging: {bidirectional_merging if bidirectional else 'N/A'}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Mamba2-based encoder.
        
        The forward pass consists of:
        1. Forward direction processing (left-to-right through time)
        2. Backward direction processing (right-to-left through time) [if bidirectional]
        3. Merging of bidirectional outputs using specified strategy
        4. Optional output activation
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, time, features).
                         Same dimensions as input for compatibility with ConformerEncoder.
        """
        batch_size, seq_len, features = x.shape
        
        # === Forward Direction Processing ===
        # Process sequence from past to future (standard temporal order)
        for_residual = None
        forward_f = x.clone()
        for block in self.forward_blocks:
            # Each block returns: (hidden_states, residual)
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        # Apply final residual connection
        forward_output = (forward_f + for_residual) if for_residual is not None else forward_f

        # === Backward Direction Processing (if enabled) ===
        if self.bidirectional:
            back_residual = None
            # Flip input sequence along time dimension for backward processing
            backward_f = torch.flip(x, [1])  # (batch, time, features) -> reversed time
            
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            backward_output = (backward_f + back_residual) if back_residual is not None else backward_f
            
            # Flip output back to original temporal order
            backward_output = torch.flip(backward_output, [1])
            
            # === Merge Bidirectional Outputs ===
            if self.bidirectional_merging == "concat":
                # Concatenate: [forward; backward] -> double dimension
                output = torch.cat([forward_output, backward_output], dim=-1)
                # Project back to original dimension for compatibility
                if self.output_proj is not None:
                    output = self.output_proj(output)
            elif self.bidirectional_merging == "add":
                # Element-wise addition: forward + backward
                output = forward_output + backward_output
            elif self.bidirectional_merging == "mul":
                # Element-wise multiplication: forward * backward
                output = forward_output * backward_output
            else:
                raise ValueError(f"Invalid bidirectional_merging: {self.bidirectional_merging}")
        else:
            # Use only forward direction output
            output = forward_output

        # === Apply Output Activation (if specified) ===
        if self.output_activate_function:
            output = self.activate_function(output)
            
        return output