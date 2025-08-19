#!/usr/bin/env python3
"""
WavLM-Samba Speaker Diarization Model

This module implements a speaker diarization model that combines Microsoft's WavLM 
self-supervised speech model with Samba encoder (hybrid Mamba + Sliding Window Attention)
for efficient and accurate speaker segmentation and identification.

Architecture Overview:
1. WavLM Feature Extraction: Extracts multi-layer contextualized speech representations
2. Weighted Layer Fusion: Learns optimal combination of WavLM layers
3. Feature Projection: Projects WavLM features to encoder input dimension
4. Samba Encoder: Hybrid SSM+Attention for efficient temporal sequence modeling
5. Classification Head: Predicts speaker activity using powerset encoding

Key Advantages over WavLM-Mamba:
- Samba combines Mamba's efficiency with Sliding Window Attention's precision
- Better handling of complex speaker interactions and overlapping speech
- Improved performance on long audio sequences
- More balanced approach between global context and local precision
"""
import os
import torch
import torch.nn as nn

from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames, 
    multi_conv_receptive_field_size, 
    multi_conv_receptive_field_center
)

from diarizen.models.module.samba_encoder import SambaEncoder
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model
from diarizen.models.module.wavlm_config import get_config

class Model(BaseModel):
    """
    WavLM-Samba Speaker Diarization Model
    
    This model combines Microsoft's WavLM self-supervised speech model with Samba encoder
    for efficient speaker diarization. The architecture follows a pipeline approach:
    
    Raw Audio -> WavLM Features -> Layer Fusion -> Projection -> Samba Encoder -> Classification
    
    The Samba encoder provides a hybrid approach that combines:
    - Mamba: Efficient O(n) sequence modeling with selective state spaces
    - SwiGLU: Gated activation for improved expressivity
    - Sliding Window Attention: Precise memory retrieval within local windows
    
    This results in better performance than pure SSM (Mamba) or pure attention approaches.
    """
    
    def __init__(
        self,
        # === WavLM Configuration ===
        wavlm_src: str = "wavlm_base",           # WavLM model source (config name or checkpoint path)
        wavlm_layer_num: int = 13,               # Number of WavLM layers to use
        wavlm_feat_dim: int = 768,               # WavLM feature dimension (768 for base, 1024 for large)
        
        # === Samba Encoder Configuration ===
        attention_in: int = 256,                 # Encoder input/output dimension
        num_layers: int = 4,                     # Number of Samba blocks
        d_state: int = 16,                       # Mamba state space dimension
        d_conv: int = 4,                         # Mamba convolution kernel size
        expand: int = 2,                         # Mamba expansion factor
        d_intermediate: int = None,              # SwiGLU intermediate dimension (default: 8/3 * attention_in)
        window_size: int = 2048,                 # Sliding window attention size
        n_heads: int = 8,                        # Number of attention heads
        n_kv_heads: int = None,                  # Number of key/value heads (for GQA)
        dropout: float = 0.1,                    # Dropout probability
        
        # === Task Configuration ===
        max_speakers_per_chunk: int = 4,         # Maximum speakers per audio chunk
        chunk_size: int = 5,                     # Audio chunk duration in seconds
        num_channels: int = 8,                   # Number of input audio channels
        selected_channel: int = 0,               # Channel index to use for processing
        sample_rate: int = 16000,                # Audio sample rate in Hz
    ):
        # Initialize base model with pyannote.audio specifications
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        
        # Store audio processing configuration
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.selected_channel = selected_channel

        # === WavLM Feature Extraction Module ===
        # Load pre-trained WavLM model for robust speech representations
        self.wavlm_model = self.load_wavlm(wavlm_src)
        
        # Learnable weighted combination of WavLM layers
        # This learns optimal fusion of different abstraction levels from WavLM
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        # === Feature Processing Pipeline ===
        # Project WavLM features to encoder input dimension
        self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        # Layer normalization for stable training
        self.lnorm = nn.LayerNorm(attention_in)

        # === Samba Encoder (Core Architecture) ===
        # Hybrid architecture combining Mamba + Sliding Window Attention + SwiGLU
        # Superior to pure Mamba approach with added precision from attention
        self.samba = SambaEncoder(
            attention_in=attention_in,
            num_layers=num_layers, 
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            d_intermediate=d_intermediate,  
            n_heads=n_heads, 
            n_kv_heads=n_kv_heads,
            window_size=window_size,
            dropout=dropout,
        )

        # === Classification Head ===
        # Final linear layer for speaker activity prediction
        self.classifier = nn.Linear(attention_in, self.dimension)
        # Default activation (typically sigmoid for multi-label classification)
        self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        """
        Return parameters that are not part of the pre-trained WavLM model.
        
        This is useful for fine-tuning scenarios where you want to:
        1. Freeze WavLM parameters and only train task-specific layers
        2. Apply different learning rates to WavLM vs. task-specific components
        3. Implement gradual unfreezing strategies
        
        Returns:
        --------
        list: List of parameter groups excluding WavLM parameters
        """
        return [
            *self.weight_sum.parameters(),    # WavLM layer fusion weights
            *self.proj.parameters(),          # Feature projection layer
            *self.lnorm.parameters(),         # Layer normalization
            *self.samba.parameters(),         # Samba encoder parameters (Mamba + SWA + SwiGLU)
            *self.classifier.parameters(),    # Classification head
        ]

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        kernel_size = [10, 3, 3, 3, 3, 2, 2]
        stride = [5, 2, 2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    
    @property
    def get_rf_info(self):     
        """Return receptive field info to dataset
        """

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * self.sample_rate)
        duration = receptive_field_size / self.sample_rate
        step=receptive_field_step / self.sample_rate
        return num_frames, duration, step
    
    def load_wavlm(self, source: str):
        """
        Load a WavLM model from either a config name or a checkpoint file.
    
        Parameters
        ----------
        source : str
            - If `source` is a config name (e.g., "wavlm_large_md_s80"), 
            the model will be initialized using predefined configuration via `get_config()`.
            - If `source` is a file path (e.g., "pytorch_model.bin", "model.ckpt", or any local .pt file),
            the model will be loaded from the checkpoint, using its saved 'config' and 'state_dict'.
    
        Returns
        -------
        model : nn.Module
            Initialized WavLM model.
        """
        if os.path.isfile(source):
            # Load from checkpoint file
            ckpt = torch.load(source, map_location="cpu")
    
            if "config" not in ckpt or "state_dict" not in ckpt:
                raise ValueError("Checkpoint must contain 'config' and 'state_dict'.")
    
            for k, v in ckpt["config"].items():
                if 'prune' in k and v is not False:
                    raise ValueError(f"Pruning must be disabled. Found: {k}={v}")
    
            model = wavlm_model(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"], strict=False)
    
        else:
            # Load from predefined config
            config = get_config(source)
            model = wavlm_model(**config)
    
        return model
    
    def wav2wavlm(self, in_wav, model):
        """
        Transform raw waveform to WavLM multi-layer features.
        
        This method extracts contextualized speech representations from all
        WavLM transformer layers. Each layer captures different levels of
        linguistic and acoustic information:
        - Lower layers: Acoustic/phonetic features
        - Middle layers: Phoneme/word-level information  
        - Higher layers: Semantic/speaker information
        
        Parameters:
        -----------
        in_wav : torch.Tensor
            Input waveform tensor of shape (batch, samples)
        model : nn.Module
            Pre-trained WavLM model
            
        Returns:
        --------
        torch.Tensor
            Multi-layer WavLM features of shape (batch, frames, layers, features)
        """
        # Extract features from all WavLM transformer layers
        layer_reps, _ = model.extract_features(in_wav)
        # Stack layers as additional dimension: (batch, frames, layers, features)
        return torch.stack(layer_reps, dim=-1)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Raw audio waveform -> Speaker diarization predictions
        
        This method implements the complete WavLM-Samba pipeline:
        1. Channel selection (for multi-channel audio)
        2. WavLM feature extraction (multi-layer contextualized representations)
        3. Learnable layer fusion (weighted combination of WavLM layers)
        4. Feature projection and normalization
        5. Samba encoder (hybrid Mamba + SWA + SwiGLU processing)
        6. Classification head (speaker activity prediction)
        7. Output activation (typically sigmoid for multi-label)

        Parameters
        ----------
        waveforms : torch.Tensor
            Input audio waveforms of shape:
            - (batch, channels, samples) for multi-channel audio
            - Must have exactly 3 dimensions

        Returns
        -------
        scores : torch.Tensor
            Speaker activity predictions of shape (batch, frames, classes)
            - frames: Temporal dimension (~100Hz frame rate)
            - classes: Number of speaker classes (powerset or standard encoding)
            - Values in [0,1] after sigmoid activation
        """
        # Ensure input has correct dimensions
        assert waveforms.dim() == 3, f"Expected 3D input (batch, channels, samples), got {waveforms.dim()}D"
        
        # === Step 1: Channel Selection ===
        # Select specific channel for processing (useful for multi-channel arrays)
        waveforms = waveforms[:, self.selected_channel, :]  # (batch, samples)

        # === Step 2: WavLM Feature Extraction ===
        # Extract multi-layer contextualized speech representations
        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model)  # (batch, frames, layers, features)
        
        # === Step 3: Learnable Layer Fusion ===
        # Learn optimal weighted combination of WavLM layers
        wavlm_feat = self.weight_sum(wavlm_feat)  # (batch, frames, 1, features)
        wavlm_feat = torch.squeeze(wavlm_feat, -2)  # (batch, frames, features)

        # === Step 4: Feature Processing ===
        # Project WavLM features to encoder input dimension
        outputs = self.proj(wavlm_feat)  # (batch, frames, attention_in)
        # Apply layer normalization for stable training
        outputs = self.lnorm(outputs)
        
        # === Step 5: Samba Encoder ===
        # Hybrid sequence modeling: Mamba + SWA + SwiGLU
        # Superior to pure Mamba with added attention precision
        outputs = self.samba(outputs)  # (batch, frames, attention_in)

        # === Step 6: Classification Head ===
        # Predict speaker activity for each frame
        outputs = self.classifier(outputs)  # (batch, frames, num_classes)
        
        # === Step 7: Output Activation ===
        # Apply sigmoid for multi-label speaker activity prediction
        outputs = self.activation(outputs)  # (batch, frames, num_classes)

        return outputs 