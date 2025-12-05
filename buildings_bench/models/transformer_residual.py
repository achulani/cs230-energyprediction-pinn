"""
Transformer-based Residual Network for Building Energy Forecasting.

This module implements a lightweight encoder-decoder transformer
with self-attention for predicting residual corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        # Add positional encoding: pe is (max_len, 1, d_model), need to transpose and slice
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].transpose(0, 1)  # (1, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with residual
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerBasedResidualNetwork(nn.Module):
    """
    Transformer-based residual network with encoder-decoder architecture.
    
    Architecture:
    1. Feature embedding layer
    2. Positional encoding
    3. Transformer encoder (processes context)
    4. Transformer decoder (processes prediction window with cross-attention to context)
    5. Output projection for residual prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        context_len: int = 168,
        pred_len: int = 24
    ):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feed-forward dimension
            dropout: Dropout probability
            context_len: Length of context window
            pred_len: Length of prediction window
        """
        super().__init__()
        
        self.d_model = d_model
        self.context_len = context_len
        self.pred_len = pred_len
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ]
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Transformer decoder
        decoder_layers = [
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ]
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Generate square subsequent mask for decoder
        self.register_buffer('tgt_mask', self._generate_square_subsequent_mask(pred_len))
        
        # Output projection for residual prediction
        self.output_proj = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, context_len: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
               where seq_len = context_len + pred_len
            context_len: Length of context window (if None, uses self.context_len)
        
        Returns:
            Residual predictions of shape (batch, pred_len, 1)
        """
        if context_len is None:
            context_len = self.context_len
        
        batch_size, seq_len, _ = x.shape
        pred_len = seq_len - context_len
        
        # Split into context and prediction windows
        context = x[:, :context_len, :]  # (batch, context_len, input_dim)
        pred_input = x[:, context_len:, :]  # (batch, pred_len, input_dim)
        
        # Input projection
        context_emb = self.input_proj(context)  # (batch, context_len, d_model)
        pred_emb = self.input_proj(pred_input)  # (batch, pred_len, d_model)
        
        # Positional encoding
        context_emb = self.pos_encoder(context_emb)
        pred_emb = self.pos_encoder(pred_emb)
        
        # Transformer encoder (process context)
        memory = context_emb
        for encoder_layer in self.encoder:
            memory = encoder_layer(memory)
        
        # Transformer decoder (process prediction window with cross-attention)
        tgt = pred_emb
        tgt_mask = self.tgt_mask[:pred_len, :pred_len]
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, memory, tgt_mask=tgt_mask)
        
        # Output projection
        residual = self.output_proj(tgt)  # (batch, pred_len, 1)
        
        return residual

