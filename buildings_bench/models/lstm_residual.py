"""
LSTM-based Residual Network for Building Energy Forecasting.

This module implements a bidirectional LSTM with attention mechanism
to predict residual corrections for LightGBM predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on relevant historical patterns."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len_q, hidden_size]
            key: [batch, seq_len_k, hidden_size]
            value: [batch, seq_len_k, hidden_size]
        Returns:
            [batch, seq_len_q, hidden_size]
        """
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class LSTMBasedResidualNetwork(nn.Module):
    """
    LSTM-based residual network with bidirectional LSTM and attention.
    
    Architecture:
    1. Feature embedding layer
    2. Bidirectional LSTM encoder
    3. Attention mechanism
    4. Residual connections and layer normalization
    5. Output projection for residual prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_residual: bool = True,
        use_layer_norm: bool = True
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # LSTM output is bidirectional, so we need to project it
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network with residual
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Output projection for residual prediction
        self.output_proj = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[(n // 4):(n // 2)].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
        
        Returns:
            Residual predictions of shape (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x_proj = self.input_proj(x)  # (batch, seq_len, hidden_size)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x_proj)  # (batch, seq_len, hidden_size * 2)
        lstm_out = self.lstm_proj(lstm_out)  # (batch, seq_len, hidden_size)
        
        # Residual connection 1
        if self.use_residual:
            lstm_out = lstm_out + x_proj
        
        # Layer normalization 1
        if self.use_layer_norm:
            lstm_out = self.layer_norm1(lstm_out)
        
        # Attention mechanism (self-attention)
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Residual connection for attention
            if self.use_residual:
                attn_out = attn_out + lstm_out
            lstm_out = attn_out
        
        # Feed-forward network
        ffn_out = self.ffn(lstm_out)
        
        # Residual connection 2
        if self.use_residual:
            ffn_out = ffn_out + lstm_out
        
        # Layer normalization 2
        if self.use_layer_norm:
            ffn_out = self.layer_norm2(ffn_out)
        
        # Output projection
        residual = self.output_proj(ffn_out)  # (batch, seq_len, 1)
        
        return residual

