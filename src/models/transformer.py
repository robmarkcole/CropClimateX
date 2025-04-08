from torch import dropout, nn
import torch.nn.functional as F
import torch
from torch import Tensor
import math

class PositionalEncoding(nn.Module):
    # adapted from here: https://pytorch.org/tutorials/beginner/translation_transformer.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEnc(nn.Module):
    """A simple Transformer Encoder."""
    def __init__(self, num_layers, layer_norm=True, **kwargs):
        """batch_first is always applied."""
        super().__init__()

        if not all(k in kwargs for k in ['d_model', 'nhead']):
            raise ValueError("d_model and nhead are required arguments.")
        d_model = kwargs['d_model']

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **kwargs)

        if layer_norm:
            if 'layer_norm_eps' in kwargs:
                encoder_norm = nn.LayerNorm(d_model, eps=kwargs['layer_norm_eps'])
            else:
                encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm = None
        
        self.net = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pos_enc = PositionalEncoding(d_model, dropout=kwargs.get('dropout', 0.1))

    def forward(self, x):
        x = self.pos_enc(x)
        return self.net(x)
