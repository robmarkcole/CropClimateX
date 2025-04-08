from torch import dropout, nn
import torch.nn.functional as F
import torch
from .base import time_distribute

class LSTM(torch.nn.Module):
    def __init__(self, input_dim=15, num_classes=0, hidden_dims=128, num_layers=4, dropout=0.6, 
                 bidirectional=True, use_layernorm=True, return_sequence=False, return_state=False):
        """num_classes: adds a linear layer with the number of classes to predict"""
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        self.return_sequence = return_sequence
        self.return_state = return_state

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.olayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional))

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_dims = hidden_dims * 2

        if num_classes:
            self.linear_class = nn.Linear(hidden_dims, num_classes, bias=True)
        else:
            self.linear_class = nn.Identity()

    def forward(self, x, states=None):
        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs, (h_n, c_n) = self.lstm(x, states)

        if self.use_layernorm:
            outputs = self.olayernorm(outputs)
        
        if self.return_sequence:
            b,t,c = outputs.shape
            # apply linear layer to each time step
            outputs = time_distribute(outputs)
            outputs = self.linear_class(outputs)
            outputs = time_distribute(outputs, time_size=t)
        else: # return last step
            outputs = self.linear_class(outputs[:,-1])
        if self.return_state:
            return outputs, (h_n, c_n)
        return outputs

if __name__ == "__main__":
    _ = LSTM()
