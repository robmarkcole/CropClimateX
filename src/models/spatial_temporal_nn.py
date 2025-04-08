from torch import dropout, nn
import torch.nn.functional as F
import torch
from .base import time_distribute, TimeDistributed
    
class SpatialTemporalNN(torch.nn.Module):
    """Combine a spatial and temporal NN. The spatial NN is applied on every time step and the output is fed to the temporal NN."""
    def __init__(self, spatial: torch.nn.Module, activation: torch.nn.Module, temporal: torch.nn.Module, final_nn: torch.nn.Module=None):
        super().__init__()
        self.spatial = TimeDistributed(nn.Sequential(spatial, activation))
        self.temporal = temporal

        # NN to apply on the final output for every time step (e.g. to reduce the dimensionality)
        if final_nn is None:
            self.final_nn = final_nn
        else:
            self.final_nn = TimeDistributed(final_nn)

    def forward(self,x):
        x = self.spatial(x)
        x = self.temporal(x)
        if self.final_nn is not None:
            x = self.final_nn(x)
        return x