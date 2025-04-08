import torch
from torch import nn

class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(self,layers: list[int], dropout:float=0., batch_norm=False, layer_norm=False, final_function=None) -> None:
        super().__init__()

        self.model = nn.Sequential()
        for i in range(len(layers) - 1):
            self.model.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # Apply ReLU and BatchNorm for all layers except the last one
                if batch_norm:
                    self.model.add_module(f"batchnorm_{i}", nn.BatchNorm1d(layers[i+1]))
                if layer_norm:
                    self.model.add_module(f"layernorm_{i}", nn.LayerNorm(layers[i+1]))
                self.model.add_module(f"relu_{i}", nn.ReLU())
            if dropout > 0:
                self.model.add_module(f"dropout_{i}", nn.Dropout(dropout))
        if final_function is not None:
            self.model.add_module("final", final_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size()[0]

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(b, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
