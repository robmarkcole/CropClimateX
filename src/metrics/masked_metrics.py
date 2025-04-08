from torch import Tensor
import torchmetrics
import torch

class R2(torchmetrics.R2Score):
    def __init__(self, masked=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.masked = masked
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if self.masked:
            masked_preds = preds * mask
            masked_target = target * mask
            # Call the base class update method
            # reshape since R2Score only accepts 1D or 2D
            super().update(masked_preds.view(-1), masked_target.view(-1))
            self.total -= (1 - mask).sum().type(self.total.dtype) # subtract the number of masked values
        else:
            super().update(preds.view(-1), target.view(-1))

    def compute(self):
        if self.masked:
            if self.total == 0:
                return torch.tensor(0.0)
        return super().compute()

class MSE(torchmetrics.MeanSquaredError):
    def __init__(self, masked=False, squared: bool = True, **kwargs) -> None:
        super().__init__(squared, **kwargs)
        self.masked = masked
    
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if self.masked:
            masked_preds = preds * mask
            masked_target = target * mask
            # Call the base class update method
            super().update(masked_preds, masked_target)
            # subtract number of masked values
            self.total -= (1 - mask).sum().type(self.total.dtype)
        else:
            super().update(preds, target)

    def compute(self):
        if self.masked:
            if self.total == 0: # in case all masked values -> division by zero
                self.total = torch.tensor(1.0)
        return super().compute()
    
class MAE(torchmetrics.MeanAbsoluteError):
    def __init__(self, masked=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.masked = masked

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if self.masked:
            masked_preds = preds * mask
            masked_target = target * mask
            # Call the base class update method
            super().update(masked_preds, masked_target)
            # subtract number of masked values
            self.total -= (1 - mask).sum().type(self.total.dtype)
        else:
            super().update(preds, target)

    def compute(self):
        if self.masked:
            if self.total == 0: # in case all masked values -> division by zero
                self.total = torch.tensor(1.0)
        return super().compute()
    
class MAPE(torchmetrics.MeanAbsolutePercentageError):
    def __init__(self, masked=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.masked = masked

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None):
        if self.masked:
            masked_preds = preds * mask
            masked_target = target * mask
            # Call the base class update method
            super().update(masked_preds, masked_target)
            # subtract number of masked values
            self.total -= (1 - mask).sum().type(self.total.dtype)
        else:
            super().update(preds, target)

    def compute(self):
        if self.masked:
            if self.total == 0: # in case all masked values -> division by zero
                self.total = torch.tensor(1.0)
        return super().compute()
