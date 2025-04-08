import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha:torch.Tensor=None, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs are assumed to be logits
        if self.alpha and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        # compute: -alpha * ((1 - pt)^gamma) * log(pt) - alpha already in ce
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
