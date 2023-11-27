import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from segmentation_models_pytorch.losses.tversky import soft_tversky_score

ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class TverskyScore(torch.nn.Module):
    """ Implementation of Tversky score for image segmentation task. 
        Where TP and FP is weighted by alpha and beta params.
        With alpha == beta == 0.5, this score becomes equal DiceScore.

    Args:
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_score(self, output, target, smooth=1e-15, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_score(*args, **kwargs)


class IoU(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps

    def __call__(self, output, target):
        intersection = (target * output).sum()
        union = target.sum() + output.sum() - intersection
        result = (intersection + self.eps) / (union + self.eps)
        return result
