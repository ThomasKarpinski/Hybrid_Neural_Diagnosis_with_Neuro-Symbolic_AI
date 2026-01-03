import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
    Reduces the loss for well-classified examples, focusing on hard negatives.
    
    Args:
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative classes.
                       For class 1, weight is alpha. For class 0, weight is 1-alpha.
        gamma (float): Focusing parameter. Higher gamma focuses more on hard examples.
        reduction (str): 'mean' or 'sum'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits (before sigmoid)
        # targets: binary labels
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # probabilities of the true class
        
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
