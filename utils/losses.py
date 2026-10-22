"""
Loss Functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
    
    def forward(self, inputs, targets):
        """
        Compute focal loss
        
        Args:
            inputs: Prediction probabilities
            targets: Ground truth labels
        
        Returns:
            Focal loss
        """
        pt = inputs
        inputs = inputs - (inputs - 0.5) ** 7
        Focal_loss = -self.alpha * (1 - inputs) ** self.gamma * torch.log(inputs) * targets - \
                     (1 - self.alpha) * (inputs) ** self.gamma * torch.log(1 - inputs) * (1 - targets)
        if self.reduce:
            return torch.mean(Focal_loss)
        else:
            return Focal_loss

