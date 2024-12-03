from loader import *
from ultralytics import YOLO
import torch.nn as nn 

if torch.backend.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} backend.")

model = YOLO('yolov8m.pt')
model.to(device)
model.train()


# retina-net loss function: Focal Loss + Smooth L1 Loss

"""
Focal Loss: - sum (ai * ((1-pi)^y) * logb(pi))
"""

class FocalLoss(nn.Module):
    """
        Args:
            alpha (float): Weighting factor for class 1.
            gamma (float): Focusing parameter to reduce the relative loss for well-classified examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """

    def __init__(self,alpha = 0.25, gamma = 2, reduction = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: (N, C) where C = number of classes
        # targets: (N) where each value is 0 <= targets[i] <= C-1

        num_classes = inputs.size(1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes).float()
        probs = nn.functional.softmax(inputs, dim = 1)

        pi = torch.sum(probs*targets_one_hot, dim = 1)
        log_pi = torch.log(pi + 1e-6)
        focal_loss = -self.alpha * ((1-pi)**self.gamma) * log_pi
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Loss(nn.Module):

    def __init__(self, num_classes):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.cls_loss = nn.FocalLoss(alpha = 0.25, gamma = 2.0, eduction = 'mean')
        self.reg_loss = nn.SmoothL1Loss(reduction = 'mean')

    def forward(self,predictions,targets):

        cls_loss = 0
        reg_loss = 0
        for pred,tar in zip(predictions,targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scrores']
            pred_labels = pred['labels']

            gt_boxes = tar['boxes'].to(device)
            gt_labels = tar['labels'].to(device)

            if gt_boxes.size(0) == 0:
                continue

            cls_loss += self.cls_loss(pred_labels, gt_labels)
            reg_loss += self.reg_loss(pred_boxes,gt_boxes)

        total_loss = cls_loss + reg_loss
        return total_loss

criterion = Loss(num_classes = 15)

