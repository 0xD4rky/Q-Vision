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

def FocalLoss(alpha = 0.25, gamma = 2.0, logits, targets):

    bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')
    probas = torch.sigmoid(logits)
    


class Loss(nn.Module):

    def __init__(self, num_classes):
        super(Loss, self).__init__()
        self.num_classes = num_classes
        self.cls_loss = FocalLoss(alpha = 0.25, gamma = 2.0, eduction = 'mean')
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

