import torch
import torch.nn as nn
import torch.nn.functional as F

class VaeGaussCriterion(nn.Module):
    def __init__(self):
        super(VaeGaussCriterion, self).__init__()

    def forward(self, mu, logvar):
        try:
            loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #/ mu.size(0)
        except:
            print("blowup!!!")
            print("logvar", torch.sum(logvar.data), torch.sum(torch.abs(logvar.data)), torch.max(logvar.data), torch.min(logvar.data))
            print("mu", torch.sum(mu.data), torch.sum(torch.abs(mu.data)), torch.max(mu.data), torch.min(mu.data))
            return 0
        return loss
    
# class BoxL1Criterion(nn.Module):
#     def __init__(self):
#         super(BoxL1Criterion, self).__init__()

#     def forward(self, pred, target):
#         loss = F.l1_loss(pred, target)
#         return loss
    
class BoxL1Criterion(nn.Module):
    def __init__(self, angle_weight=1.0):
        super(BoxL1Criterion, self).__init__()
        self.angle_weight = angle_weight

    def forward(self, pred, target):
        # Split位置参数 & 角度
        pred_box = pred[..., :4]
        target_box = target[..., :4]

        pred_angle = pred[..., 4]
        target_angle = target[..., 4]

        # 位置 + 尺寸 L1
        loc_loss = F.l1_loss(pred_box, target_box)

        # 仅对非 __image__ 的对象计算角度损失（其框恒为 [0.5, 0.5, 1.0, 1.0]）
        image_mask = (target_box == torch.tensor([0.5, 0.5, 1.0, 1.0], device=target_box.device)).all(dim=-1)
        valid_mask = ~image_mask

        if valid_mask.any():
            angle_diff = pred_angle[valid_mask] - target_angle[valid_mask]
            angle_loss = torch.mean(1 - torch.cos(angle_diff))  # ∈ [0, 2]
        else:
            angle_loss = torch.tensor(0.0, device=target_box.device)

        return loc_loss + self.angle_weight * angle_loss
