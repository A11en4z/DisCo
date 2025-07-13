import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import box_iou_rotated



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

    # def forward(self, pred, target):
    #     # Split位置参数 & 角度
    #     pred_box = pred[..., :4]
    #     target_box = target[..., :4]

    #     pred_angle = pred[..., 4]
    #     target_angle = target[..., 4]

    #     # 位置 + 尺寸 L1
    #     loc_loss = F.l1_loss(pred_box, target_box) 
        
    #     # Mask 掉最后一个 box（__image__）,容器不需要预测角度
    #     if pred.size(0) > 1:
    #         pred_angle = pred_angle[:-1]
    #         target_angle = target_angle[:-1]
            
    #     # 角度周期 loss（平滑 + 可导）
    #     angle_diff = pred_angle - target_angle
    #     angle_loss = torch.mean(1 - torch.cos(angle_diff))  # ∈ [0, 2]

    #     # 总损失
    #     return loc_loss + self.angle_weight * angle_loss
    
    def forward(self, pred, target):
        # 位置信息
        pred_box = pred[..., :4]
        target_box = target[..., :4]
        loc_loss = F.l1_loss(pred_box, target_box)

        # 角度信息
        pred_angle_vec = pred[..., 4:6]  # [cosθ, sinθ]
        angle_rad = target[..., 4]       # 弧度
        target_angle_vec = torch.stack([
            torch.cos(angle_rad),
            torch.sin(angle_rad)
        ], dim=-1)

        # Mask 掉 __image__
        if pred.size(0) > 1:
            pred_angle_vec = pred_angle_vec[:-1]
            target_angle_vec = target_angle_vec[:-1]

        angle_loss = F.mse_loss(pred_angle_vec, target_angle_vec)

        return loc_loss + self.angle_weight * angle_loss

# 减少框之间的重叠度
class BoxRepelLoss(nn.Module):
    def __init__(self, repel_margin=0.08, min_size=0.02, 
                 size_weight=1.0, iou_weight=1.0, iou_margin=0.1):
        super().__init__()
        self.repel_margin = repel_margin
        self.min_size = min_size
        self.size_weight = size_weight
        self.iou_weight = iou_weight
        self.iou_margin = iou_margin

    def forward(self, pred):  
        if pred.size(0) <= 1:
            return torch.tensor(0.0, device=pred.device)

        # 去掉 __image__ 容器损失
        pred = pred[:-1]

        total_loss = 0.0

        # 目标中心排斥损失
        centers = pred[:, :2]
        dist = torch.cdist(centers, centers)
        mask = torch.eye(dist.size(0), device=dist.device).bool()
        dist = dist[~mask].view(dist.size(0), -1)
        repel_loss = torch.clamp(self.repel_margin - dist, min=0).mean()

        # 尺寸最小约束
        w, h = pred[:, 2], pred[:, 3]
        size_penalty = torch.clamp(self.min_size - w, min=0) + torch.clamp(self.min_size - h, min=0)
        size_loss = size_penalty.mean()

        total_loss += repel_loss + self.size_weight * size_loss

        # Rotated IoU 惩罚
        if self.iou_weight > 0:
            cos_theta, sin_theta = pred[:, 4], pred[:, 5]
            theta = torch.atan2(sin_theta, cos_theta)
            obb = torch.cat([pred[:, :4], theta.unsqueeze(-1)], dim=-1)  # [N, 5]
            ious = box_iou_rotated(obb, obb, aligned = False)
            ious = ious - torch.eye(len(obb), device=obb.device)
            iou_loss = torch.clamp(ious - self.iou_margin, min=0).mean()
            total_loss += self.iou_weight * iou_loss

        return total_loss

def compute_rotated_iou_loss(pred):  # pred: [N, 6], 最后一列是 cosθ/sinθ
    if pred.size(0) <= 1:
        return torch.tensor(0.0, device=pred.device)

    pred = pred[:-1]  # 去掉 __image__
    
    # 1. 恢复角度 θ ∈ [-π, π]
    cos_theta, sin_theta = pred[:, 4], pred[:, 5]
    theta = torch.atan2(sin_theta, cos_theta)

    # 2. 构造 OBB：[cx, cy, w, h, θ]
    obb = torch.cat([pred[:, :4], theta.unsqueeze(-1)], dim=-1)  # [N, 5]

    # 3. 计算两两旋转 IoU
    ious = box_iou_rotated(obb, obb, is_aligned=False)  # [N, N]
    ious = ious - torch.eye(len(obb), device=obb.device)  # 去掉对角线

    # 4. 惩罚 IoU > margin 的重叠框对
    margin = 0.0
    loss = torch.clamp(ious - margin, min=0).mean()
    return loss

