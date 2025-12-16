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

    def forward(self, pred, target, objs=None, class_weights=None):
        # 支持类别加权的几何拟合损失：
        # - 几何位置/尺寸采用逐对象加权平均；
        # - 角度一致性仅作用于非 __image__ 的对象；
        # - class_weights 为类别→权重张量，若为空则退化为普通平均。
        pred_box = pred[..., :4].float()
        target_box = target[..., :4].float()
        pred_angle = pred[..., 4].float()
        target_angle = target[..., 4].float()
        if class_weights is not None and objs is not None:
            w = class_weights[objs.long()].float()
        else:
            w = None
        loc_diff = torch.abs(pred_box - target_box)
        loc_loss_i = loc_diff.mean(dim=-1)
        pred_w, pred_h = pred_box[..., 2], pred_box[..., 3]
        tgt_w, tgt_h = target_box[..., 2], target_box[..., 3]
        area_pred = pred_w * pred_h
        area_tgt = tgt_w * tgt_h
        area_l1_i = torch.abs(area_pred - area_tgt)
        geom_i = loc_loss_i + 0.5 * area_l1_i
        if w is not None:
            geom_loss = (geom_i * w).sum() / (w.sum() + 1e-6)
        else:
            geom_loss = geom_i.mean()
        image_mask = (target_box == torch.tensor([0.5, 0.5, 1.0, 1.0], device=target_box.device)).all(dim=-1)
        valid_mask = ~image_mask
        if valid_mask.any():
            angle_diff = pred_angle[valid_mask] - target_angle[valid_mask]
            base = (1 - torch.cos(angle_diff))
            if w is not None:
                wv = w[valid_mask]
                angle_loss = (base * wv).sum() / (wv.sum() + 1e-6)
            else:
                angle_loss = base.mean()
        else:
            angle_loss = torch.tensor(0.0, device=target_box.device, dtype=geom_loss.dtype)
        return geom_loss + self.angle_weight * angle_loss

class SameClassOverlapCriterion(nn.Module):
    def __init__(self):
        super(SameClassOverlapCriterion, self).__init__()

    def forward(self, pred, objs, obj_to_img, image_idx=None, class_weights=None, mode: str = 'oriented', theta_eps: float = 0.087):
        # 同类重叠抑制：对同一图像内、同类别对象对的 IoU 求平均作为惩罚项
        # - 输入为预测的 [cx, cy, w, h, a]，注意此处只用到 cx,cy,w,h；
        # - 支持按类别权重调整同类对的惩罚强度；
        # - 跳过 __image__ 类的框。
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
        cx = pred[:, 0].float()
        cy = pred[:, 1].float()
        w = pred[:, 2].float()
        h = pred[:, 3].float()
        a = pred[:, 4].float()
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0
        x1 = cx + w / 2.0
        y1 = cy + h / 2.0
        imgs = obj_to_img.long()
        classes = objs.long()
        total = torch.tensor(0.0, device=pred.device)
        count = 0.0
        for img_id in imgs.unique().tolist():
            mask = (imgs == img_id)
            idx = mask.nonzero().view(-1)
            if idx.numel() <= 1:
                continue
            ci = classes[idx]
            if image_idx is not None:
                keep = (ci != int(image_idx))
                idx = idx[keep]
                ci = ci[keep]
                if idx.numel() <= 1:
                    continue
            xi0 = x0[idx]
            yi0 = y0[idx]
            xi1 = x1[idx]
            yi1 = y1[idx]
            cxi = cx[idx]
            cyi = cy[idx]
            wi = w[idx]
            hi = h[idx]
            ai = a[idx]
            for k in range(idx.numel()):
                for l in range(k + 1, idx.numel()):
                    if ci[k] != ci[l]:
                        continue
                    if mode == 'aabb' or torch.abs(ai[k] - ai[l]) < theta_eps:
                        xx0 = torch.maximum(xi0[k], xi0[l])
                        yy0 = torch.maximum(yi0[k], yi0[l])
                        xx1 = torch.minimum(xi1[k], xi1[l])
                        yy1 = torch.minimum(yi1[k], yi1[l])
                        inter_w = torch.clamp(xx1 - xx0, min=0.0)
                        inter_h = torch.clamp(yy1 - yy0, min=0.0)
                        inter = inter_w * inter_h
                        area_k = (xi1[k] - xi0[k]) * (yi1[k] - yi0[k])
                        area_l = (xi1[l] - xi0[l]) * (yi1[l] - yi0[l])
                        union = area_k + area_l - inter + 1e-6
                        iou = inter / union
                    else:
                        # 平均轴投影交叠近似（低开销旋转重叠）
                        theta_avg = 0.5 * (ai[k] + ai[l])
                        u = torch.stack([torch.cos(theta_avg), torch.sin(theta_avg)])
                        v = torch.stack([-torch.sin(theta_avg), torch.cos(theta_avg)])
                        d = torch.stack([cxi[l] - cxi[k], cyi[l] - cyi[k]])
                        du = torch.abs(torch.dot(d, u))
                        dv = torch.abs(torch.dot(d, v))
                        dtk = ai[k] - theta_avg
                        dtl = ai[l] - theta_avg
                        hu_k = torch.abs((wi[k] * 0.5) * torch.cos(dtk)) + torch.abs((hi[k] * 0.5) * torch.sin(dtk))
                        hv_k = torch.abs((wi[k] * 0.5) * torch.sin(dtk)) + torch.abs((hi[k] * 0.5) * torch.cos(dtk))
                        hu_l = torch.abs((wi[l] * 0.5) * torch.cos(dtl)) + torch.abs((hi[l] * 0.5) * torch.sin(dtl))
                        hv_l = torch.abs((wi[l] * 0.5) * torch.sin(dtl)) + torch.abs((hi[l] * 0.5) * torch.cos(dtl))
                        lu = torch.clamp(hu_k + hu_l - du, min=0.0)
                        lv = torch.clamp(hv_k + hv_l - dv, min=0.0)
                        inter = lu * lv
                        area_k = wi[k] * hi[k]
                        area_l = wi[l] * hi[l]
                        union = area_k + area_l - inter + 1e-6
                        iou = inter / union
                    if class_weights is not None:
                        wk = class_weights[ci[k]]
                        wl = class_weights[ci[l]]
                        wpair = (wk + wl) / 2.0
                        total = total + iou * wpair
                    else:
                        total = total + iou
                    count += 1.0
        if count == 0.0:
            return torch.tensor(0.0, device=pred.device)
        return total / count

class CategoryPriorCriterion(nn.Module):
    def __init__(self):
        super(CategoryPriorCriterion, self).__init__()

    def forward(self, pred, objs, priors, use_angle_prior=False, class_weights=None):
        # 类别几何先验正则：将每对象的 (w,h,log(w/h))（可选附加角度）
        # 与对应类别的高斯先验比较，采用马氏距离作为惩罚；支持类别权重。
        # 低样本类别的先验建议通过累计统计/平滑处理。
        if pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device)
        w = pred[:, 2].float()
        h = pred[:, 3].float()
        a = pred[:, 4].float()
        r = torch.log((w / (h + 1e-6)) + 1e-6)
        if use_angle_prior:
            feats = torch.stack([w, h, r, a], dim=-1)
        else:
            feats = torch.stack([w, h, r], dim=-1)
        classes = objs.long()
        losses = []
        weights = []
        for i in range(feats.shape[0]):
            ci = int(classes[i].item())
            prior = priors.get(ci, None)
            if prior is None:
                continue
            mean = prior["mean"].to(pred.device).float()
            cov = prior["cov"].to(pred.device).float()
            diff = feats[i] - mean
            inv = torch.inverse(cov + torch.eye(cov.shape[0], device=cov.device) * 1e-6)
            d = torch.matmul(torch.matmul(diff.unsqueeze(0), inv), diff.unsqueeze(1)).squeeze(0).squeeze(0)
            losses.append(d)
            if class_weights is not None:
                weights.append(class_weights[ci])
            else:
                weights.append(torch.tensor(1.0, device=pred.device))
        if len(losses) == 0:
            return torch.tensor(0.0, device=pred.device)
        losses = torch.stack(losses)
        weights = torch.stack(weights)
        return (losses * weights).sum() / (weights.sum() + 1e-6)
