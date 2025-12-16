import sys
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn

class FourierEmbedder():
    def __init__(self, num_freqs=8, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)


class ObjectFusionTokenizer(nn.Module):
    def __init__(self):
        super(ObjectFusionTokenizer, self).__init__()

        fourier_freqs = 16
        self.text_dim = 768
        # 由 4 维扩到 5 维（包含 angle），让对象嵌入也感知旋转
        self.box_dim = fourier_freqs * 2 * 5
        embedding_dim = self.text_dim + self.box_dim
        self.box_encoder = FourierEmbedder(fourier_freqs)
        self.null_padding_embeddings = torch.nn.Parameter(torch.zeros([embedding_dim]))

        input_dim = embedding_dim
        hidden_dim = self.text_dim * 4
        output_dim = self.text_dim
        self.object_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.attribute_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, boxes, text_embeddings, obj_to_img):
        meta_data = self.prepare_data(boxes, text_embeddings, obj_to_img)

        masks = meta_data["masks"].unsqueeze(-1)
        box_embeddings = self.box_encoder(meta_data["boxes"])
        text_embeddings = meta_data["text_embeddings"]
        object_embeddings = torch.cat([text_embeddings, box_embeddings], dim=-1)
        padding_embeddings = self.null_padding_embeddings.view(1,1,-1)

        object_embeddings = object_embeddings * masks + (1 - masks) * padding_embeddings
        object_embeddings = self.object_mlp(object_embeddings)
        return object_embeddings, meta_data

    def get_instance_mask(self, att_masks, idx, box, image_size):
        # 角度感知的旋转矩形掩码（在 64x64 上栅格化）
        # box: [cx, cy, w, h, angle]，angle 为弧度 [-π, π]
        cx, cy, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
        a = box[4].item() if box.shape[0] >= 5 else 0.0

        cx_pix, cy_pix = cx * image_size, cy * image_size
        w_pix, h_pix = w * image_size, h * image_size
        dx, dy = w_pix / 2.0, h_pix / 2.0
        cos_t, sin_t = np.cos(a), np.sin(a)
        corners = [(-dx, -dy), (-dx,  dy), ( dx,  dy), ( dx, -dy)]
        points = [(cx_pix + cos_t*px - sin_t*py, cy_pix + sin_t*px + cos_t*py) for px, py in corners]

        from PIL import Image, ImageDraw
        mask_img = Image.new('L', (image_size, image_size), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(points, outline=1, fill=1)  # 0/1 掩码
        mask_np = np.array(mask_img, dtype=np.uint8)
        mask_t = torch.from_numpy(mask_np).to(att_masks.device)
        att_masks[idx][:] = mask_t
        return att_masks

    def get_attention_mask(self, box_masks):
        B = box_masks.shape[0]
        HW =  box_masks.shape[2] *  box_masks.shape[3]
        N = HW + box_masks.shape[1]

        n_objs =  box_masks.shape[1]
        # 显存优化：计算时用 float16
        box_masks_f = box_masks.to(dtype=torch.float16)

        attention_mask = torch.ones(B, 1, N, N, dtype=box_masks_f.dtype, device=box_masks.device)

        #############################################
        # visual_attention_mask = box_masks_f.view(B * n_objs, HW, 1)
        # visual_attention_mask = torch.bmm(visual_attention_mask, visual_attention_mask.permute(0,2,1))
        # visual_attention_mask = visual_attention_mask.view(B, n_objs , HW, HW).sum(dim=1)
        # visual_attention_mask = torch.clamp(visual_attention_mask, max=1.0)
        # attention_mask[:, :, :HW, :HW] = visual_attention_mask.view(B, 1, HW, HW)
        # 修改：将视觉自注意力从“并集”改为“逐对象隔离”
        box_masks_flat = box_masks_f.view(B, n_objs, HW)
        sum_mask = box_masks_flat.sum(dim=1)
        has_obj = (sum_mask > 0).to(torch.long)
        group_ids = torch.argmax(box_masks_flat, dim=1)
        group_ids = group_ids + (1 - has_obj) * n_objs
        group_eq = (group_ids.unsqueeze(-1) == group_ids.unsqueeze(-2)).to(attention_mask.dtype)
        attention_mask[:, :, :HW, :HW] = group_eq.unsqueeze(1)
        ##########################################

        cond_attention_masks =  box_masks_f.view(B, 1, n_objs, HW)
        attention_mask[:, :, HW:, :HW] = cond_attention_masks
        attention_mask[:, :, :HW, HW:] = cond_attention_masks.permute(0,1,3,2)
        eps = 1e-6 if attention_mask.dtype == torch.float16 else 1e-9
        diagonal_epsilon = torch.eye(N, device=box_masks.device, dtype=attention_mask.dtype).view(1,1,N,N) * eps
        attention_mask = attention_mask + diagonal_epsilon

        return attention_mask


    def prepare_data(self, boxes, text_embeddings, obj_to_img):
        device = boxes.device
        B = int(obj_to_img.max().item() + 1)
        max_objs = 32

        masks = torch.zeros(B, max_objs, device=device)
        text_out = torch.zeros(B, max_objs, 768, device=device)
        # 保留角度维度：5D [cx, cy, w, h, angle]
        boxes_in = boxes[..., :5]
        box_out = torch.zeros(B, max_objs, 5, device=device)
        # 显存优化：实例掩码用 uint8，后续计算 attention 时再转浮点
        instance_mask = torch.zeros(B, max_objs, 64, 64, device=device, dtype=torch.uint8)

        for b in range(B):
            idxs = torch.nonzero(obj_to_img == b, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue
            n = min(idxs.shape[0], max_objs)
            sel = idxs[:n]
            box_out[b, :n, :] = boxes_in[sel]
            text_out[b, :n, :] = text_embeddings[sel]
            masks[b, :n] = 1
            for j in range(n):
                instance_mask[b] = self.get_instance_mask(instance_mask[b], j, box_out[b, j], 64)

        attention_mask64 = self.get_attention_mask(instance_mask)

        meta_data = {
            "boxes": box_out,
            "masks": masks,
            "text_embeddings": text_out,
            "object_attention_masks": attention_mask64
        }
        return meta_data




