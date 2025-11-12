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
        self.box_dim = fourier_freqs * 2 * 4
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
        cx, cy, w, h = box[0], box[1], box[2], box[3]
        x1 = int(torch.round((cx - w / 2).clamp(0, 1) * image_size).item())
        y1 = int(torch.round((cy - h / 2).clamp(0, 1) * image_size).item())
        x2 = int(torch.round((cx + w / 2).clamp(0, 1) * image_size).item())
        y2 = int(torch.round((cy + h / 2).clamp(0, 1) * image_size).item())
        # 修正索引顺序为 [y, x]
        att_masks[idx][y1:y2, x1:x2] = 1
        return att_masks
    
    def get_attention_mask(self, box_masks):
        B = box_masks.shape[0]
        HW =  box_masks.shape[2] *  box_masks.shape[3]
        N = HW + box_masks.shape[1]
        
        n_objs =  box_masks.shape[1]
        attention_mask =  torch.ones(B, 1, N, N).type(box_masks.dtype).to(box_masks.device)

        visual_attention_mask = box_masks.view(B * n_objs, HW, 1)
        visual_attention_mask = torch.bmm(visual_attention_mask, visual_attention_mask.permute(0,2,1))
        visual_attention_mask = visual_attention_mask.view(B, n_objs , HW, HW).sum(dim=1)
        visual_attention_mask[visual_attention_mask > 1] = 1
        attention_mask[:, :, :HW, :HW] = visual_attention_mask.view(B, 1, HW, HW)

        cond_attention_masks =  box_masks.view(B, 1, n_objs, HW)
        attention_mask[:, :, HW:, :HW] = cond_attention_masks
        attention_mask[:, :, :HW, HW:] = cond_attention_masks.permute(0,1,3,2)
        diagonal_epsilon = torch.eye(N, device=box_masks.device).view(1,1,N,N) * 1e-9
        attention_mask = attention_mask + diagonal_epsilon

        return attention_mask


    def prepare_data(self, boxes, text_embeddings, obj_to_img):
        device = boxes.device
        B = int(obj_to_img.max().item() + 1)
        max_objs = 32

        masks = torch.zeros(B, max_objs, device=device)
        text_out = torch.zeros(B, max_objs, 768, device=device)
        boxes_in = boxes[..., :4]
        box_out = torch.zeros(B, max_objs, 4, device=device)
        instance_mask = torch.zeros(B, max_objs, 64, 64, device=device)

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




