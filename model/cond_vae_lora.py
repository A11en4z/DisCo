import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from .gcn import GraphTripleConvNet, _init_weights, build_mlp

class SceneVAEModel(nn.Module):
    def __init__(self, args, num_objs, num_rels, image_obj_idx=None):
        super(SceneVAEModel, self).__init__()

        gconv_dim = args.embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        box_embedding_dim = args.embedding_dim
        obj_embedding_dim = args.embedding_dim

        self.image_obj_idx = image_obj_idx if image_obj_idx is not None else 0

        self.obj_embeddings_encoder = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.obj_embeddings_decoder = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.rel_embeddings_encoder = nn.Embedding(num_rels, args.embedding_dim * 2)
        self.rel_embeddings_decoder = nn.Embedding(num_rels, args.embedding_dim * 2)
        self.box_embeddings = nn.Linear(5, box_embedding_dim)

        self.mlp_mean_var = build_mlp(
            [args.embedding_dim * 2 + 512, gconv_hidden_dim, args.embedding_dim * 2],
            batch_norm="layer",
            final_nonlinearity=True
        )
        self.mlp_mean = build_mlp(
            [args.embedding_dim * 2, box_embedding_dim],
            batch_norm="layer",
            final_nonlinearity=False
        )
        self.mlp_var = build_mlp(
            [args.embedding_dim * 2, box_embedding_dim],
            batch_norm="layer",
            final_nonlinearity=False
        )
        self.mlp_box = build_mlp(
            [args.embedding_dim * 2 + 512, gconv_hidden_dim, 5],
            batch_norm="layer",
            final_nonlinearity=False
        )

        self.cond_mlp = build_mlp(
            [gconv_dim * 2 + 512, 960, 768],
            batch_norm="layer",
            final_nonlinearity=False
        )

        gconv_encoder_kwargs = {
            'input_dim_obj':        gconv_dim * 2 + 512,
            'input_dim_pred':       gconv_dim * 2 + 512,
            'hidden_dim':           gconv_hidden_dim,
            'num_layers':           5,
            'pooling':              'avg',
            'mlp_normalization':    'layer',
            'residual':             True
        }

        gconv_decoder_kwargs = {
            'input_dim_obj':        gconv_dim * 2 + 512,
            'input_dim_pred':       gconv_dim * 2 + 512,
            'hidden_dim':           gconv_hidden_dim,
            'num_layers':           5,
            'pooling':              'avg',
            'mlp_normalization':    'layer',
            'residual':             True
        }

        gconv_conditioner_kwargs = {
            'input_dim_obj':        gconv_dim * 2 + 512,
            'input_dim_pred':       gconv_dim * 2 + 512,
            'hidden_dim':           gconv_hidden_dim,
            'num_layers':           5,
            'pooling':              'avg',
            'mlp_normalization':    'layer',
            'residual':             True
        }

        self.gconv_encoder = GraphTripleConvNet(**gconv_encoder_kwargs)
        self.gconv_decoder = GraphTripleConvNet(**gconv_decoder_kwargs)
        self.gconv_conditioner = GraphTripleConvNet(**gconv_conditioner_kwargs)

        self.box_embeddings.apply(_init_weights)
        self.mlp_mean_var.apply(_init_weights)
        self.mlp_mean.apply(_init_weights)
        self.mlp_var.apply(_init_weights)
        self.mlp_box.apply(_init_weights)

    def encoder(self, objs, obj_clip_embs, boxes, triples, rel_clip_embs):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        rel_embs = self.rel_embeddings_encoder(p)
        rel_embs = torch.cat([rel_clip_embs, rel_embs], dim=1)

        obj_embs = self.obj_embeddings_encoder(objs)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)
        box_embs = self.box_embeddings(boxes)
        obj_embs = torch.cat([obj_embs, box_embs], dim=1)

        all_embs, _ = self.gconv_encoder(obj_embs, rel_embs, edges)
        all_embs = self.mlp_mean_var(all_embs)
        mu = self.mlp_mean(all_embs)
        logvar = self.mlp_var(all_embs)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, objs, obj_clip_embs, z, triples, rel_clip_embs):
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        rel_embs = self.rel_embeddings_decoder(p)
        rel_embs = torch.cat([rel_clip_embs, rel_embs], dim=1)

        obj_embs = self.obj_embeddings_decoder(objs)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)
        obj_embs = torch.cat([obj_embs, z], dim=1)

        all_embs, _ = self.gconv_decoder(obj_embs, rel_embs, edges)
        box_pred = self.mlp_box(all_embs)

        pos_raw = torch.sigmoid(box_pred[..., :4])
        angle_raw = torch.tanh(box_pred[..., 4:5]) * np.pi
        image_mask = (objs == self.image_obj_idx)

        w = pos_raw[..., 2]
        h = pos_raw[..., 3]
        need_swap = (w < h)
        new_w = torch.where(need_swap, h, w)
        new_h = torch.where(need_swap, w, h)
        pos_raw = torch.stack([pos_raw[..., 0], pos_raw[..., 1], new_w, new_h], dim=-1)
        angle_raw = angle_raw + need_swap.unsqueeze(-1).to(angle_raw.dtype) * (np.pi / 2.0)
        angle_raw = torch.remainder(angle_raw + np.pi, 2 * np.pi) - np.pi

        fixed_pos_row = torch.tensor([0.5, 0.5, 1.0, 1.0],
                                     dtype=pos_raw.dtype, device=pos_raw.device).unsqueeze(0).expand(pos_raw.size(0), 4)
        pos = torch.where(image_mask.unsqueeze(-1), fixed_pos_row, pos_raw)
        angle = torch.where(image_mask.unsqueeze(-1), torch.zeros_like(angle_raw), angle_raw)

        box_pred = torch.cat([pos, angle], dim=-1)
        return box_pred

    def conditioner(self, objs, obj_clip_embs, z, triples, rel_clip_embs):
        s, p, o = triples.chunk(3, dim=1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]
        edges = torch.stack([s, o], dim=1)

        rel_embs = self.rel_embeddings_decoder(p)
        rel_embs = torch.cat([rel_clip_embs, rel_embs], dim=1)

        obj_embs = self.obj_embeddings_decoder(objs)
        obj_embs = torch.cat([obj_clip_embs, obj_embs], dim=1)
        obj_embs = torch.cat([obj_embs, z], dim=1)

        cond_embs, _ = self.gconv_conditioner(obj_embs, rel_embs, edges)
        cond_embs = self.cond_mlp(cond_embs)
        cond_embs = torch.unsqueeze(cond_embs, dim=0)

        uncond_embs = self.cond_mlp(obj_embs)
        uncond_embs = torch.unsqueeze(obj_embs, dim=0)

        return uncond_embs, cond_embs

    def forward(self, objs, obj_clip_embs, boxes, triples, rel_clip_embs):
        mu, logvar = self.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
        z = self.reparameterize(mu, logvar)
        box_pred = self.decoder(objs, obj_clip_embs, z, triples, rel_clip_embs)
        uncond_embs, cond_embs = self.conditioner(objs, obj_clip_embs, z, triples, rel_clip_embs)
        return mu, logvar, box_pred, cond_embs

    def sample(self, mean_est, cov_est, objs, obj_clip_embs, triples, rel_clip_embs, device):
        with torch.no_grad():
            z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float().to(device)
            box_pred = self.decoder(objs, obj_clip_embs, z, triples, rel_clip_embs)
            uncond_embs, cond_embs = self.conditioner(objs, obj_clip_embs, z, triples, rel_clip_embs)
            return box_pred, cond_embs

    def collect_data_statistics(self, train_loader, device):
        prev_mode = self.training
        self.eval()
        try:
            pbar = tqdm(train_loader, file=sys.stdout)
            mean_cat = []
            for idx, batch in enumerate(pbar):
                imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
                objs, triples, boxes = objs.to(device), triples.to(device), boxes.to(device)
                obj_clip_embs, rel_clip_embs = obj_clip_embs.to(device), rel_clip_embs.to(device)
                mean, logvar = self.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
                mean, logvar = mean.cpu().clone(), logvar.cpu().clone()
                mean = mean.data.cpu().clone()
                mean_cat.append(mean)
            mean_cat = torch.cat(mean_cat, dim=0)
            mean_est = torch.mean(mean_cat, dim=0, keepdim=True)
            cov_est = np.cov((mean_cat - mean_est).numpy().T)
            mean_est = mean_est[0]
        finally:
            self.train(prev_mode)
        return mean_est, cov_est