import math
import torch
import torch.nn as nn

from diffusers.models.attention_processor import Attention, AttnProcessor


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class MaskedSelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.attention_slice_size = 256
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, attention_masks):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        B, N, HC = q.shape
        H = self.heads
        C = HC // H
        q = q.view(B, N, H, C).permute(0, 2, 1, 3)
        k = k.view(B, N, H, C).permute(0, 2, 1, 3)
        v = v.view(B, N, H, C).permute(0, 2, 1, 3)
        q = q.reshape(B * H, N, C)
        k = k.reshape(B * H, N, C)
        v = v.reshape(B * H, N, C)

        k_t = k.transpose(1, 2)
        out = torch.empty((B * H, N, C), device=q.device, dtype=q.dtype)
        slice_size = int(self.attention_slice_size) if self.attention_slice_size is not None else N
        slice_size = max(1, min(slice_size, N))
        neg_inf = torch.finfo(q.dtype).min

        for start in range(0, N, slice_size):
            end = min(N, start + slice_size)
            q_chunk = q[:, start:end, :]
            sim = torch.bmm(q_chunk, k_t) * self.scale
            if attention_masks is not None:
                sim = sim.view(B, H, end - start, N)
                mask_chunk = attention_masks[:, :, start:end, :]
                sim = sim.masked_fill(mask_chunk <= 0.0, neg_inf)
                sim = sim.view(B * H, end - start, N)
            attn = sim.softmax(dim=-1, dtype=torch.float32).to(sim.dtype)
            out[:, start:end, :] = torch.bmm(attn, v)
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))
        return self.to_out(out)


class CustomCMALoraProcessor(nn.Module):
    def __init__(self, query_dim):
        super().__init__()
        self.is_cma = True
        self.linear = nn.Linear(768, query_dim)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, glu=True)
        self.cma = MaskedSelfAttention(query_dim)
        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.0)))

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, object_embeddings=None, object_attention_masks=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            n_visual = hidden_states.shape[1]
            if object_embeddings is not None and object_attention_masks is not None:
                N_mask = object_attention_masks.shape[-1]
                n_objs = object_embeddings.shape[1]
                HW_mask = N_mask - n_objs
                if n_visual == HW_mask:
                    B_hs = hidden_states.shape[0]
                    B_obj = object_embeddings.shape[0]
                    if B_obj != B_hs:
                        if B_obj > B_hs:
                            object_embeddings = object_embeddings[:B_hs]
                            object_attention_masks = object_attention_masks[:B_hs]
                        else:
                            reps = int(B_hs / B_obj)
                            object_embeddings = object_embeddings.repeat(reps, 1, 1)[:B_hs]
                            object_attention_masks = object_attention_masks.repeat(reps, 1, 1, 1)[:B_hs]
                    object_embeddings = object_embeddings.to(hidden_states.dtype)
                    object_embeddings = self.linear(object_embeddings)
                    attention_output = self.cma(self.norm1(torch.cat([hidden_states, object_embeddings], dim=1)), object_attention_masks)
                    hidden_states = hidden_states + torch.tanh(self.alpha_attn) * attention_output[:, 0:n_visual, :]
                    hidden_states = hidden_states + torch.tanh(self.alpha_dense) * self.ff(self.norm2(hidden_states))
                    hidden_states = self.norm3(hidden_states)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is not None:
            b_hs = hidden_states.shape[0]
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(0).expand(b_hs, -1, -1)
            elif encoder_hidden_states.dim() == 3 and encoder_hidden_states.shape[0] != b_hs:
                if encoder_hidden_states.shape[0] == 1:
                    encoder_hidden_states = encoder_hidden_states.expand(b_hs, -1, -1)
                elif encoder_hidden_states.shape[1] == b_hs:
                    encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
                else:
                    reps = math.ceil(b_hs / encoder_hidden_states.shape[0])
                    encoder_hidden_states = encoder_hidden_states.repeat(reps, 1, 1)[:b_hs]
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.zeros(out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.B.t()), self.A.t()) * (self.alpha / self.rank)


class NoOpAttnProcessor(AttnProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, object_embeddings=None, object_attention_masks=None):
        return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)


def register_attention_control_lora(unet, lora_rank=8):
    attn_procs = {}
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            query_dim = getattr(getattr(module, 'to_q', None), 'in_features', None)
            if query_dim is None:
                continue
            is_cross = ('attn2' in name)
            proc = CustomCMALoraProcessor(query_dim) if is_cross else NoOpAttnProcessor()
            if hasattr(module, 'set_processor'):
                module.set_processor(proc)
            else:
                module.processor = proc
            attn_procs[name + '.processor'] = proc
    return attn_procs


def get_lora_parameters(unet):
    params = []
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            for lin in [module.to_q, module.to_k, module.to_v, module.to_out[0]]:
                if hasattr(lin, 'lora_layer') and lin.lora_layer is not None:
                    params += list(lin.lora_layer.parameters())
    return params


def get_cma_small_parameters(unet):
    params = []
    for _, module in unet.named_modules():
        if isinstance(module, Attention):
            proc = getattr(module, 'processor', None)
            if isinstance(proc, CustomCMALoraProcessor):
                params += [proc.alpha_attn, proc.alpha_dense]
                params += list(proc.linear.parameters())
                params += list(proc.norm1.parameters())
                params += list(proc.norm2.parameters())
                params += list(proc.norm3.parameters())
                params += list(proc.ff.parameters())
                params += list(proc.cma.parameters())
    return params


def attach_lora_layers(unet, rank=8, scope='self'):
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            is_cross = ('attn2' in name)
            if scope == 'self' and is_cross:
                continue
            def wrap_linear(lin):
                if not hasattr(lin, 'lora_layer') or lin.lora_layer is None:
                    in_f = lin.in_features
                    out_f = lin.out_features
                    lin.lora_layer = LoRALinear(in_f, out_f, rank=rank, alpha=rank)
                    orig_forward = lin.forward
                    def forward_with_lora(x):
                        y = orig_forward(x)
                        y = y + lin.lora_layer(x)
                        return y
                    lin.forward = forward_with_lora
            wrap_linear(module.to_q)
            wrap_linear(module.to_k)
            wrap_linear(module.to_v)
            wrap_linear(module.to_out[0])
