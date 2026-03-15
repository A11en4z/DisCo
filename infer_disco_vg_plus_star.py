import argparse
import json
import os
import sys
import contextlib
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

from data_lora import build_train_dataloader
from model.cond_vae_lora import SceneVAEModel
from model.fusion import ObjectFusionTokenizer
from model.attention_lora import register_attention_control_lora, attach_lora_layers


def parse_args():
    parser = argparse.ArgumentParser(description="DisCo inference for VG+Star test set.")
    parser.add_argument("--model_dir", type=str, default="/gz-data/outputs/train/q2_4w_base_diorplusstarair-20251231-17h46m42s")
    parser.add_argument("--data_dir", type=str, default="/gz-data/vg_plus_star")
    parser.add_argument("--output_dir", type=str, default="/gz-data/outputs/train_gen_disco_gtbox")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--layout_mode", type=str, choices=["sample", "gtbox"], default="sample")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--disable_relations", action="store_true")
    parser.add_argument("--cond_mode", type=str, choices=["both", "semantic_only", "layout_only"], default="both")
    return parser.parse_args()


def load_config(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def to_args(config, overrides):
    data = dict(config)
    data["data_dir"] = overrides.data_dir
    data["dataloader_num_workers"] = overrides.num_workers
    data["batch_size"] = overrides.batch_size
    data["output_dir"] = overrides.output_dir
    data["split"] = overrides.split
    data["layout_mode"] = overrides.layout_mode
    if overrides.num_inference_steps is not None:
        data["num_inference_steps"] = overrides.num_inference_steps
    if overrides.guidance_scale is not None:
        data["guidance_scale"] = overrides.guidance_scale
    if overrides.seed is not None:
        data["seed"] = overrides.seed
    if overrides.disable_relations:
        data["disable_relations"] = True
    data["cond_mode"] = overrides.cond_mode
    return SimpleNamespace(**data)


def get_weight_dtype(mixed_precision, device):
    if device.type == "cpu":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def build_relation_free_triples(objs, obj_to_img, rel_clip_embs, image_obj_idx, vocab):
    num_objs = int(objs.size(0))
    pred_idx = vocab["pred_name_to_idx"].get("__in_image__", 0)
    if num_objs == 0:
        triples = torch.zeros((1, 3), device=objs.device, dtype=torch.long)
        rel_embs = torch.zeros((1, rel_clip_embs.size(1)), device=rel_clip_embs.device, dtype=rel_clip_embs.dtype)
        if obj_to_img is not None and obj_to_img.numel() > 0:
            triple_to_img = torch.zeros((1,), device=obj_to_img.device, dtype=obj_to_img.dtype)
        else:
            triple_to_img = torch.zeros((1,), device=objs.device, dtype=torch.long)
        return triples, rel_embs, triple_to_img
    image_ids = torch.unique(obj_to_img)
    image_obj_map = {}
    for img_id in image_ids:
        mask = obj_to_img == img_id
        image_mask = mask & (objs == image_obj_idx)
        if image_mask.any():
            target = int(image_mask.nonzero(as_tuple=False)[0].item())
        else:
            target = int(mask.nonzero(as_tuple=False)[0].item())
        image_obj_map[int(img_id.item())] = target
    target_idx = torch.empty((num_objs,), device=objs.device, dtype=torch.long)
    for i in range(num_objs):
        img_id = int(obj_to_img[i].item())
        target_idx[i] = image_obj_map.get(img_id, i)
    s = torch.arange(num_objs, device=objs.device, dtype=torch.long)
    p = torch.full((num_objs,), pred_idx, device=objs.device, dtype=torch.long)
    o = target_idx
    triples = torch.stack([s, p, o], dim=1)
    rel_embs = torch.zeros((num_objs, rel_clip_embs.size(1)), device=rel_clip_embs.device, dtype=rel_clip_embs.dtype)
    triple_to_img = obj_to_img[s]
    return triples, rel_embs, triple_to_img


def save_scene_graph(out_path, image_path, objs_i, boxes_pred_i, boxes_gt_i, triples_i, vocab):
    objects = []
    for local_id, obj_idx in enumerate(objs_i.tolist()):
        obj_name = vocab["object_idx_to_name"][int(obj_idx)]
        box_pred = [float(x) for x in boxes_pred_i[local_id].tolist()]
        box_gt = [float(x) for x in boxes_gt_i[local_id].tolist()]
        objects.append(
            {
                "local_id": int(local_id),
                "global_id": int(local_id),
                "name": obj_name,
                "box_pred": box_pred,
                "box_gt": box_gt,
            }
        )
    relationships = []
    for triple in triples_i.tolist():
        s, p, o = triple
        relationships.append(
            {
                "subject": int(s),
                "predicate": vocab["pred_idx_to_name"][int(p)],
                "object": int(o),
                "subject_global": int(s),
                "object_global": int(o),
            }
        )
    payload = {
        "image_path": image_path,
        "objects": objects,
        "relationships": relationships,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_unet_custom_weights(unet, model_dir):
    unet_dir = os.path.join(model_dir, "unet")
    safetensors_path = os.path.join(unet_dir, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device="cpu")
        except Exception:
            from eval.lora_diffusion.safe_open import safe_open
            safetensors_obj = safe_open(safetensors_path, framework="pt", device="cpu")
            state_dict = {k: safetensors_obj.get_tensor(k) for k in safetensors_obj.keys()}
        unet.load_state_dict(state_dict, strict=False)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        unet.load_state_dict(state_dict, strict=False)


def select_dataloader(args, tokenizer):
    train_dataloader, val_dataloader, test_dataloader, vocab = build_train_dataloader(
        args, tokenizer=tokenizer, with_clip_embs=True
    )
    if args.split == "train":
        dataloader = train_dataloader
    elif args.split == "val":
        dataloader = val_dataloader
    else:
        dataloader = test_dataloader
    return dataloader, vocab


def main():
    overrides = parse_args()
    config = load_config(overrides.model_dir)
    args = to_args(config, overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(args.mixed_precision, device)
    layout_dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    if args.layout_mode == "gtbox":
        images_output_dir = os.path.join(args.output_dir, f"gtbox_{args.split}")
        scene_graph_dir = os.path.join(args.output_dir, f"scene_graph_gtbox_{args.split}")
    else:
        images_output_dir = args.output_dir
        scene_graph_dir = os.path.join(args.output_dir, "scene_graph")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(scene_graph_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(overrides.model_dir, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(overrides.model_dir, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(overrides.model_dir, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(overrides.model_dir, subfolder="unet")
    register_attention_control_lora(unet, lora_rank=getattr(args, "lora_rank", 8))
    attach_lora_layers(unet, rank=getattr(args, "lora_rank", 8), scope="self")
    load_unet_custom_weights(unet, overrides.model_dir)
    scheduler = PNDMScheduler.from_pretrained(overrides.model_dir, subfolder="scheduler")

    text_encoder.to(device, dtype=weight_dtype).eval()
    vae.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype).eval()
    enable_xformers = getattr(args, "enable_xformers_memory_efficient_attention", True)
    if enable_xformers and is_xformers_available():
        if hasattr(unet, "enable_xformers_memory_efficient_attention"):
            unet.enable_xformers_memory_efficient_attention()
    else:
        attention_slice_size = int(getattr(args, "attention_slice_size", 128))
        if hasattr(unet, "attn_processors"):
            for proc in unet.attn_processors.values():
                if hasattr(proc, "attention_slice_size"):
                    proc.attention_slice_size = attention_slice_size
                if hasattr(proc, "cma") and hasattr(proc.cma, "attention_slice_size"):
                    proc.cma.attention_slice_size = attention_slice_size

    dataloader, vocab = select_dataloader(args, tokenizer)

    image_obj_idx = vocab["object_name_to_idx"].get("__image__", 0)
    pure_sd1_5 = bool(getattr(args, "pure_sd1_5", False))
    disable_relations = bool(getattr(args, "disable_relations", False))

    sl_vae = None
    object_fusion_tokenizer = None
    if not pure_sd1_5:
        sl_vae = SceneVAEModel(args, len(vocab["object_idx_to_name"]), len(vocab["pred_idx_to_name"]), image_obj_idx=image_obj_idx)
        sl_vae.load_state_dict(torch.load(os.path.join(overrides.model_dir, "sl_vae.pt"), map_location="cpu"))
        sl_vae.to(device, dtype=layout_dtype).eval()
        object_fusion_tokenizer = ObjectFusionTokenizer().to(device, dtype=layout_dtype).eval()
        object_fusion_tokenizer.load_state_dict(torch.load(os.path.join(overrides.model_dir, "fusion.pt"), map_location="cpu"))

        stats = torch.load(os.path.join(overrides.model_dir, "box_stats.pt"), map_location="cpu")
        mean_est = stats["mean"].cpu().numpy()
        cov_est = stats["cov"].cpu().numpy()
    else:
        mean_est, cov_est = None, None

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
    else:
        generator = None
    use_amp = device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16)
    amp_ctx = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

    total = 0
    with torch.inference_mode():
        for batch in tqdm(
            dataloader,
            desc=f"Generating[{args.split}]",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
            disable=False,
        ):
            if overrides.max_samples is not None and total >= overrides.max_samples:
                break
            imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, captions = batch
            batch_size = int(captions.shape[0])

            objs = objs.to(device)
            obj_clip_embs = obj_clip_embs.to(device, dtype=layout_dtype)
            boxes = boxes.to(device, dtype=layout_dtype)
            triples = triples.to(device)
            rel_clip_embs = rel_clip_embs.to(device, dtype=layout_dtype)
            obj_to_img = obj_to_img.to(device)
            triple_to_img = triple_to_img.to(device)
            captions = captions.to(device)

            if disable_relations and not pure_sd1_5:
                triples, rel_clip_embs, triple_to_img = build_relation_free_triples(
                    objs, obj_to_img, rel_clip_embs, image_obj_idx, vocab
                )

            if not pure_sd1_5:
                if args.layout_mode == "gtbox":
                    mu, _ = sl_vae.encoder(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
                    z = mu
                    _, semantics_embs = sl_vae.conditioner(objs, obj_clip_embs, z, triples, rel_clip_embs)
                    layout_preds = boxes
                else:
                    layout_preds, semantics_embs = sl_vae.sample(mean_est, cov_est, objs, obj_clip_embs, triples, rel_clip_embs, device)
                sem_in = semantics_embs.squeeze(0)
                lay_in = layout_preds.detach()
                cond_mode = str(getattr(args, "cond_mode", "both"))
                if cond_mode == "semantic_only":
                    fixed = torch.tensor([0.5, 0.5, 1.0, 1.0, 0.0], device=lay_in.device, dtype=lay_in.dtype)
                    lay_in = torch.zeros_like(lay_in) + fixed
                    object_embeddings, meta_data = object_fusion_tokenizer(lay_in, sem_in, obj_to_img)
                    if "object_attention_masks" in meta_data:
                        meta_data["object_attention_masks"] = torch.ones_like(meta_data["object_attention_masks"])
                elif cond_mode == "layout_only":
                    sem_zero = torch.zeros_like(sem_in)
                    object_embeddings, meta_data = object_fusion_tokenizer(lay_in, sem_zero, obj_to_img)
                else:
                    object_embeddings, meta_data = object_fusion_tokenizer(lay_in, sem_in, obj_to_img)
                object_embeddings = object_embeddings.to(device, dtype=weight_dtype)
                object_attention_masks = meta_data["object_attention_masks"].to(device, dtype=weight_dtype)
                cross_attention_kwargs = {
                    "object_embeddings": object_embeddings,
                    "object_attention_masks": object_attention_masks,
                }
            else:
                layout_preds = boxes
                cross_attention_kwargs = {}

            uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=captions.shape[-1], return_tensors="pt")
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
            cond_embeddings = text_encoder(captions)[0]
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

            scheduler.set_timesteps(args.num_inference_steps)
            latent_size = (batch_size, unet.config.in_channels, args.resolution // 8, args.resolution // 8)
            if generator is not None:
                latents = torch.randn(latent_size, generator=generator, device=device, dtype=weight_dtype)
            else:
                latents = torch.randn(latent_size, device=device, dtype=weight_dtype)
            latents = latents * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
                with amp_ctx(dtype=weight_dtype):
                    noise_pred = unet(latent_model_input, t, text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            scaled_latents = 1.0 / 0.18215 * latents
            with amp_ctx(dtype=weight_dtype):
                images = vae.decode(scaled_latents).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.detach().to(torch.float32).cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")

            for i in range(batch_size):
                image_path = img_paths[i] if isinstance(img_paths[i], str) else img_paths[i].decode("utf-8")
                basename = os.path.splitext(os.path.basename(image_path))[0]
                gen_path = os.path.join(images_output_dir, f"{basename}_gen.jpg")
                Image.fromarray(images[i]).save(gen_path)

                mask_obj = obj_to_img == i
                mask_rel = triple_to_img == i
                objs_i = objs[mask_obj].detach().cpu()
                boxes_gt_i = boxes[mask_obj].detach().to(torch.float32).cpu()
                layout_i = layout_preds[mask_obj].detach().to(torch.float32).cpu()
                triples_i = triples[mask_rel].detach().cpu()

                global_obj_indices = mask_obj.nonzero(as_tuple=False).view(-1).tolist()
                global_to_local = {int(g): idx for idx, g in enumerate(global_obj_indices)}
                if triples_i.numel() > 0:
                    triples_i = triples_i.clone()
                    triples_i[:, 0] = torch.tensor([global_to_local[int(x)] for x in triples_i[:, 0].tolist()], dtype=triples_i.dtype)
                    triples_i[:, 2] = torch.tensor([global_to_local[int(x)] for x in triples_i[:, 2].tolist()], dtype=triples_i.dtype)

                graph_path = os.path.join(scene_graph_dir, f"{basename}_graph.json")
                save_scene_graph(graph_path, image_path, objs_i, layout_i, boxes_gt_i, triples_i, vocab)

            total += batch_size


if __name__ == "__main__":
    main()