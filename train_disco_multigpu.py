import os
import sys
import math
import json
import time
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from packaging import version
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import numpy as np

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs

from model.fusion import ObjectFusionTokenizer
from model.cond_vae import SceneVAEModel
from model.attention_lora import register_attention_control_lora, get_cma_small_parameters
from data import build_train_dataloader
from loss import VaeGaussCriterion, BoxL1Criterion


def parse_args():
    parser = argparse.ArgumentParser(description="Full fine-tuning script for DisCo.")
    parser.add_argument("--pretrained_diffusion_model_path", type=str, default='/inspire/hdd/global_user/yeziqi-240108100047/yxy/stable-diffusion-v1-5')
    parser.add_argument('--data_dir', type=str, default='/inspire/hdd/global_user/yeziqi-240108100047/yxy/vg')
    parser.add_argument('--output_dir', type=str, default="/inspire/hdd/global_user/yeziqi-240108100047/yxy/outputs")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    parser.add_argument('--dataloader_shuffle', type=bool, default=True)
    parser.add_argument("--tracker_project_name", type=str, default="text2image_fullft")
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_validation_images", type=int, default=8)

    parser.add_argument("--vae_loss_weight", type=float, default=0.1)
    parser.add_argument("--box_loss_weight", type=float, default=1.0)
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument('--embedding_dim', type=int, default=64)

    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    timestamp = time.strftime("%Y%m%d-%Hh%Mm%Ss", time.localtime())
    args.output_dir = os.path.join(args.output_dir, 'train', f'{args.tracker_project_name}-{timestamp}')
    return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(__name__, log_level="INFO")
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="tensorboard",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if args.seed is not None:
            set_seed(args.seed)

        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(f'{args.output_dir}/config.json', 'wt') as f:
                    json.dump(vars(args), f, indent=4)

        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_diffusion_model_path, subfolder="scheduler")
        self.scheduler = PNDMScheduler.from_pretrained(args.pretrained_diffusion_model_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_diffusion_model_path, subfolder="tokenizer")

        def deepspeed_zero_init_disabled_context_manager():
            deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
            if deepspeed_plugin is None:
                return []
            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_diffusion_model_path, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(args.pretrained_diffusion_model_path, subfolder="vae")

        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_diffusion_model_path, subfolder="unet")

        # Data
        self.train_dataloader, self.val_dataloader, _, self.vocab = build_train_dataloader(args, tokenizer=self.tokenizer)

        num_objs = len(self.vocab['object_idx_to_name'])
        num_rels = len(self.vocab['pred_idx_to_name'])
        self.sl_vae = SceneVAEModel(self.args, num_objs, num_rels)
        self.object_fusion_tokenizer = ObjectFusionTokenizer()

        # 预计算并缓存 box 统计量，避免验证阶段重复遍历训练集
        stats_path = os.path.join(args.output_dir, "box_stats.pt")
        if self.accelerator.is_main_process:
            self.logger.info("[init] collect_data_statistics:start")
            device_cpu = torch.device("cpu")
            self.box_mean_est, self.box_cov_est = self.sl_vae.collect_data_statistics(self.train_dataloader, device_cpu)
            mean_t = torch.as_tensor(self.box_mean_est)
            cov_t = torch.as_tensor(self.box_cov_est)
            torch.save({"mean": mean_t.cpu(), "cov": cov_t.cpu()}, stats_path)
            self.logger.info("[init] collect_data_statistics:done & saved")
        else:
            self.box_mean_est, self.box_cov_est = None, None

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        register_attention_control_lora(self.unet)
        for _, module in self.unet.named_modules():
            proc = getattr(module, 'processor', None)
            if hasattr(proc, 'alpha_attn'):
                proc.alpha_attn.data.copy_(torch.tensor(0.1))
            if hasattr(proc, 'alpha_dense'):
                proc.alpha_dense.data.copy_(torch.tensor(0.1))

        self.unet.train()
        self.sl_vae.train()
        self.object_fusion_tokenizer.train()

        # Criterion
        self.vae_criterion = VaeGaussCriterion()
        self.box_criterion = BoxL1Criterion()

        cma_params = get_cma_small_parameters(self.unet)
        cma_ids = {id(p) for p in cma_params}
        unet_base_params = [p for p in self.unet.parameters() if id(p) not in cma_ids]
        self.optimizer = torch.optim.AdamW([
            dict(params=unet_base_params, lr=args.learning_rate),
            dict(params=cma_params, lr=args.learning_rate * 10, weight_decay=0.0),
            dict(params=self.object_fusion_tokenizer.parameters(), lr=args.learning_rate),
            dict(params=self.sl_vae.parameters(), lr=args.learning_rate),
        ], lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)


        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=args.max_train_steps * self.accelerator.num_processes,
        )

        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.sl_vae, self.object_fusion_tokenizer = self.accelerator.prepare(
            self.unet,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.sl_vae,
            self.object_fusion_tokenizer,
        )
        # 同步并加载缓存的 box 统计量到各进程
        self.accelerator.wait_for_everyone()
        stats = torch.load(stats_path, map_location="cpu")
        self.box_mean_est = stats["mean"].numpy()
        self.box_cov_est = stats["cov"].numpy()

        if args.use_ema:
            unet_native = self.accelerator.unwrap_model(self.unet)
            self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet_native.config)
            self.ema_unet.to(self.accelerator.device)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
            args.mixed_precision = self.accelerator.mixed_precision
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
            args.mixed_precision = self.accelerator.mixed_precision

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        self.progress_bar = tqdm(range(0, self.args.max_train_steps), initial=0, desc="Steps", disable=not self.accelerator.is_local_main_process)

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(args))
            self.accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def start(self):
        self.logger.info('  Global configuration as follows:')
        for key, val in vars(self.args).items():
            self.logger.info("  {:28} {}".format(key, val))

        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.logger.info("\n")
        self.logger.info(f"  Running training:")
        self.logger.info(f"  Num Iterations = {len(self.train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        self.train()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.args.use_ema:
                self.ema_unet.copy_to(self.unet.parameters())
            self.unet = self.accelerator.unwrap_model(self.unet)
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.args.pretrained_diffusion_model_path,
                text_encoder=self.text_encoder,
                vae=self.vae,
                unet=self.unet,
            )
            pipeline.save_pretrained(self.args.output_dir)
            self.sl_vae = self.accelerator.unwrap_model(self.sl_vae)
            self.object_fusion_tokenizer = self.accelerator.unwrap_model(self.object_fusion_tokenizer)
            torch.save(self.sl_vae.state_dict(), os.path.join(self.args.output_dir, "sl_vae.pt"))
            torch.save(self.object_fusion_tokenizer.state_dict(), os.path.join(self.args.output_dir, "fusion.pt"))

        self.accelerator.end_training()

    def train(self):
        self.global_step = 0
        for epoch in range(0, self.args.num_train_epochs):
            self.logger.info(f"[epoch={epoch}] train_one_epoch:start")
            self.train_one_epoch(epoch)
            self.logger.info(f"[epoch={epoch}] train_one_epoch:end")

    def train_one_epoch(self, epoch):
        log_loss = 0.0
        log_box_loss = 0.0
        log_vae_loss = 0.0
        log_diff_loss = 0.0
        if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(epoch)
            self.logger.info(f"[epoch={epoch}] set train sampler epoch")
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.unet):
                imgs, objs, obj_clip_embs, layout, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch

                latents = self.vae.encode(imgs.to(self.weight_dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                bsz = latents.shape[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                mu, logvar, layout_pred, semantics_embs = self.sl_vae(objs, obj_clip_embs, layout, triples, rel_clip_embs)
                object_embeddings, meta_data = self.object_fusion_tokenizer(layout, semantics_embs.squeeze(0), obj_to_img)

                cross_attention_kwargs = {}
                cross_attention_kwargs['object_embeddings'] = object_embeddings
                cross_attention_kwargs['object_attention_masks'] = meta_data['object_attention_masks']

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = self.text_encoder(caption, return_dict=False)[0]
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]

                vae_loss = self.vae_criterion(mu, logvar)
                box_loss = self.box_criterion(layout_pred, layout)
                diff_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = box_loss * self.args.box_loss_weight + vae_loss * self.args.vae_loss_weight + diff_loss * self.args.diff_loss_weight

                log_loss += self.gather_loss(loss)
                log_box_loss += self.gather_loss(box_loss)
                log_vae_loss += self.gather_loss(vae_loss)
                log_diff_loss += self.gather_loss(diff_loss)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)

                self.lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                if self.args.use_ema:
                    self.ema_unet.step(self.unet.parameters())
                self.progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": log_loss}, step=self.global_step)
                self.accelerator.log({"box_loss": log_box_loss}, step=self.global_step)
                self.accelerator.log({"vae_loss": log_vae_loss}, step=self.global_step)
                self.accelerator.log({"diff_loss": log_diff_loss}, step=self.global_step)
                self.accelerator.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)
                log_loss = 0.0
                log_box_loss = 0.0
                log_vae_loss = 0.0
                log_diff_loss = 0.0

                logs = {"step_loss": '%.4f' % loss.detach().item(), "lr": '%.2e' % self.lr_scheduler.get_last_lr()[0]}
                self.progress_bar.set_postfix(**logs)

                if self.global_step % self.args.checkpointing_steps == 0:
                    self.logger.info(f"[epoch={epoch}][step={step}] validation:start")
                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            self.log_validation(self.global_step)
                    self.accelerator.wait_for_everyone()
                    self.logger.info(f"[epoch={epoch}][step={step}] validation:end")

            if self.global_step >= self.args.max_train_steps:
                break

    def gather_loss(self, loss):
        avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
        loss = avg_loss.item() / self.args.gradient_accumulation_steps
        return loss

    @torch.no_grad()
    def log_validation(self, step):
        sl_vae_native = self.accelerator.unwrap_model(self.sl_vae)
        unet_native = self.accelerator.unwrap_model(self.unet)
        max_val_images = self.args.num_validation_images
        pbar = tqdm(self.val_dataloader, total=max_val_images, file=sys.stdout)
        pil_images = []
        for idx, batch in enumerate(pbar):
            if idx >= max_val_images:
                break
            imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = batch
            objs = objs.to(self.accelerator.device)
            triples = triples.to(self.accelerator.device)
            obj_clip_embs = obj_clip_embs.to(self.accelerator.device)
            rel_clip_embs = rel_clip_embs.to(self.accelerator.device)
            caption = caption.to(self.accelerator.device)

            layout_preds, semantics_embs = sl_vae_native.sample(self.box_mean_est, self.box_cov_est, objs, obj_clip_embs, triples, rel_clip_embs, self.accelerator.device)
            idxs = torch.nonzero(obj_to_img == 0, as_tuple=False).squeeze(1)
            boxes0 = layout_preds[idxs]
            objs0 = objs[idxs]
            sem0 = semantics_embs.squeeze(0)[idxs]
            layout_image = self.layout_visualization([int(x) for x in objs0.detach().cpu().tolist()], boxes0)
            obj_to_img0 = torch.zeros_like(idxs)
            object_embeddings, meta_data = self.object_fusion_tokenizer(boxes0, sem0, obj_to_img0)
            obj_emb_1 = object_embeddings[:1]
            att_mask_1 = meta_data['object_attention_masks'][:1]
            cross_attention_kwargs = {}
            cross_attention_kwargs['object_embeddings'] = torch.cat([obj_emb_1, obj_emb_1])
            cross_attention_kwargs['object_attention_masks'] = torch.cat([att_mask_1, att_mask_1])

            cond_embeddings = self.text_encoder(caption[:1])[0]
            max_length = caption.shape[-1]
            uncond_input = self.tokenizer([""] , padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.accelerator.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

            self.scheduler.set_timesteps(self.args.num_inference_steps)
            latent_size = (1, unet_native.config.in_channels, self.args.resolution // 8, self.args.resolution // 8)
            if self.args.seed is not None:
                gen = torch.Generator(device=self.accelerator.device)
                gen.manual_seed(int(self.args.seed))
                latent = torch.randn(latent_size, generator=gen, device=self.accelerator.device)
            else:
                latent = torch.randn(latent_size, device=self.accelerator.device)
            latent = latent * self.scheduler.init_noise_sigma

            image = None
            for t in tqdm(self.scheduler.timesteps):
                latent_model_input = torch.cat([latent] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
                noise_pred = unet_native(latent_model_input, t, text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent = self.scheduler.step(noise_pred, t, latent).prev_sample
                scaled_latents = 1.0 / 0.18215 * latent.clone()
                image = self.vae.decode(scaled_latents.to(self.weight_dtype)).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).round().astype("uint8")

            grid = Image.new('RGB', size=(self.args.resolution * 2, self.args.resolution))
            grid.paste(Image.fromarray(image.squeeze(0)), box=(0, 0))
            grid.paste(layout_image, box=(self.args.resolution, 0))
            pil_images.append(grid)

        if self.accelerator.is_main_process:
            save_root = os.path.join(self.args.output_dir, "validation", f"step_{int(step):06d}")
            os.makedirs(save_root, exist_ok=True)
            for index, pil_image in enumerate(pil_images):
                pil_image.save(os.path.join(save_root, f"{index:04d}.png"))
            for tracker in self.accelerator.trackers:
                np_images = np.stack([np.asarray(img) for img in pil_images])
                tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")


    def layout_visualization(self, objs, boxes, images=None):
        color = list(np.random.choice(range(256), size=(len(boxes), 3)))
        if images is None:
            layout = Image.new('RGB', size=(self.args.resolution, self.args.resolution))
        else:
            layout = images.copy()
        draw_layout = ImageDraw.Draw(layout)

        for i, (obj, box) in enumerate(zip(objs, boxes)):
            obj_text = self.vocab['object_idx_to_name'][obj]
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()

            if len(box) == 4:
                x0, y0, x1, y1 = (box * self.args.resolution).tolist()
                if x1 < x0 or y1 < y0:
                    continue
                draw_layout.rectangle([x0, y0, x1, y1], outline=tuple(color[i]))
                draw_layout.text(xy=(x0, y0), text=obj_text, fill=tuple(color[i]))
            elif len(box) == 5:
                cx, cy, w, h, a = box
                cx, cy, w, h = cx * self.args.resolution, cy * self.args.resolution, w * self.args.resolution, h * self.args.resolution
                theta = float(a)
                dx, dy = w / 2.0, h / 2.0
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                corners = [(-dx, -dy), (-dx,  dy), ( dx,  dy), ( dx, -dy)]
                points = [(cx + cos_t*px - sin_t*py, cy + sin_t*px + cos_t*py) for px, py in corners]
                draw_layout.polygon(points, outline=tuple(color[i]))
                draw_layout.text(xy=(cx, cy), text=obj_text, fill=tuple(color[i]))

        return layout

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.start()

## CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_disco_multigpu.py --resolution=512 --batch_size=8 --gradient_accumulation_steps=2 --gradient_checkpointing --max_train_steps=40000 --learning_rate=1e-04 --lr_scheduler="linear" --checkpointing_steps=4000 --mixed_precision="fp16"