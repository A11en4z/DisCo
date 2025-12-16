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
from model.cond_vae_lora import SceneVAEModel
from model.attention_lora import register_attention_control_lora, get_lora_parameters, get_cma_small_parameters, attach_lora_layers
from data_lora import build_train_dataloader
from loss import VaeGaussCriterion, BoxL1Criterion
from loss import SameClassOverlapCriterion, CategoryPriorCriterion


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning script for DisCo.")
    parser.add_argument("--pretrained_diffusion_model_path", type=str, default='/gz-data/stable-diffusion-v1-5')
    parser.add_argument('--data_dir', type=str, default='/gz-data/vg')
    parser.add_argument('--output_dir', type=str, default="/gz-data/outputs/dq")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument('--dataloader_num_workers', type=int, default=8)
    parser.add_argument('--dataloader_shuffle', type=bool, default=True)
    parser.add_argument("--tracker_project_name", type=str, default="text2image_lora")
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
    parser.add_argument("--angle_loss_weight", type=float, default=1.0)
    parser.add_argument("--aux_loss_warmup_steps", type=int, default=10000)

    parser.add_argument("--vae_loss_weight", type=float, default=0.1)
    parser.add_argument("--box_loss_weight", type=float, default=1.0)
    parser.add_argument("--diff_loss_weight", type=float, default=1.0)
    parser.add_argument('--embedding_dim', type=int, default=64)

    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--freeze_unet', action='store_true')
    parser.add_argument('--unet_base_lr', type=float, default=5e-6)

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
        image_obj_idx = self.vocab['object_name_to_idx'].get('__image__', 0)
        self.sl_vae = SceneVAEModel(self.args, num_objs, num_rels, image_obj_idx=image_obj_idx)
        self.object_fusion_tokenizer = ObjectFusionTokenizer()

        # 加载数据集统计量 box_stats.pt（不再训练时生成）
        stats_path = os.path.join(args.data_dir, "box_stats.pt")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"box_stats.pt not found in data_dir: {stats_path}")

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Register LoRA + CMA processors; freeze UNet base if requested
        if args.freeze_unet:
            self.unet.requires_grad_(False)
        register_attention_control_lora(self.unet, lora_rank=args.lora_rank)
        attach_lora_layers(self.unet, rank=args.lora_rank, scope='self')
        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.unet.train()
        self.sl_vae.train()
        self.object_fusion_tokenizer.train()

        # Criterion
        self.vae_criterion = VaeGaussCriterion()
        self.box_criterion = BoxL1Criterion(angle_weight=self.args.angle_loss_weight)
        self.same_class_criterion = SameClassOverlapCriterion()
        self.category_prior_criterion = CategoryPriorCriterion()
        self.class_weights = None
        self.same_overlap_weight = 0.0
        self.category_prior_weight = 0.0
        self.use_angle_prior = False

        # Optimizer param groups: LoRA, CMA small, fusion, sl_vae
        lora_params = get_lora_parameters(self.unet)
        cma_params = get_cma_small_parameters(self.unet)
        fusion_params = list(self.object_fusion_tokenizer.parameters())
        sl_vae_params = list(self.sl_vae.parameters())
        if not args.freeze_unet:
            all_unet_params = list(self.unet.parameters())
            exclude_ids = {id(p) for p in list(lora_params) + list(cma_params)}
            unet_base_params = [p for p in all_unet_params if id(p) not in exclude_ids]
        else:
            unet_base_params = []

        param_groups = [
            dict(params=lora_params, lr=args.learning_rate),
            dict(params=cma_params, lr=args.learning_rate),
            dict(params=fusion_params, lr=args.learning_rate * 10),
            dict(params=sl_vae_params, lr=args.learning_rate * 10),
        ]
        if len(unet_base_params) > 0:
            param_groups.insert(2, dict(params=unet_base_params, lr=args.unet_base_lr))

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

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
        # 同步并加载数据集提供的 box 统计量到各进程
        self.accelerator.wait_for_everyone()
        stats = torch.load(stats_path, map_location="cpu")
        self.box_mean_est = stats["mean"].numpy()
        self.box_cov_est = stats["cov"].numpy()
        priors_path = os.path.join(args.data_dir, "box_priors.pt")
        if not os.path.exists(priors_path):
            raise FileNotFoundError(f"box_priors.pt not found in data_dir: {priors_path}")
        self.box_priors = torch.load(priors_path, map_location="cpu")

        if args.use_ema:
            self.ema_unet = UNet2DConditionModel.from_pretrained(args.pretrained_diffusion_model_path, subfolder="unet")
            register_attention_control_lora(self.ema_unet, lora_rank=args.lora_rank)
            attach_lora_layers(self.ema_unet, rank=args.lora_rank, scope='self')
            self.ema_unet = EMAModel(self.ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.ema_unet.config)
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
        log_overlap_loss = 0.0
        log_prior_loss = 0.0
        log_overlap_loss_w = 0.0
        log_prior_loss_w = 0.0
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

                with torch.cuda.amp.autocast(enabled=False):
                    mu, logvar, layout_pred, semantics_embs = self.sl_vae(objs, obj_clip_embs, layout, triples, rel_clip_embs)
                object_embeddings, meta_data = self.object_fusion_tokenizer(layout, semantics_embs.squeeze(0), obj_to_img)

                cross_attention_kwargs = {}
                cross_attention_kwargs['object_embeddings'] = object_embeddings
                cross_attention_kwargs['object_attention_masks'] = meta_data['object_attention_masks']

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = self.text_encoder(caption, return_dict=False)[0]
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]

                vae_loss = self.vae_criterion(mu.float(), logvar.float())
                box_loss = self.box_criterion(layout_pred.float(), layout.float(), objs=objs.to(self.accelerator.device), class_weights=self.class_weights)
                diff_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                aux_enable = self.global_step >= self.args.aux_loss_warmup_steps
                if aux_enable:
                    if self.class_weights is None:
                        self.class_weights = torch.ones((len(self.vocab['object_idx_to_name']),), device=self.accelerator.device)
                    image_idx = self.vocab['object_name_to_idx'].get('__image__', None)
                    overlap_loss = self.same_class_criterion(
                        layout_pred,
                        objs.to(self.accelerator.device),
                        obj_to_img.to(self.accelerator.device),
                        image_idx=image_idx,
                        class_weights=self.class_weights,
                        mode='oriented',
                        theta_eps=0.087,
                    )
                    if not torch.isfinite(overlap_loss):
                        self.logger.warning(f"[step={self.global_step}] overlap_loss is non-finite; zeroed")
                        overlap_loss = torch.tensor(0.0, device=self.accelerator.device)
                    priors_device = {}
                    for k, v in self.box_priors.items():
                        priors_device[int(k)] = {"mean": v["mean"].to(self.accelerator.device), "cov": v["cov"].to(self.accelerator.device)}
                    prior_loss = self.category_prior_criterion(layout_pred, objs.to(self.accelerator.device), priors_device, use_angle_prior=self.use_angle_prior, class_weights=self.class_weights)
                    if not torch.isfinite(prior_loss):
                        self.logger.warning(f"[step={self.global_step}] prior_loss is non-finite; zeroed")
                        prior_loss = torch.tensor(0.0, device=self.accelerator.device)
                    aux_term = self.same_overlap_weight * overlap_loss + self.category_prior_weight * prior_loss
                else:
                    overlap_loss = torch.tensor(0.0, device=self.accelerator.device)
                    prior_loss = torch.tensor(0.0, device=self.accelerator.device)
                    aux_term = torch.tensor(0.0, device=self.accelerator.device)
                loss = box_loss * self.args.box_loss_weight + vae_loss * self.args.vae_loss_weight + diff_loss * self.args.diff_loss_weight + aux_term

                if not torch.isfinite(loss):
                    self.logger.warning(f"[step={self.global_step}] total loss is non-finite; skip step. box={box_loss.item():.4f}, vae={vae_loss.item():.4f}, diff={diff_loss.item():.4f}, overlap={overlap_loss.item():.4f}, prior={prior_loss.item():.4f}")
                    self.optimizer.zero_grad()
                    continue

                log_loss += self.gather_loss(loss)
                log_box_loss += self.gather_loss(box_loss)
                log_vae_loss += self.gather_loss(vae_loss)
                log_diff_loss += self.gather_loss(diff_loss)
                log_overlap_loss += self.gather_loss(overlap_loss)
                log_prior_loss += self.gather_loss(prior_loss)
                log_overlap_loss_w += self.gather_loss(self.same_overlap_weight * overlap_loss)
                log_prior_loss_w += self.gather_loss(self.category_prior_weight * prior_loss)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    all_params = list(self.unet.parameters()) + list(self.sl_vae.parameters()) + list(self.object_fusion_tokenizer.parameters())
                    self.accelerator.clip_grad_norm_(all_params, self.args.max_grad_norm)

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
                self.accelerator.log({"overlap_loss": log_overlap_loss}, step=self.global_step)
                self.accelerator.log({"prior_loss": log_prior_loss}, step=self.global_step)
                self.accelerator.log({"overlap_loss_weighted": log_overlap_loss_w}, step=self.global_step)
                self.accelerator.log({"prior_loss_weighted": log_prior_loss_w}, step=self.global_step)
                self.accelerator.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)
                log_loss = 0.0
                log_box_loss = 0.0
                log_vae_loss = 0.0
                log_diff_loss = 0.0
                log_overlap_loss = 0.0
                log_prior_loss = 0.0
                log_overlap_loss_w = 0.0
                log_prior_loss_w = 0.0

                logs = {"step_loss": '%.4f' % loss.detach().item(), "lr": '%.2e' % self.lr_scheduler.get_last_lr()[0]}
                self.progress_bar.set_postfix(**logs)

                if self.global_step % self.args.checkpointing_steps == 0:
                    self.logger.info(f"[epoch={epoch}][step={step}] validation:start")
                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            self.log_validation(self.global_step, batch)
                    self.accelerator.wait_for_everyone()
                    self.logger.info(f"[epoch={epoch}][step={step}] validation:end")

            if self.global_step >= self.args.max_train_steps:
                break

    def gather_loss(self, loss):
        avg_loss = self.accelerator.gather(loss.repeat(self.args.batch_size)).mean()
        loss = avg_loss.item() / self.args.gradient_accumulation_steps
        return loss

    @torch.no_grad()
    def log_validation(self, step, ref_batch):
        sl_vae_native = self.accelerator.unwrap_model(self.sl_vae)
        unet_native = self.accelerator.unwrap_model(self.unet)
        imgs, objs, obj_clip_embs, boxes, triples, rel_clip_embs, obj_to_img, triple_to_img, img_paths, caption = ref_batch
        objs, triples = objs.to(self.accelerator.device), triples.to(self.accelerator.device)
        obj_clip_embs, rel_clip_embs = obj_clip_embs.to(self.accelerator.device), rel_clip_embs.to(self.accelerator.device)
        boxes = boxes.to(self.accelerator.device)
        caption = caption.to(self.accelerator.device)
        mu, logvar, layout_preds, semantics_embs = sl_vae_native(objs, obj_clip_embs, boxes, triples, rel_clip_embs)
        object_embeddings, meta_data = self.object_fusion_tokenizer(layout_preds.detach(), semantics_embs.squeeze(0), obj_to_img)
        cross_attention_kwargs = {}
        cross_attention_kwargs['object_embeddings'] = torch.cat([object_embeddings, object_embeddings])
        cross_attention_kwargs['object_attention_masks'] = torch.cat([meta_data['object_attention_masks'], meta_data['object_attention_masks']])
        cond_embeddings = self.text_encoder(caption)[0]
        max_length = caption.shape[-1]
        batch_size = caption.shape[0]
        n_vis = max(batch_size, self.args.num_validation_images)
        uncond_input = self.tokenizer([""] * n_vis, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.accelerator.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings[:n_vis]])
        self.scheduler.set_timesteps(self.args.num_inference_steps)
        latent_size = (n_vis, unet_native.config.in_channels, self.args.resolution // 8, self.args.resolution // 8)
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
        scaled_latents = 1.0 / 0.18215 * latent
        image = self.vae.decode(scaled_latents.to(self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        obj_to_img = obj_to_img.to(self.accelerator.device)
        triple_to_img = triple_to_img.to(self.accelerator.device)
        if self.accelerator.is_main_process:
            save_root = os.path.join(self.args.output_dir, "validation", f"step_{int(step):06d}")
            real_root = os.path.join(save_root, "real_image")
            os.makedirs(save_root, exist_ok=True)
            os.makedirs(real_root, exist_ok=True)
        for i in range(n_vis):
            pair = Image.new('RGB', size=(self.args.resolution * 2, self.args.resolution))
            gen_img_np = image[i]
            pair.paste(Image.fromarray(gen_img_np), box=(0, 0))
            mask_obj = (obj_to_img == i)
            mask_rel = (triple_to_img == i)
            objs_i = objs[mask_obj]
            boxes_pred_i = layout_preds[mask_obj]
            triples_i = triples[mask_rel]
            layout_img = self.layout_visualization(objs_i, boxes_pred_i, images=None, triples=triples_i, obj_indices=mask_obj.nonzero().view(-1).detach().cpu().numpy().tolist())
            pair.paste(layout_img, box=(self.args.resolution, 0))
            if self.accelerator.is_main_process:
                basename = os.path.basename(img_paths[i]) if isinstance(img_paths[i], str) else str(i)
                pair_path = os.path.join(save_root, f"{i:03d}_{basename}_pair.png")
                pair.save(pair_path)
                real_pair = Image.new('RGB', size=(self.args.resolution * 2, self.args.resolution))
                real_img = imgs[i].detach().cpu()
                real_img = (real_img / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
                real_img = (real_img * 255).round().astype("uint8")
                real_pair.paste(Image.fromarray(real_img), box=(0, 0))
                boxes_real_i = boxes[mask_obj]
                layout_real = self.layout_visualization(objs_i, boxes_real_i, images=None, triples=triples_i, obj_indices=mask_obj.nonzero().view(-1).detach().cpu().numpy().tolist())
                real_pair.paste(layout_real, box=(self.args.resolution, 0))
                real_pair.save(os.path.join(real_root, f"{i:03d}_{basename}_real_pair.png"))
        latent = None
        text_embeddings = None
        uncond_embeddings = None
        cond_embeddings = None
        object_embeddings = None
        meta_data = None
        torch.cuda.empty_cache()


    def layout_visualization(self, objs, boxes, images=None, triples=None, obj_indices=None):
        palette = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0), (255, 255, 255)]
        if images is None:
            layout = Image.new('RGB', size=(self.args.resolution, self.args.resolution), color=(16, 16, 16))
        else:
            layout = images.copy()
        draw_layout = ImageDraw.Draw(layout)

        centers = []
        image_idx = self.vocab['object_name_to_idx'].get('__image__', None)
        local_obj_indices = []
        for i, (obj, box) in enumerate(zip(objs, boxes)):
            obj_idx = int(obj.item()) if isinstance(obj, torch.Tensor) else obj
            obj_text = self.vocab['object_idx_to_name'][obj_idx]
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()
            if image_idx is not None and obj_idx == image_idx:
                continue

            arr = np.asarray(box, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.flatten()
            if not np.all(np.isfinite(arr)):
                try:
                    self.logger.error(f"检测到NaN/Inf：对象={obj_text}(idx={obj_idx})，box={arr.tolist()}")
                except Exception:
                    pass
                continue

            if len(arr) == 4:
                x0, y0, x1, y1 = (arr * self.args.resolution).tolist()
                if x1 < x0 or y1 < y0:
                    continue
                x0 = float(np.clip(x0, 0, self.args.resolution - 1))
                y0 = float(np.clip(y0, 0, self.args.resolution - 1))
                x1 = float(np.clip(x1, 0, self.args.resolution - 1))
                y1 = float(np.clip(y1, 0, self.args.resolution - 1))
                c = palette[i % len(palette)]
                draw_layout.rectangle([x0, y0, x1, y1], outline=c, width=1)
                draw_layout.text(xy=(int(x0), int(y0)), text=obj_text, fill=c)
                centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
                local_obj_indices.append(i)
            elif len(arr) == 5:
                cx, cy, w, h, a = arr
                if w <= 0 or h <= 0:
                    continue
                cx, cy, w, h = cx * self.args.resolution, cy * self.args.resolution, w * self.args.resolution, h * self.args.resolution
                if not np.all(np.isfinite([cx, cy, w, h, a])):
                    continue
                theta = float(a)
                dx, dy = w / 2.0, h / 2.0
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                corners = [(-dx, -dy), (-dx, dy), (dx, dy), (dx, -dy)]
                points = [(cx + cos_t*px - sin_t*py, cy + sin_t*px + cos_t*py) for px, py in corners]
                c = palette[i % len(palette)]
                draw_layout.line(points + [points[0]], fill=c, width=1)
                draw_layout.text(xy=(int(np.clip(cx, 0, self.args.resolution - 1)), int(np.clip(cy, 0, self.args.resolution - 1))), text=obj_text, fill=c)
                centers.append((cx, cy))
                local_obj_indices.append(i)

        if triples is not None and len(centers) > 0:
            if isinstance(triples, torch.Tensor):
                triples_np = triples.detach().cpu().numpy()
            else:
                triples_np = np.array(triples)
            if obj_indices is None:
                return layout
            filtered_global_indices = []
            for li in local_obj_indices:
                if li < len(obj_indices):
                    filtered_global_indices.append(int(obj_indices[li]))
            global_to_local = {int(g): li for li, g in enumerate(filtered_global_indices)}
            rel_color = (200, 200, 200)
            for t in triples_np:
                if len(t) < 3:
                    continue
                s_idx, p_idx, o_idx = int(t[0]), int(t[1]), int(t[2])
                if image_idx is not None and (s_idx == image_idx or o_idx == image_idx):
                    continue
                if s_idx not in global_to_local or o_idx not in global_to_local:
                    continue
                si = global_to_local[s_idx]
                oi = global_to_local[o_idx]
                sx, sy = centers[si]
                ox, oy = centers[oi]
                delta = 4
                sx2, sy2 = sx + np.random.randint(-delta, delta+1), sy + np.random.randint(-delta, delta+1)
                ox2, oy2 = ox + np.random.randint(-delta, delta+1), oy + np.random.randint(-delta, delta+1)
                draw_layout.line([(sx2, sy2), (ox2, oy2)], fill=rel_color, width=1)
                vx, vy = ox2 - sx2, oy2 - sy2
                vlen = np.hypot(vx, vy) + 1e-6
                ux, uy = vx / vlen, vy / vlen
                ah = 8
                perp = (-uy, ux)
                tip = (ox2, oy2)
                left = (ox2 - ah*ux + ah*perp[0], oy2 - ah*uy + ah*perp[1])
                right = (ox2 - ah*ux - ah*perp[0], oy2 - ah*uy - ah*perp[1])
                draw_layout.polygon([tip, left, right], fill=rel_color)
                pred_text = str(p_idx)
                if 'pred_idx_to_name' in self.vocab:
                    if p_idx < len(self.vocab['pred_idx_to_name']):
                        pred_text = self.vocab['pred_idx_to_name'][p_idx]
                mx, my = (sx2 + ox2) / 2.0, (sy2 + oy2) / 2.0
                draw_layout.text(xy=(int(mx), int(my)), text=pred_text, fill=rel_color)

        return layout

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.start()
