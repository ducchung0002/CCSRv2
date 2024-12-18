#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from ADD.models.discriminator import ProjectedDiscriminator

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from models.controlnet import ControlNetModel
from models.unet_2d_condition import UNet2DConditionModel
# from models.losses import LPIPSWithDiscriminator
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from accelerate import DistributedDataParallelKwargs

from dataloaders.paired_dataset_txt import PairedCaptionDataset


from ADD.models.vit import vit_large, vit_small
import ADD.utils.util_net as util_net

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    validation_prompt, validation_image, num_inference_steps=20, generator=generator
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
inference: true
---
    """
    model_card = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")


    parser.add_argument('--dataset_root_folders', type=str, default="")
    parser.add_argument("--is_module", action="store_true")
    parser.add_argument("--t_max", type=float, default=0.6666)
    parser.add_argument("--t_min", type=float, default=0.5)
    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--start_timesteps", type=int, default=999)

    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_lpips", type=float, default=1.0)
    parser.add_argument("--lambda_disc", type=float, default=0.05)
    parser.add_argument("--lambda_disc_train", type=float, default=0.5)
    parser.add_argument("--begin_disc", type=float, default=100)

    parser.add_argument(
        "--is_start_lr",
        type=bool,
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default='',
        help="Path to pretrained vae model."
        " If not specified vae weights are initialized from pre-trained model.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default='',
        help="Path to pretrained controlnet model."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    
    
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=[""],
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=100,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_ccsr_stage2",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def previous_timestep(timestep):
    if noise_scheduler.custom_timesteps:
        index = (noise_scheduler.timesteps == timestep).nonzero(as_tuple=True)[0][0]
        if index == noise_scheduler.timesteps.shape[0] - 1:
            prev_t = torch.tensor(-1)
        else:
            prev_t = noise_scheduler.timesteps[index + 1]
    else:
        num_inference_steps = (
            noise_scheduler.num_inference_steps if noise_scheduler.num_inference_steps else noise_scheduler.config.num_train_timesteps
        )
        prev_t = timestep - noise_scheduler.config.num_train_timesteps // num_inference_steps

    return prev_t

def predict_start_from_noise(sample, t, model_output):
    t = t.to(noise_scheduler.alphas_cumprod.device)
    prev_t = previous_timestep(t)

    # 1. compute alphas, betas
    alpha_prod_t = noise_scheduler.alphas_cumprod[t].to(sample.device)
    alpha_prod_t_prev = noise_scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else noise_scheduler.one
    alpha_prod_t_prev = alpha_prod_t_prev.to(sample.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if noise_scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif noise_scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif noise_scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    return pred_original_sample

# def main(args):
args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id

# Load the tokenizer
if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
elif args.pretrained_model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

# import correct text encoder class
text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

# Load scheduler and smodels
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
noise_scheduler.set_timesteps(args.num_inference_steps)

text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
)
unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
)

# Load VAE model
if args.vae_model_name_or_path:
    logger.info("Loading existing vae weights")
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, subfolder="vae", revision=args.revision)
else:
    logger.info("Loading pretrained vae weights")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

# Load Controlnet model
if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path,  subfolder="controlnet")
else:
    logger.info("Initializing controlnet weights from unet")
    controlnet = ControlNetModel.from_unet(unet, use_vae_encode_condition=True)
    
# # Load discriminator model
# discriminatornet = LPIPSWithDiscriminator(disc_start=1.0, kl_weight=0, perceptual_weight=1.0, disc_weight=0.5, disc_factor=1.0)

# Load discriminator model
discriminatornet = ProjectedDiscriminator(c_dim=384).train()
criterion_GAN = torch.nn.BCEWithLogitsLoss()
# 实例化提取cls_lr的特征网络
model_fea = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1.0)
util_net.reload_model(model_fea, torch.load('preset/models/dino/dinov2_vits14_pretrain.pth'))
model_fea.requires_grad_(False)

# load lpips model
# import lpips
# net_lpips = lpips.LPIPS(net='vgg').cuda()

# `accelerate` 0.16.0 will have better support for customized saving
if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        i = len(weights) - 1
        assert len(models) == 2 and len(weights) == 2
        for i, model in enumerate(models):
            if i==0:
                sub_dir = 'vae'
                model.save_pretrained(os.path.join(output_dir, sub_dir))
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 2
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if not isinstance(model, UNet2DConditionModel):
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet") # , low_cpu_mem_usage=False, ignore_mismatched_sizes=True

            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
controlnet.requires_grad_(False)
discriminatornet.train()
vae.train()

# unlease vae decoder for training
for name, params in vae.named_parameters():
    if 'decoder' in name:
        params.requires_grad = True

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.gradient_checkpointing:
    vae.enable_gradient_checkpointing()
    discriminatornet.enable_gradient_checkpointing()

# Check that all trainable models are in full precision
low_precision_error_string = (
    " Please make sure to always have all model weights in full float32 precision when starting training - even if"
    " doing mixed precision training, copy of the weights should still be float32."
)

if accelerator.unwrap_model(vae).dtype != torch.float32:
    raise ValueError(
        f"vae loaded as datatype {accelerator.unwrap_model(vae).dtype}. {low_precision_error_string}"
    )

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# Optimizer creation
params_to_optimize = list(vae.parameters())
optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)
    
params_to_optimize_disc = list(discriminatornet.parameters())
optimizer_disc = optimizer_class(
    params_to_optimize_disc,
    lr=args.learning_rate,
    betas=(0.9, 0.999),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

train_dataset = PairedCaptionDataset(root_folders=args.dataset_root_folders,
                                    tokenizer=tokenizer,
                                    gt_ratio=0) # let lr is gt

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=args.dataloader_num_workers,
    batch_size=args.train_batch_size,
    shuffle=False
)

# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

# Move controlnet, unet and text_encoder to device and cast to weight_dtype
controlnet.to(accelerator.device, dtype=weight_dtype)
text_encoder.to(accelerator.device, dtype=weight_dtype)
unet.to(accelerator.device, dtype=weight_dtype)
model_fea.to(accelerator.device, dtype=weight_dtype)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

lr_scheduler_disc = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer_disc,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

# Prepare everything with our `accelerator`.
vae, discriminatornet, optimizer, optimizer_disc, train_dataloader, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
    vae, discriminatornet, optimizer, optimizer_disc, train_dataloader, lr_scheduler, lr_scheduler_disc
)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    tracker_config = dict(vars(args))

    # tensorboard cannot handle list types for config
    tracker_config.pop("validation_prompt")
    tracker_config.pop("validation_image")

    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")

logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

for epoch in range(first_epoch, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        l_acc = [vae, discriminatornet]
        with accelerator.accumulate(*l_acc):
            with torch.no_grad():
                total_time_steps = noise_scheduler.timesteps
                num_time_steps = len(total_time_steps)
                if num_time_steps != 1:
                    timesteps_loop = total_time_steps[-round(num_time_steps*args.t_max):]
                    timesteps_loop = timesteps_loop[:-round(num_time_steps*args.t_min)]
                    t_max = timesteps_loop[0]
                    t_min = timesteps_loop[-1]

                pixel_values = batch["pixel_values"].to(accelerator.device)
                # if args.is_module:
                #     latents_gt = vae.module.encode(pixel_values).latent_dist.sample()
                #     latents_gt = latents_gt * vae.module.config.scaling_factor # Convert images to latent space
                # else:
                latents_gt = vae.encode(pixel_values).latent_dist.sample()
                latents_gt = latents_gt * vae.config.scaling_factor # Convert images to latent space

                encoder_hidden_states = text_encoder(batch["input_caption"].to(accelerator.device))[0]

                controlnet_image = batch["conditioning_pixel_values"].to(accelerator.device)
                controlnet_image_encode = 2*controlnet_image-1
                # if args.is_module:
                #     vae_encode_condition_hidden_states = vae.module.encode(controlnet_image_encode).latent_dist.sample()
                #     vae_encode_condition_hidden_states = vae_encode_condition_hidden_states * vae.module.config.scaling_factor
                # else:
                vae_encode_condition_hidden_states = vae.encode(controlnet_image_encode).latent_dist.sample()
                vae_encode_condition_hidden_states = vae_encode_condition_hidden_states * vae.config.scaling_factor # Convert images to latent space
                                
                if global_step > args.begin_disc:
                    lambda_l2 = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_disc = args.lambda_disc
                    lambda_disc_train = args.lambda_disc_train
                else:
                    lambda_l2 = args.lambda_l2
                    lambda_lpips = 0
                    lambda_disc = 0
                    lambda_disc_train = args.lambda_disc_train

                noise = torch.randn_like(latents_gt)
                bsz = latents_gt.shape[0]
                
                timesteps = args.start_timesteps * torch.ones(latents_gt.shape[0]).to(accelerator.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.start_timesteps==1:
                    noisy_latents = noise_scheduler.add_noise(vae_encode_condition_hidden_states, noise, timesteps)
                    noisy_latents = noisy_latents.to(dtype=weight_dtype)
                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)
                    controlnet_image = controlnet_image.to(dtype=weight_dtype)
                    vae_encode_condition_hidden_states = vae_encode_condition_hidden_states.to(dtype=weight_dtype)

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                        vae_encode_condition_hidden_states=vae_encode_condition_hidden_states,
                    )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                    # Predict x0 for T
                    x0_T = noisy_latents - model_pred
                else:
                    # Sample noise based on LR (controlnet_image) or a Random Noise?
                    if args.is_start_lr:
                        noisy_latents = noise_scheduler.add_noise(vae_encode_condition_hidden_states, noise, timesteps)
                        noisy_latents = noisy_latents.to(dtype=weight_dtype)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents_gt, noise, timesteps)
                        noisy_latents = noisy_latents.to(dtype=weight_dtype)

                    encoder_hidden_states = encoder_hidden_states.to(dtype=weight_dtype)
                    controlnet_image = controlnet_image.to(dtype=weight_dtype)
                    vae_encode_condition_hidden_states = vae_encode_condition_hidden_states.to(dtype=weight_dtype)

                    down_block_res_samples, mid_block_res_sample = controlnet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                            vae_encode_condition_hidden_states=vae_encode_condition_hidden_states,
                    )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                        return_dict=False,
                    )[0]

                    # Predict x0 for T
                    x0_T = predict_start_from_noise(noisy_latents, timesteps[0], model_pred)

                if num_time_steps!=1:
                    # Re-add noise on x0_tmax
                    noise2 = torch.randn_like(latents_gt)
                    timesteps = t_max * torch.ones(model_pred.shape[0]).to(accelerator.device)
                    timesteps = timesteps.long()
                    latents = noise_scheduler.add_noise(x0_T, noise2, timesteps[0])


                    # Denoising loop
                    for i, t in enumerate(timesteps_loop):
                        
                        # controlnet_latent_model_input = noise_scheduler.scale_model_input(latents, t)
                        latents = latents.to(dtype=weight_dtype)
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            latents,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            controlnet_cond=controlnet_image,
                            return_dict=False,
                            vae_encode_condition_hidden_states=vae_encode_condition_hidden_states,
                        )

                        # predict the noise residual
                        noise_pred = unet(
                            latents,
                            t,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            return_dict=False,
                        )[0]

                        # compute the previous noisy sample x_t -> x_t-1
                        latents_old = latents
                        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    x0_tmin = predict_start_from_noise(latents_old, t, noise_pred)
                    latents = x0_tmin
                    latents = latents.to(dtype=torch.float32)
                else:
                    latents = x0_T.to(dtype=torch.float32)

        
            # optimize the generator: vae decoder
            discriminatornet.requires_grad_(False)
            # if args.is_module:
            #     image = vae.module.decode(latents / vae.module.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
            # else:
            image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
            # compute the discriminator loss & update parameters
            _, cls_lr = model_fea(F.interpolate(controlnet_image, size=518, mode='bilinear'))

            # compute the generator loss
            pred_fake, _ = discriminatornet(image, cls_lr.detach())
            pred_fake = torch.cat(pred_fake, dim=1)
            gan_loss = -torch.mean(pred_fake)

            loss_x0 = F.mse_loss(image.float(), pixel_values.float(), reduction="mean") * lambda_l2
            # if lambda_lpips != 0:
                # loss_lpips = net_lpips(image.float(), pixel_values.float()).mean() * lambda_lpips
                # loss_x0 = loss_lpips + loss_x0

            loss = loss_x0 + lambda_disc * gan_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = vae.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # update discriminator
            discriminatornet.requires_grad_(True)
            # if args.is_module:
            #     discriminatornet.module.dino.requires_grad_(False)
            # else:
            discriminatornet.dino.requires_grad_(False)
            pred_real, features = discriminatornet(pixel_values, cls_lr.detach())
            pred_fake, _ = discriminatornet(image.detach(), cls_lr.detach())
            pred_fake = torch.cat(pred_fake, dim=1)

            pred_real = torch.cat(pred_real, dim=1)
            loss_real = torch.mean(torch.relu(1.0 - pred_real)) * lambda_disc_train

            loss_fake = torch.mean(torch.relu(1.0 + pred_fake)) * lambda_disc_train
            loss_disc = loss_real + loss_fake

            accelerator.backward(loss_disc)
            if accelerator.sync_gradients:
                params_to_clip = discriminatornet.parameters()
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer_disc.step()
            lr_scheduler_disc.step()
            optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
            model_fea.zero_grad(set_to_none=args.set_grads_to_none)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                if False:
                    image_logs = log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        controlnet,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

# Create the pipeline using using the trained modules and save it.
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    vae = accelerator.unwrap_model(vae)
    vae.save_pretrained(args.output_dir)

    if args.push_to_hub:
        save_model_card(
            repo_id,
            image_logs=image_logs,
            base_model=args.pretrained_model_name_or_path,
            repo_folder=args.output_dir,
        )
        upload_folder(
            repo_id=repo_id,
            folder_path=args.output_dir,
            commit_message="End of training",
            ignore_patterns=["step_*", "epoch_*"],
        )

accelerator.end_training()