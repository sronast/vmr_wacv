import argparse
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop, hflip
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config_painting_to_sketch.json'
)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=DEFAULT_CONFIG, type=str, help='path to the json config')
    parser.add_argument('--set', default=[], nargs='*', help='overrides as key.subkey=value')
    return parser.parse_args()


def load_config(path, overrides, required_paths):
    with open(path, 'r') as f:
        cfg = json.load(f)

    for item in overrides:
        key, value = item.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    missing = [name for name in required_paths if not str(cfg['paths'].get(name, '')).strip()]
    if missing:
        raise ValueError(f"Fill in these paths in {path}: {', '.join(missing)}")
    return cfg


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_domain(cfg, name):
    preset = cfg.get('domains', {}).get(name.lower())
    if preset is None:
        print(f'WARNING: no preset for domain "{name}", using a generic description')
        preset = {
            'description': f'a {name} style image',
            'llm_instructions': '',
            'prompt_suffix': name,
            'negative_prompt': '',
        }
    return preset


def list_images(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Image directory not found: {directory}')
    paths = []
    for current, _, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(current, name))
    return sorted(paths)


def sample_target_images(cfg):
    paths = list_images(cfg['paths']['target_dir'])
    subset = cfg.get('target_subset', {})
    count = len(paths)
    fraction = subset.get('fraction', 1.0)
    if fraction < 1.0:
        count = max(1, round(count * fraction))
    if subset.get('max_images'):
        count = min(count, subset['max_images'])
    total = len(paths)
    if count < total:
        paths = sorted(random.Random(cfg['seed']).sample(paths, count))
    if total:
        print(f'Using {len(paths)} of {total} target images ({100.0 * len(paths) / total:.2f}%)')
    return paths


def build_captions(cfg, image_paths):
    finetune_cfg = cfg['finetune']
    domain = get_domain(cfg, cfg['target_domain'])
    description = domain['description']
    suffix = domain.get('prompt_suffix', '')
    mode = finetune_cfg['caption_mode']

    if mode == 'fixed':
        caption = f'{description}, {suffix}' if suffix else description
        return {path: caption for path in image_paths}
    if mode != 'vlm':
        raise ValueError(f'Unknown finetune.caption_mode: {mode}')

    captions_path = os.path.join(cfg['paths']['finetuned_model_save_dir'], 'captions.json')
    captions = {}
    if os.path.exists(captions_path):
        with open(captions_path, 'r') as f:
            captions = json.load(f)

    missing = [path for path in image_paths if path not in captions]
    if missing:
        captioner = AutoModelForCausalLM.from_pretrained(
            cfg['models']['vlm_id'],
            revision=cfg['models']['vlm_revision'],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={'': cfg['device']},
        )
        captioner.eval()
        for path in tqdm(missing, desc='Captioning target images'):
            image = Image.open(path).convert('RGB')
            with torch.no_grad():
                caption = captioner.caption(image, length=finetune_cfg['caption_length'])['caption'].strip()
            captions[path] = f'{caption}, {suffix}' if suffix else caption

        with open(captions_path, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        del captioner
        gc.collect()
        torch.cuda.empty_cache()

    return {path: captions[path] for path in image_paths}


class TargetDataset(Dataset):
    def __init__(self, image_paths, captions, finetune_cfg):
        self.image_paths = image_paths
        self.captions = captions
        self.resolution = finetune_cfg['resolution']
        self.center_crop = finetune_cfg['center_crop']
        self.random_flip = finetune_cfg['random_flip']
        self.caption_dropout = finetune_cfg['caption_dropout']
        self.resize = transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        original_size = (image.height, image.width)
        image = self.resize(image)

        if self.random_flip and random.random() < 0.5:
            image = hflip(image)

        if self.center_crop:
            top = max(0, int(round((image.height - self.resolution) / 2.0)))
            left = max(0, int(round((image.width - self.resolution) / 2.0)))
        else:
            top = random.randint(0, max(0, image.height - self.resolution))
            left = random.randint(0, max(0, image.width - self.resolution))
        image = crop(image, top, left, self.resolution, self.resolution)

        caption = self.captions[path]
        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            caption = ''

        time_ids = torch.tensor(list(original_size) + [top, left] + [self.resolution, self.resolution])
        return {
            'pixel_values': self.to_tensor(image),
            'time_ids': time_ids,
            'caption': caption,
        }


def collate_fn(examples):
    return {
        'pixel_values': torch.stack([example['pixel_values'] for example in examples]).contiguous().float(),
        'time_ids': torch.stack([example['time_ids'] for example in examples]),
        'captions': [example['caption'] for example in examples],
    }


def encode_prompt(captions, tokenizers, text_encoders, device):
    embeds = []
    pooled = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        input_ids = tokenizer(
            captions,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids.to(device)
        outputs = text_encoder(input_ids, output_hidden_states=True)
        pooled = outputs[0]
        embeds.append(outputs.hidden_states[-2])
    return torch.cat(embeds, dim=-1), pooled


def save_lora(unet, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(save_directory=output_dir, unet_lora_layers=state_dict)


def main(cfg):
    setup_seed(cfg['seed'])
    finetune_cfg = cfg['finetune']
    models = cfg['models']
    device = cfg['device']
    save_dir = cfg['paths']['finetuned_model_save_dir']
    os.makedirs(save_dir, exist_ok=True)

    image_paths = sample_target_images(cfg)
    if not image_paths:
        raise ValueError(f"No images found in {cfg['paths']['target_dir']}")
    print(f"Target domain: {cfg['target_domain']} ({len(image_paths)} unlabeled images)")

    with open(os.path.join(save_dir, 'finetune_images.json'), 'w') as f:
        json.dump(image_paths, f, indent=2)

    captions = build_captions(cfg, image_paths)

    use_fp16 = finetune_cfg['mixed_precision'] == 'fp16'
    weight_dtype = torch.float16 if use_fp16 else torch.float32

    tokenizers = [
        AutoTokenizer.from_pretrained(models['sd_id'], subfolder='tokenizer', use_fast=False),
        AutoTokenizer.from_pretrained(models['sd_id'], subfolder='tokenizer_2', use_fast=False),
    ]
    text_encoders = [
        CLIPTextModel.from_pretrained(models['sd_id'], subfolder='text_encoder'),
        CLIPTextModelWithProjection.from_pretrained(models['sd_id'], subfolder='text_encoder_2'),
    ]
    vae = AutoencoderKL.from_pretrained(models['vae_id'])
    unet = UNet2DConditionModel.from_pretrained(models['sd_id'], subfolder='unet')
    noise_scheduler = DDPMScheduler.from_pretrained(models['sd_id'], subfolder='scheduler')

    for module in text_encoders + [vae, unet]:
        module.requires_grad_(False)
        module.to(device, dtype=weight_dtype)

    unet.add_adapter(LoraConfig(
        r=finetune_cfg['lora_rank'],
        lora_alpha=finetune_cfg['lora_alpha'],
        init_lora_weights='gaussian',
        target_modules=finetune_cfg['lora_target_modules'],
    ))
    lora_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    for parameter in lora_parameters:
        parameter.data = parameter.data.to(torch.float32)
    print(f'Trainable LoRA parameters: {sum(parameter.numel() for parameter in lora_parameters):,}')

    if finetune_cfg['gradient_checkpointing']:
        unet.enable_gradient_checkpointing()

    dataset = TargetDataset(image_paths, captions, finetune_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=finetune_cfg['train_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=finetune_cfg['num_workers'],
        drop_last=len(dataset) >= finetune_cfg['train_batch_size'],
    )

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=finetune_cfg['learning_rate'],
        weight_decay=finetune_cfg['weight_decay'],
    )
    max_train_steps = finetune_cfg['max_train_steps']
    accumulation = finetune_cfg['gradient_accumulation_steps']
    lr_scheduler = get_scheduler(
        finetune_cfg['lr_scheduler'],
        optimizer=optimizer,
        num_warmup_steps=finetune_cfg['lr_warmup_steps'],
        num_training_steps=max_train_steps,
    )
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    with open(os.path.join(save_dir, 'finetune_config.json'), 'w') as f:
        json.dump({
            'dataset': cfg['dataset'],
            'target_domain': cfg['target_domain'],
            'num_images': len(image_paths),
            'models': models,
            'finetune': finetune_cfg,
        }, f, indent=2)

    unet.train()
    global_step = 0
    micro_step = 0
    progress = tqdm(total=max_train_steps, desc='LoRA finetuning')
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_train_steps:
        for batch in dataloader:
            with torch.no_grad():
                pixel_values = batch['pixel_values'].to(device, dtype=vae.dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                prompt_embeds, pooled_embeds = encode_prompt(batch['captions'], tokenizers, text_encoders, device)
                time_ids = batch['time_ids'].to(device, dtype=prompt_embeds.dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={'text_embeds': pooled_embeds, 'time_ids': time_ids},
                ).sample

            if noise_scheduler.config.prediction_type == 'v_prediction':
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
            scaler.scale(loss / accumulation).backward()
            micro_step += 1

            if micro_step % accumulation != 0:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_parameters, finetune_cfg['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f'{loss.item():.4f}', lr=f'{lr_scheduler.get_last_lr()[0]:.2e}')

            if finetune_cfg['checkpoint_steps'] and global_step % finetune_cfg['checkpoint_steps'] == 0:
                save_lora(unet, os.path.join(save_dir, f'checkpoint-{global_step}'))

            if global_step >= max_train_steps:
                break

    progress.close()
    save_lora(unet, save_dir)
    print(f'LoRA weights saved to {save_dir}')


if __name__ == '__main__':
    opt = parse_opt()
    cfg = load_config(opt.config, opt.set, ['target_dir', 'finetuned_model_save_dir'])
    main(cfg)