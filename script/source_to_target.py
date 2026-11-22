import argparse
import gc
import json
import os
import random

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, DPTForDepthEstimation, DPTImageProcessor
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
)

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

LORA_WEIGHT_NAME = 'pytorch_lora_weights.safetensors'
WEIGHT_EXTENSIONS = ('.safetensors', '.bin', '.pt', '.ckpt')
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


def list_source_images(cfg):
    source_dir = cfg['paths']['source_dir']
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    classes = cfg['convert'].get('classes')
    items = []
    for current, _, files in os.walk(source_dir):
        class_name = os.path.basename(os.path.normpath(current))
        if classes and class_name not in classes:
            continue
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                items.append((os.path.join(current, name), class_name))
    return sorted(items)


def load_tags(cfg):
    tags_path = cfg['paths']['tags_path']
    if not os.path.exists(tags_path):
        raise FileNotFoundError(f'Tags file not found: {tags_path}. Run generate_tags.py first.')
    with open(tags_path, 'r') as f:
        tags = json.load(f)['global_metadata']
    tags = {category: values for category, values in tags.items() if values}
    if not tags:
        raise ValueError(f'Tags file has no attribute values: {tags_path}')
    return tags


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_messages(caption, attributes, domain, class_name, max_words):
    description = domain['description']
    system = (
        f'You are an art expert who rewrites image descriptions so that they describe the same content '
        f'rendered as {description}. Keep the main subjects and scene of the original description and '
        f'express the visual appearance with the provided attribute terms. '
        f"{domain.get('llm_instructions', '')}"
    ).strip()

    attribute_text = ', '.join(f"{category.replace('_', ' ')}: {value}" for category, value in attributes.items())
    subject_line = f'Main subject: {class_name}\n' if class_name else ''
    opening = description[0].upper() + description[1:]
    user = (
        f'Original description: {caption}\n'
        f'{subject_line}'
        f'Attribute terms: {attribute_text}\n'
        f'Write one sentence of at most {max_words} words that starts with "{opening} of", '
        f'keeps the main subjects of the original description, and uses the attribute terms naturally. '
        f'Output only the sentence.'
    )
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]


def fallback_prompt(caption, attributes, domain):
    description = domain['description']
    opening = description[0].upper() + description[1:]
    terms = ', '.join(attributes.values())
    return f'{opening} of {caption.rstrip(".")}, {terms}'


def clean_llm_output(text, max_words):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ''
    sentence = lines[0].strip('"').strip("'").strip()
    return ' '.join(sentence.split()[:max_words]).rstrip('.,;: ')


def build_prompts(cfg):
    convert_cfg = cfg['convert']
    prompts_path = os.path.join(cfg['paths']['output_dir'], 'prompts.json')

    if convert_cfg['reuse_prompts'] and os.path.exists(prompts_path):
        with open(prompts_path, 'r') as f:
            records = json.load(f)
        print(f'Reusing {len(records)} prompts from {prompts_path}')
        return records

    items = list_source_images(cfg)
    if not items:
        raise ValueError(f"No source images found in {cfg['paths']['source_dir']}")
    print(f"Source domain: {cfg['source_domain']} ({len(items)} images)")

    tags = load_tags(cfg)
    domain = get_domain(cfg, cfg['target_domain'])
    rng = random.Random(cfg['seed'])

    captioner = AutoModelForCausalLM.from_pretrained(
        cfg['models']['vlm_id'],
        revision=cfg['models']['vlm_revision'],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={'': cfg['device']},
    )
    captioner.eval()

    llm = transformers.pipeline(
        'text-generation',
        model=cfg['models']['llm_id'],
        model_kwargs={'torch_dtype': torch.bfloat16},
        device_map='auto',
    )

    num_variants = convert_cfg['num_images_per_source']
    negative_parts = [convert_cfg.get('negative_prompt', ''), domain.get('negative_prompt', '')]
    negative_prompt = ', '.join(part for part in negative_parts if part)

    records = []
    for image_path, class_name in tqdm(items, desc='Building prompts'):
        image = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            caption = captioner.caption(image, length=convert_cfg['caption_length'])['caption'].strip()

        stem = os.path.splitext(os.path.basename(image_path))[0]
        for variant in range(num_variants):
            attributes = {category: rng.choice(values) for category, values in tags.items()}
            messages = build_messages(
                caption,
                attributes,
                domain,
                class_name.replace('_', ' ') if convert_cfg['include_class_name'] else None,
                convert_cfg['max_prompt_words'],
            )
            outputs = llm(messages, max_new_tokens=convert_cfg['llm_max_new_tokens'])
            target_caption = clean_llm_output(
                outputs[0]['generated_text'][-1]['content'], convert_cfg['max_prompt_words']
            )
            used_fallback = not target_caption
            if used_fallback:
                target_caption = fallback_prompt(caption, attributes, domain)

            suffix = domain.get('prompt_suffix', '')
            prompt = f'{target_caption}, {suffix}' if suffix else target_caption
            file_name = f'{stem}.png' if num_variants == 1 else f'{stem}_{variant}.png'

            records.append({
                'image_path': image_path,
                'class_name': class_name,
                'source_caption': caption,
                'attributes': attributes,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'used_fallback': used_fallback,
                'output_path': os.path.join(cfg['paths']['output_dir'], class_name, file_name),
            })

    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)
    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f'Prompts saved to {prompts_path}')

    del captioner, llm
    free_memory()
    return records


class DepthEstimator:
    def __init__(self, model_id, device):
        self.device = device
        self.model = DPTForDepthEstimation.from_pretrained(model_id).to(device)
        self.processor = DPTImageProcessor.from_pretrained(model_id)

    def __call__(self, image, resolution):
        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            depth_map = self.model(pixel_values).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1).float(),
            size=(resolution, resolution),
            mode='bicubic',
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
        depth = torch.cat([depth_map] * 3, dim=1).permute(0, 2, 3, 1).cpu().numpy()[0]
        return Image.fromarray((depth * 255.0).clip(0, 255).astype(np.uint8))


def get_canny_map(image, low_threshold, high_threshold):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def has_unet_weights(directory):
    if not os.path.exists(os.path.join(directory, 'config.json')):
        return False
    return any(name.startswith('diffusion_pytorch_model') for name in os.listdir(directory))


def load_finetuned_checkpoint(pipe, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Finetuned checkpoint not found: {checkpoint_path}. '
            f'Set paths.finetuned_model_load_path or set convert.use_finetuned to false.'
        )

    if os.path.isfile(checkpoint_path):
        pipe.load_lora_weights(os.path.dirname(checkpoint_path), weight_name=os.path.basename(checkpoint_path))
        print(f'Loaded LoRA weights from {checkpoint_path}')
        return 'lora'

    for unet_dir in (checkpoint_path, os.path.join(checkpoint_path, 'unet')):
        if os.path.isdir(unet_dir) and has_unet_weights(unet_dir):
            unet = UNet2DConditionModel.from_pretrained(unet_dir, torch_dtype=torch.float16)
            pipe.register_modules(unet=unet)
            print(f'Loaded finetuned U-Net from {unet_dir}')
            return 'unet'

    weight_files = sorted(name for name in os.listdir(checkpoint_path) if name.endswith(WEIGHT_EXTENSIONS))
    if not weight_files:
        raise FileNotFoundError(f'No LoRA or U-Net weights found in {checkpoint_path}')
    weight_name = LORA_WEIGHT_NAME if LORA_WEIGHT_NAME in weight_files else weight_files[0]
    pipe.load_lora_weights(checkpoint_path, weight_name=weight_name)
    print(f'Loaded LoRA weights from {os.path.join(checkpoint_path, weight_name)}')
    return 'lora'


def load_pipeline(cfg):
    convert_cfg = cfg['convert']
    models = cfg['models']
    control_types = convert_cfg['control_types']

    controlnets = [
        ControlNetModel.from_pretrained(
            models['controlnets'][control_type],
            variant='fp16',
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        for control_type in control_types
    ]
    controlnet = controlnets[0] if len(controlnets) == 1 else controlnets

    vae = AutoencoderKL.from_pretrained(models['vae_id'], torch_dtype=torch.float16)

    pipeline_class = (
        StableDiffusionXLControlNetImg2ImgPipeline
        if convert_cfg['init_from_source']
        else StableDiffusionXLControlNetPipeline
    )
    pipe = pipeline_class.from_pretrained(
        models['sd_id'],
        controlnet=controlnet,
        vae=vae,
        variant='fp16',
        use_safetensors=True,
        torch_dtype=torch.float16,
    )

    finetuned_kind = None
    if convert_cfg['use_finetuned']:
        checkpoint_path = str(cfg['paths'].get('finetuned_model_load_path', '')).strip()
        if not checkpoint_path:
            raise ValueError('Fill in paths.finetuned_model_load_path or set convert.use_finetuned to false.')
        finetuned_kind = load_finetuned_checkpoint(pipe, checkpoint_path)
    else:
        print('WARNING: convert.use_finetuned is false, generating with the base diffusion model')

    if convert_cfg['cpu_offload']:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(cfg['device'])
    pipe.set_progress_bar_config(disable=True)
    return pipe, finetuned_kind


def generate(cfg, records):
    convert_cfg = cfg['convert']
    control_types = convert_cfg['control_types']
    scales = convert_cfg['controlnet_conditioning_scale']
    if len(scales) != len(control_types):
        raise ValueError('convert.controlnet_conditioning_scale needs one value per entry in convert.control_types')
    resolution = convert_cfg['resolution']

    pending = [
        (index, record) for index, record in enumerate(records)
        if convert_cfg['overwrite'] or not os.path.exists(record['output_path'])
    ]
    print(f'{len(pending)} images to generate ({len(records) - len(pending)} already done)')
    if not pending:
        return

    pipe, finetuned_kind = load_pipeline(cfg)
    depth_estimator = None
    if 'depth' in control_types:
        depth_estimator = DepthEstimator(cfg['models']['depth_estimator_id'], cfg['device'])

    for index, record in tqdm(pending, desc='Generating'):
        image = Image.open(record['image_path']).convert('RGB')
        image = image.resize((resolution, resolution), Image.LANCZOS)

        control_maps = []
        for control_type in control_types:
            if control_type == 'canny':
                control_maps.append(
                    get_canny_map(image, convert_cfg['canny_low_threshold'], convert_cfg['canny_high_threshold'])
                )
            elif control_type == 'depth':
                control_maps.append(depth_estimator(image, resolution))
            else:
                raise ValueError(f'Unknown control type: {control_type}')

        single = len(control_maps) == 1
        control_input = control_maps[0] if single else control_maps

        kwargs = {
            'prompt': record['prompt'],
            'negative_prompt': record['negative_prompt'],
            'controlnet_conditioning_scale': scales[0] if single else scales,
            'num_inference_steps': convert_cfg['num_inference_steps'],
            'guidance_scale': convert_cfg['guidance_scale'],
            'generator': torch.Generator(device='cpu').manual_seed(cfg['seed'] + index),
        }
        if convert_cfg['init_from_source']:
            kwargs.update(image=image, control_image=control_input, strength=convert_cfg['strength'])
        else:
            kwargs.update(image=control_input, height=resolution, width=resolution)
        if finetuned_kind == 'lora':
            kwargs['cross_attention_kwargs'] = {'scale': convert_cfg['lora_scale']}

        generated = pipe(**kwargs).images[0].convert('RGB')
        os.makedirs(os.path.dirname(record['output_path']), exist_ok=True)
        generated.save(record['output_path'])

    print(f"Generated images saved to {cfg['paths']['output_dir']}")


def main(cfg):
    setup_seed(cfg['seed'])
    print(f"Setting: {cfg['dataset']} {cfg['source_domain']} -> {cfg['target_domain']}")
    records = build_prompts(cfg)
    generate(cfg, records)


if __name__ == '__main__':
    opt = parse_opt()
    cfg = load_config(opt.config, opt.set, ['source_dir', 'tags_path', 'output_dir'])
    main(cfg)