import argparse
import json
import os
import random
import re
from collections import Counter
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

STOP_WORDS = {'a', 'an', 'the', 'is', 'are', 'and', 'or', 'with', 'of', 'in', 'on', 'at', 'to', 'for'}

RESPONSE_PREFIX = re.compile(
    r'^(this|the|it)?\s*(image|sketch|drawing|painting|picture|photo|photograph|illustration|style|medium|'
    r'background|mood|composition|perspective|lighting|texture)?\s*'
    r'(appears to be|seems to be|looks like|is|shows|uses|conveys|are)\s+'
)
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


def clean_response(response, max_keywords):
    response = response.lower().strip()
    response = RESPONSE_PREFIX.sub('', response)
    response = re.sub(r'[^\w\s-]', '', response)
    words = response.split()
    meaningful = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    selected = meaningful[:max_keywords] if meaningful else words[:max_keywords]
    return ' '.join(selected)


def load_vlm(cfg):
    model = AutoModelForCausalLM.from_pretrained(
        cfg['models']['vlm_id'],
        revision=cfg['models']['vlm_revision'],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={'': cfg['device']},
    )
    model.eval()
    return model


def main(cfg):
    tag_cfg = cfg['tags']
    attributes = tag_cfg['attributes']

    image_paths = sample_target_images(cfg)
    if not image_paths:
        raise ValueError(f"No images found in {cfg['paths']['target_dir']}")
    print(f"Target domain: {cfg['target_domain']} ({len(image_paths)} unlabeled images)")

    model = load_vlm(cfg)
    counters = {category: Counter() for category in attributes}
    failed = 0
    start_time = datetime.now()

    for image_path in tqdm(image_paths, desc='Extracting tags'):
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f'Error loading {image_path}: {e}')
            failed += 1
            continue

        if max(image.size) > tag_cfg['max_image_size']:
            image.thumbnail((tag_cfg['max_image_size'], tag_cfg['max_image_size']), Image.Resampling.LANCZOS)

        with torch.no_grad():
            try:
                encoded = model.encode_image(image)
            except Exception:
                encoded = image

            for category, question in attributes.items():
                try:
                    answer = model.query(encoded, question)['answer']
                except Exception as e:
                    print(f'Error processing {category} for {image_path}: {e}')
                    continue
                value = clean_response(answer, tag_cfg['max_keywords'])
                if value:
                    counters[category][value] += 1

    processing_time = (datetime.now() - start_time).total_seconds()

    global_metadata = {}
    counts = {}
    for category, counter in counters.items():
        kept = [(value, count) for value, count in counter.most_common() if count >= tag_cfg['min_count']]
        if tag_cfg.get('top_k'):
            kept = kept[:tag_cfg['top_k']]
        if not kept:
            print(f'WARNING: no values kept for category "{category}"')
        global_metadata[category] = sorted(value for value, _ in kept)
        counts[category] = dict(kept)

    output = {
        'processing_info': {
            'dataset': cfg['dataset'],
            'target_domain': cfg['target_domain'],
            'source_directory': cfg['paths']['target_dir'],
            'total_images': len(image_paths),
            'failed_images': failed,
            'processed_at': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'model_used': cfg['models']['vlm_id'],
            'model_revision': cfg['models']['vlm_revision'],
        },
        'global_metadata': global_metadata,
        'counts': counts,
        'statistics': {category: len(values) for category, values in global_metadata.items()},
    }

    tags_path = cfg['paths']['tags_path']
    os.makedirs(os.path.dirname(tags_path), exist_ok=True)
    with open(tags_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'Processed {len(image_paths) - failed} images in {processing_time:.1f} seconds')
    print(f'Tags saved to {tags_path}')


if __name__ == '__main__':
    opt = parse_opt()
    cfg = load_config(opt.config, opt.set, ['target_dir', 'tags_path'])
    main(cfg)