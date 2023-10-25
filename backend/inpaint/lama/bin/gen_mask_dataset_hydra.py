#!/usr/bin/env python3

import glob
import os
import shutil
import traceback
import hydra
from omegaconf import OmegaConf

import PIL.Image as Image
import numpy as np
from joblib import Parallel, delayed

from saicinpainting.evaluation.masks.mask import SegmentationMask, propose_random_square_crop
from saicinpainting.evaluation.utils import load_yaml, SmallMode
from saicinpainting.training.data.masks import MixedMaskGenerator


class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


def process_images(src_images, indir, outdir, config):
    if config.generator_kind == 'segmentation':
        mask_generator = SegmentationMask(**config.mask_generator_kwargs)
    elif config.generator_kind == 'random':
        mask_generator_kwargs = OmegaConf.to_container(config.mask_generator_kwargs, resolve=True)
        variants_n = mask_generator_kwargs.pop('variants_n', 2)
        mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**mask_generator_kwargs),
                                              variants_n=variants_n)
    else:
        raise ValueError(f'Unexpected generator kind: {config.generator_kind}')

    max_tamper_area = config.get('max_tamper_area', 1)

    for infile in src_images:
        try:
            file_relpath = infile[len(indir):]
            img_outpath = os.path.join(outdir, file_relpath)
            os.makedirs(os.path.dirname(img_outpath), exist_ok=True)

            image = Image.open(infile).convert('RGB')

            # scale input image to output resolution and filter smaller images
            if min(image.size) < config.cropping.out_min_size:
                handle_small_mode = SmallMode(config.cropping.handle_small_mode)
                if handle_small_mode == SmallMode.DROP:
                    continue
                elif handle_small_mode == SmallMode.UPSCALE:
                    factor = config.cropping.out_min_size / min(image.size)
                    out_size = (np.array(image.size) * factor).round().astype('uint32')
                    image = image.resize(out_size, resample=Image.BICUBIC)
            else:
                factor = config.cropping.out_min_size / min(image.size)
                out_size = (np.array(image.size) * factor).round().astype('uint32')
                image = image.resize(out_size, resample=Image.BICUBIC)

            # generate and select masks
            src_masks = mask_generator.get_masks(image)

            filtered_image_mask_pairs = []
            for cur_mask in src_masks:
                if config.cropping.out_square_crop:
                    (crop_left,
                     crop_top,
                     crop_right,
                     crop_bottom) = propose_random_square_crop(cur_mask,
                                                               min_overlap=config.cropping.crop_min_overlap)
                    cur_mask = cur_mask[crop_top:crop_bottom, crop_left:crop_right]
                    cur_image = image.copy().crop((crop_left, crop_top, crop_right, crop_bottom))
                else:
                    cur_image = image

                if len(np.unique(cur_mask)) == 0 or cur_mask.mean() > max_tamper_area:
                    continue

                filtered_image_mask_pairs.append((cur_image, cur_mask))

            mask_indices = np.random.choice(len(filtered_image_mask_pairs),
                                            size=min(len(filtered_image_mask_pairs), config.max_masks_per_image),
                                            replace=False)

            # crop masks; save masks together with input image
            mask_basename = os.path.join(outdir, os.path.splitext(file_relpath)[0])
            for i, idx in enumerate(mask_indices):
                cur_image, cur_mask = filtered_image_mask_pairs[idx]
                cur_basename = mask_basename + f'_crop{i:03d}'
                Image.fromarray(np.clip(cur_mask * 255, 0, 255).astype('uint8'),
                                mode='L').save(cur_basename + f'_mask{i:03d}.png')
                cur_image.save(cur_basename + '.png')
        except KeyboardInterrupt:
            return
        except Exception as ex:
            print(f'Could not make masks for {infile} due to {ex}:\n{traceback.format_exc()}')


@hydra.main(config_path='../configs/data_gen/whydra', config_name='random_medium_256.yaml')
def main(config: OmegaConf):
    if not config.indir.endswith('/'):
        config.indir += '/'

    os.makedirs(config.outdir, exist_ok=True)

    in_files = list(glob.glob(os.path.join(config.indir, '**', f'*.{config.location.extension}'),
                              recursive=True))
    if config.n_jobs == 0:
        process_images(in_files, config.indir, config.outdir, config)
    else:
        in_files_n = len(in_files)
        chunk_size = in_files_n // config.n_jobs + (1 if in_files_n % config.n_jobs > 0 else 0)
        Parallel(n_jobs=config.n_jobs)(
            delayed(process_images)(in_files[start:start+chunk_size], config.indir, config.outdir, config)
            for start in range(0, len(in_files), chunk_size)
        )


if __name__ == '__main__':
    main()
