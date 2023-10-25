#!/usr/bin/env python3
import glob
import logging
import os
import shutil
import sys
import traceback

from saicinpainting.evaluation.data import load_image
from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)


def main(args):
    try:
        if not args.indir.endswith('/'):
            args.indir += '/'

        for in_img in glob.glob(os.path.join(args.indir, '**', '*' + args.img_suffix), recursive=True):
            if 'mask' in os.path.basename(in_img):
                continue

            out_img_path = os.path.join(args.outdir, os.path.splitext(in_img[len(args.indir):])[0] + '.png')
            out_mask_path = f'{os.path.splitext(out_img_path)[0]}_mask.png'

            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

            img = load_image(in_img)
            height, width = img.shape[1:]
            pad_h, pad_w = int(height * args.coef / 2), int(width * args.coef / 2)

            mask = np.zeros((height, width), dtype='uint8')

            if args.expand:
                img = np.pad(img, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))
                mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
            else:
                mask[:pad_h] = 255
                mask[-pad_h:] = 255
                mask[:, :pad_w] = 255
                mask[:, -pad_w:] = 255

            # img = np.pad(img, ((0, 0), (pad_h * 2, pad_h * 2), (pad_w * 2, pad_w * 2)), mode='symmetric')
            # mask = np.pad(mask, ((pad_h * 2, pad_h * 2), (pad_w * 2, pad_w * 2)), mode = 'symmetric')

            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_img_path, img)

            cv2.imwrite(out_mask_path, mask)
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('indir', type=str, help='Root directory with images')
    aparser.add_argument('outdir', type=str, help='Where to store results')
    aparser.add_argument('--img-suffix', type=str, default='.png', help='Input image extension')
    aparser.add_argument('--expand', action='store_true', help='Generate mask by padding (true) or by cropping (false)')
    aparser.add_argument('--coef', type=float, default=0.2, help='How much to crop/expand in order to get masks')

    main(aparser.parse_args())
