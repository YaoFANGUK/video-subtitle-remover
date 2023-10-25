#!/usr/bin/env python3

import glob
import os

import PIL.Image as Image
import cv2
import numpy as np
import tqdm
import shutil


from saicinpainting.evaluation.utils import load_yaml


def generate_masks_for_img(infile, outmask_pattern, mask_size=200, step=0.5):
    inimg = Image.open(infile)
    width, height = inimg.size
    step_abs = int(mask_size * step)

    mask = np.zeros((height, width), dtype='uint8')
    mask_i = 0

    for start_vertical in range(0, height - step_abs, step_abs):
        for start_horizontal in range(0, width - step_abs, step_abs):
            mask[start_vertical:start_vertical + mask_size, start_horizontal:start_horizontal + mask_size] = 255

            cv2.imwrite(outmask_pattern.format(mask_i), mask)

            mask[start_vertical:start_vertical + mask_size, start_horizontal:start_horizontal + mask_size] = 0
            mask_i += 1


def main(args):
    if not args.indir.endswith('/'):
        args.indir += '/'
    if not args.outdir.endswith('/'):
        args.outdir += '/'

    config = load_yaml(args.config)

    in_files = list(glob.glob(os.path.join(args.indir, '**', f'*{config.img_ext}'), recursive=True))
    for infile in tqdm.tqdm(in_files):
        outimg = args.outdir + infile[len(args.indir):]
        outmask_pattern = outimg[:-len(config.img_ext)] + '_mask{:04d}.png'

        os.makedirs(os.path.dirname(outimg), exist_ok=True)
        shutil.copy2(infile, outimg)

        generate_masks_for_img(infile, outmask_pattern, **config.gen_kwargs)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to config for dataset generation')
    aparser.add_argument('indir', type=str, help='Path to folder with images')
    aparser.add_argument('outdir', type=str, help='Path to folder to store aligned images and masks to')

    main(aparser.parse_args())
