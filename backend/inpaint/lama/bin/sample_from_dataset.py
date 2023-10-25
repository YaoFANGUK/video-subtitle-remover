#!/usr/bin/env python3

import os

import numpy as np
import tqdm
from skimage import io
from skimage.segmentation import mark_boundaries

from saicinpainting.evaluation.data import InpaintingDataset
from saicinpainting.evaluation.vis import save_item_for_vis

def save_mask_for_sidebyside(item, out_file):
    mask = item['mask']# > 0.5
    if mask.ndim == 3:
        mask = mask[0]
    mask = np.clip(mask * 255, 0, 255).astype('uint8')
    io.imsave(out_file, mask)

def save_img_for_sidebyside(item, out_file):
    img = np.transpose(item['image'], (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype('uint8')
    io.imsave(out_file, img)

def save_masked_img_for_sidebyside(item, out_file):
    mask = item['mask']
    img  = item['image']

    img = (1-mask) * img + mask
    img = np.transpose(img, (1, 2, 0))

    img = np.clip(img * 255, 0, 255).astype('uint8')
    io.imsave(out_file, img)

def main(args):
    dataset = InpaintingDataset(args.datadir, img_suffix='.png')

    area_bins = np.linspace(0, 1, args.area_bins + 1)

    heights = []
    widths = []
    image_areas = []
    hole_areas = []
    hole_area_percents = []
    area_bins_count = np.zeros(args.area_bins)
    area_bin_titles = [f'{area_bins[i] * 100:.0f}-{area_bins[i + 1] * 100:.0f}' for i in range(args.area_bins)]

    bin2i = [[] for _ in range(args.area_bins)]

    for i, item in enumerate(tqdm.tqdm(dataset)):
        h, w = item['image'].shape[1:]
        heights.append(h)
        widths.append(w)
        full_area = h * w
        image_areas.append(full_area)
        hole_area = (item['mask'] == 1).sum()
        hole_areas.append(hole_area)
        hole_percent = hole_area / full_area
        hole_area_percents.append(hole_percent)
        bin_i = np.clip(np.searchsorted(area_bins, hole_percent) - 1, 0, len(area_bins_count) - 1)
        area_bins_count[bin_i] += 1
        bin2i[bin_i].append(i)

    os.makedirs(args.outdir, exist_ok=True)
   
    for bin_i in range(args.area_bins):
        bindir = os.path.join(args.outdir, area_bin_titles[bin_i])
        os.makedirs(bindir, exist_ok=True)
        bin_idx = bin2i[bin_i]
        for sample_i in np.random.choice(bin_idx, size=min(len(bin_idx), args.samples_n), replace=False):
            item = dataset[sample_i]
            path = os.path.join(bindir, dataset.img_filenames[sample_i].split('/')[-1])
            save_masked_img_for_sidebyside(item, path)
           

if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--datadir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('--outdir', type=str, help='Where to put results')
    aparser.add_argument('--samples-n', type=int, default=10,
                         help='Number of sample images with masks to copy for visualization for each area bin')
    aparser.add_argument('--area-bins', type=int, default=10, help='How many area bins to have')

    main(aparser.parse_args())
