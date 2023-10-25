#!/usr/bin/env python3

import os

import numpy as np
import tqdm
from scipy.ndimage.morphology import distance_transform_edt

from saicinpainting.evaluation.data import InpaintingDataset
from saicinpainting.evaluation.vis import save_item_for_vis


def main(args):
    dataset = InpaintingDataset(args.datadir, img_suffix='.png')

    area_bins = np.linspace(0, 1, args.area_bins + 1)

    heights = []
    widths = []
    image_areas = []
    hole_areas = []
    hole_area_percents = []
    known_pixel_distances = []

    area_bins_count = np.zeros(args.area_bins)
    area_bin_titles = [f'{area_bins[i] * 100:.0f}-{area_bins[i + 1] * 100:.0f}' for i in range(args.area_bins)]

    bin2i = [[] for _ in range(args.area_bins)]

    for i, item in enumerate(tqdm.tqdm(dataset)):
        h, w = item['image'].shape[1:]
        heights.append(h)
        widths.append(w)
        full_area = h * w
        image_areas.append(full_area)
        bin_mask = item['mask'] > 0.5
        hole_area = bin_mask.sum()
        hole_areas.append(hole_area)
        hole_percent = hole_area / full_area
        hole_area_percents.append(hole_percent)
        bin_i = np.clip(np.searchsorted(area_bins, hole_percent) - 1, 0, len(area_bins_count) - 1)
        area_bins_count[bin_i] += 1
        bin2i[bin_i].append(i)

        cur_dist = distance_transform_edt(bin_mask)
        cur_dist_inside_mask = cur_dist[bin_mask]
        known_pixel_distances.append(cur_dist_inside_mask.mean())

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'summary.txt'), 'w') as f:
        f.write(f'''Location:          {args.datadir}

Number of samples: {len(dataset)}

Image height: min {min(heights):5d} max {max(heights):5d} mean {np.mean(heights):.2f}
Image width:  min {min(widths):5d} max {max(widths):5d} mean {np.mean(widths):.2f}
Image area:   min {min(image_areas):7d} max {max(image_areas):7d} mean {np.mean(image_areas):.2f}
Hole area:    min {min(hole_areas):7d} max {max(hole_areas):7d} mean {np.mean(hole_areas):.2f}
Hole area %:  min {min(hole_area_percents) * 100:2.2f} max {max(hole_area_percents) * 100:2.2f} mean {np.mean(hole_area_percents) * 100:2.2f}
Dist 2known:  min {min(known_pixel_distances):2.2f} max {max(known_pixel_distances):2.2f} mean {np.mean(known_pixel_distances):2.2f} median {np.median(known_pixel_distances):2.2f}

Stats by hole area %:
''')
        for bin_i in range(args.area_bins):
            f.write(f'{area_bin_titles[bin_i]}%: '
                    f'samples number {area_bins_count[bin_i]}, '
                    f'{area_bins_count[bin_i] / len(dataset) * 100:.1f}%\n')

    for bin_i in range(args.area_bins):
        bindir = os.path.join(args.outdir, 'samples', area_bin_titles[bin_i])
        os.makedirs(bindir, exist_ok=True)
        bin_idx = bin2i[bin_i]
        for sample_i in np.random.choice(bin_idx, size=min(len(bin_idx), args.samples_n), replace=False):
            save_item_for_vis(dataset[sample_i], os.path.join(bindir, f'{sample_i}.png'))


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('datadir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('outdir', type=str, help='Where to put results')
    aparser.add_argument('--samples-n', type=int, default=10,
                         help='Number of sample images with masks to copy for visualization for each area bin')
    aparser.add_argument('--area-bins', type=int, default=10, help='How many area bins to have')

    main(aparser.parse_args())
