#!/usr/bin/env python3
import os
import random

import cv2
import numpy as np

from saicinpainting.evaluation.data import PrecomputedInpaintingResultsDataset
from saicinpainting.evaluation.utils import load_yaml
from saicinpainting.training.visualizers.base import visualize_mask_and_images


def main(args):
    config = load_yaml(args.config)

    datasets = [PrecomputedInpaintingResultsDataset(args.datadir, cur_predictdir, **config.dataset_kwargs)
                for cur_predictdir in args.predictdirs]
    assert len({len(ds) for ds in datasets}) == 1
    len_first = len(datasets[0])

    indices = list(range(len_first))
    if len_first > args.max_n:
        indices = sorted(random.sample(indices, args.max_n))

    os.makedirs(args.outpath, exist_ok=True)

    filename2i = {}

    keys = ['image'] + [i for i in range(len(datasets))]
    for img_i in indices:
        try:
            mask_fname = os.path.basename(datasets[0].mask_filenames[img_i])
            if mask_fname in filename2i:
                filename2i[mask_fname] += 1
                idx = filename2i[mask_fname]
                mask_fname_only, ext = os.path.split(mask_fname)
                mask_fname = f'{mask_fname_only}_{idx}{ext}'
            else:
                filename2i[mask_fname] = 1

            cur_vis_dict = datasets[0][img_i]
            for ds_i, ds in enumerate(datasets):
                cur_vis_dict[ds_i] = ds[img_i]['inpainted']

            vis_img = visualize_mask_and_images(cur_vis_dict, keys,
                                                last_without_mask=False,
                                                mask_only_first=True,
                                                black_mask=args.black)
            vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

            out_fname = os.path.join(args.outpath, mask_fname)



            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_fname, vis_img)
        except Exception as ex:
            print(f'Could not process {img_i} due to {ex}')


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('--max-n', type=int, default=100, help='Maximum number of images to print')
    aparser.add_argument('--black', action='store_true', help='Whether to fill mask on GT with black')
    aparser.add_argument('config', type=str, help='Path to evaluation config (e.g. configs/eval1.yaml)')
    aparser.add_argument('outpath', type=str, help='Where to put results')
    aparser.add_argument('datadir', type=str,
                         help='Path to folder with images and masks')
    aparser.add_argument('predictdirs', type=str,
                         nargs='+',
                         help='Path to folders with predicts')


    main(aparser.parse_args())
