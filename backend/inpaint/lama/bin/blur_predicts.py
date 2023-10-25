#!/usr/bin/env python3

import os

import cv2
import numpy as np
import tqdm

from saicinpainting.evaluation.data import PrecomputedInpaintingResultsDataset
from saicinpainting.evaluation.utils import load_yaml


def main(args):
    config = load_yaml(args.config)

    if not args.predictdir.endswith('/'):
        args.predictdir += '/'

    dataset = PrecomputedInpaintingResultsDataset(args.datadir, args.predictdir, **config.dataset_kwargs)

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    for img_i in tqdm.trange(len(dataset)):
        pred_fname = dataset.pred_filenames[img_i]
        cur_out_fname = os.path.join(args.outpath, pred_fname[len(args.predictdir):])
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

        sample = dataset[img_i]
        img = sample['image']
        mask = sample['mask']
        inpainted = sample['inpainted']

        inpainted_blurred = cv2.GaussianBlur(np.transpose(inpainted, (1, 2, 0)),
                                             ksize=(args.k, args.k),
                                             sigmaX=args.s, sigmaY=args.s,
                                             borderType=cv2.BORDER_REFLECT)

        cur_res = (1 - mask) * np.transpose(img, (1, 2, 0)) + mask * inpainted_blurred
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to evaluation config')
    aparser.add_argument('datadir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('predictdir', type=str,
                         help='Path to folder with predicts (e.g. predict_hifill_baseline.py)')
    aparser.add_argument('outpath', type=str, help='Where to put results')
    aparser.add_argument('-s', type=float, default=0.1, help='Gaussian blur sigma')
    aparser.add_argument('-k', type=int, default=5, help='Kernel size in gaussian blur')

    main(aparser.parse_args())
