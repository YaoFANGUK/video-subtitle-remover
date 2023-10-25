#!/usr/bin/env python3

import os

import pandas as pd

from saicinpainting.evaluation.data import PrecomputedInpaintingResultsDataset
from saicinpainting.evaluation.evaluator import InpaintingEvaluator, lpips_fid100_f1
from saicinpainting.evaluation.losses.base_loss import SegmentationAwareSSIM, \
    SegmentationClassStats, SSIMScore, LPIPSScore, FIDScore, SegmentationAwareLPIPS, SegmentationAwareFID
from saicinpainting.evaluation.utils import load_yaml


def main(args):
    config = load_yaml(args.config)

    dataset = PrecomputedInpaintingResultsDataset(args.datadir, args.predictdir, **config.dataset_kwargs)

    metrics = {
        'ssim': SSIMScore(),
        'lpips': LPIPSScore(),
        'fid': FIDScore()
    }
    enable_segm = config.get('segmentation', dict(enable=False)).get('enable', False)
    if enable_segm:
        weights_path = os.path.expandvars(config.segmentation.weights_path)
        metrics.update(dict(
            segm_stats=SegmentationClassStats(weights_path=weights_path),
            segm_ssim=SegmentationAwareSSIM(weights_path=weights_path),
            segm_lpips=SegmentationAwareLPIPS(weights_path=weights_path),
            segm_fid=SegmentationAwareFID(weights_path=weights_path)
        ))
    evaluator = InpaintingEvaluator(dataset, scores=metrics,
                                    integral_title='lpips_fid100_f1', integral_func=lpips_fid100_f1,
                                    **config.evaluator_kwargs)

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    results = evaluator.evaluate()

    results = pd.DataFrame(results).stack(1).unstack(0)
    results.dropna(axis=1, how='all', inplace=True)
    results.to_csv(args.outpath, sep='\t', float_format='%.4f')

    if enable_segm:
        only_short_results = results[[c for c in results.columns if not c[0].startswith('segm_')]].dropna(axis=1, how='all')
        only_short_results.to_csv(args.outpath + '_short', sep='\t', float_format='%.4f')

        print(only_short_results)

        segm_metrics_results = results[['segm_ssim', 'segm_lpips', 'segm_fid']].dropna(axis=1, how='all').transpose().unstack(0).reorder_levels([1, 0], axis=1)
        segm_metrics_results.drop(['mean', 'std'], axis=0, inplace=True)

        segm_stats_results = results['segm_stats'].dropna(axis=1, how='all').transpose()
        segm_stats_results.index = pd.MultiIndex.from_tuples(n.split('/') for n in segm_stats_results.index)
        segm_stats_results = segm_stats_results.unstack(0).reorder_levels([1, 0], axis=1)
        segm_stats_results.sort_index(axis=1, inplace=True)
        segm_stats_results.dropna(axis=0, how='all', inplace=True)

        segm_results = pd.concat([segm_metrics_results, segm_stats_results], axis=1, sort=True)
        segm_results.sort_values(('mask_freq', 'total'), ascending=False, inplace=True)

        segm_results.to_csv(args.outpath + '_segm', sep='\t', float_format='%.4f')
    else:
        print(results)


if __name__ == '__main__':
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('config', type=str, help='Path to evaluation config')
    aparser.add_argument('datadir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('predictdir', type=str,
                         help='Path to folder with predicts (e.g. predict_hifill_baseline.py)')
    aparser.add_argument('outpath', type=str, help='Where to put results')

    main(aparser.parse_args())
