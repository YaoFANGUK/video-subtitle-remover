#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

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
from saicinpainting.training.trainers import load_checkpoint, DefaultInpaintingTrainingModule
from saicinpainting.utils import register_debug_signal_handlers, get_shape

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='../configs/prediction', config_name='default_inner_features.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        checkpoint_path = os.path.join(predict_config.model.path, 'models', predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False)
        model.freeze()
        model.to(device)

        assert isinstance(model, DefaultInpaintingTrainingModule), 'Only DefaultInpaintingTrainingModule is supported'
        assert isinstance(getattr(model.generator, 'model', None), torch.nn.Sequential)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)

        max_level = max(predict_config.levels)

        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(predict_config.outdir, os.path.splitext(mask_fname[len(predict_config.indir):])[0])
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                batch = move_to_device(default_collate([dataset[img_i]]), device)

                img = batch['image']
                mask = batch['mask']
                mask[:] = 0
                mask_h, mask_w = mask.shape[-2:]
                mask[:, :,
                    mask_h // 2 - predict_config.hole_radius : mask_h // 2 + predict_config.hole_radius,
                    mask_w // 2 - predict_config.hole_radius : mask_w // 2 + predict_config.hole_radius] = 1

                masked_img = torch.cat([img * (1 - mask), mask], dim=1)

                feats = masked_img
                for level_i, level in enumerate(model.generator.model):
                    feats = level(feats)
                    if level_i in predict_config.levels:
                        cur_feats = torch.cat([f for f in feats if torch.is_tensor(f)], dim=1) \
                            if isinstance(feats, tuple) else feats

                        if predict_config.slice_channels:
                            cur_feats = cur_feats[:, slice(*predict_config.slice_channels)]

                        cur_feat = cur_feats.pow(2).mean(1).pow(0.5).clone()
                        cur_feat -= cur_feat.min()
                        cur_feat /= cur_feat.std()
                        cur_feat = cur_feat.clamp(0, 1) / 1
                        cur_feat = cur_feat.cpu().numpy()[0]
                        cur_feat *= 255
                        cur_feat = np.clip(cur_feat, 0, 255).astype('uint8')
                        cv2.imwrite(cur_out_fname + f'_lev{level_i:02d}_norm.png', cur_feat)

                        # for channel_i in predict_config.channels:
                        #
                        #     cur_feat = cur_feats[0, channel_i].clone().detach().cpu().numpy()
                        #     cur_feat -= cur_feat.min()
                        #     cur_feat /= cur_feat.max()
                        #     cur_feat *= 255
                        #     cur_feat = np.clip(cur_feat, 0, 255).astype('uint8')
                        #     cv2.imwrite(cur_out_fname + f'_lev{level_i}_ch{channel_i}.png', cur_feat)
                    elif level_i >= max_level:
                        break
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
