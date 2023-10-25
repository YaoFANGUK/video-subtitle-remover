#!/usr/bin/env bash

# paths to data are valid for mml7

source "$(dirname $0)/env.sh"

"$BINDIR/predict_inner_features.py" \
    -cn default_inner_features_ffc \
    model.path="/data/inpainting/paper_data/final_models/ours/r.suvorov_2021-03-05_17-34-05_train_ablv2_work_ffc075_resume_epoch39" \
    indir="/data/inpainting/paper_data/inner_features_vis/input/" \
    outdir="/data/inpainting/paper_data/inner_features_vis/output/ffc" \
    dataset.img_suffix=.png


"$BINDIR/predict_inner_features.py" \
    -cn default_inner_features_work \
    model.path="/data/inpainting/paper_data/final_models/ours/r.suvorov_2021-03-05_17-08-35_train_ablv2_work_resume_epoch37" \
    indir="/data/inpainting/paper_data/inner_features_vis/input/" \
    outdir="/data/inpainting/paper_data/inner_features_vis/output/work" \
    dataset.img_suffix=.png
