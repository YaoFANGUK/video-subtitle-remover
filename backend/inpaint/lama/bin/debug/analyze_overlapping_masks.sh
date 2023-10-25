#!/bin/bash

BASEDIR="$(dirname $0)"

# paths are valid for mml7

# select images
#ls /data/inpainting/work/data/train | shuf | head -2000 | xargs -n1 -I{} cp {} /data/inpainting/mask_analysis/src

# generate masks
#"$BASEDIR/../gen_debug_mask_dataset.py" \
#    "$BASEDIR/../../configs/debug_mask_gen.yaml" \
#    "/data/inpainting/mask_analysis/src" \
#    "/data/inpainting/mask_analysis/generated"

# predict
#"$BASEDIR/../predict.py" \
#    model.path="simple_pix2pix2_gap_sdpl_novgg_large_b18_ffc075_batch8x15/saved_checkpoint/r.suvorov_2021-04-30_14-41-12_train_simple_pix2pix2_gap_sdpl_novgg_large_b18_ffc075_batch8x15_epoch22-step-574999" \
#    indir="/data/inpainting/mask_analysis/generated" \
#    outdir="/data/inpainting/mask_analysis/predicted" \
#    dataset.img_suffix=.jpg \
#    +out_ext=.jpg

# analyze good and bad samples
"$BASEDIR/../analyze_errors.py" \
    --only-report \
    --n-jobs 8 \
    "$BASEDIR/../../configs/analyze_mask_errors.yaml" \
    "/data/inpainting/mask_analysis/small/generated" \
    "/data/inpainting/mask_analysis/small/predicted" \
    "/data/inpainting/mask_analysis/small/report"
