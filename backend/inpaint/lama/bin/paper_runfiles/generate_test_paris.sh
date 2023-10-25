#!/usr/bin/env bash

# paths to data are valid for mml-ws01
OUT_DIR="/media/inpainting/paper_data/Paris_StreetView_Dataset_val"

source "$(dirname $0)/env.sh"

for datadir in paris_eval_gt
do
    for conf in random_thin_256 random_medium_256 random_thick_256 segm_256
    do
        "$BINDIR/gen_mask_dataset_hydra.py" -cn $conf datadir=$datadir location=mml-ws01-paris \
         location.out_dir=OUT_DIR cropping.out_square_crop=False cropping.out_min_size=227

        "$BINDIR/calc_dataset_stats.py" --samples-n 20 "$OUT_DIR/$datadir/$conf" "$OUT_DIR/$datadir/${conf}_stats"
    done
done
