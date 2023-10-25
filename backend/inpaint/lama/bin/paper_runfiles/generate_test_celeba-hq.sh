#!/usr/bin/env bash

# paths to data are valid for mml-ws01
OUT_DIR="/media/inpainting/paper_data/CelebA-HQ_val_test"

source "$(dirname $0)/env.sh"

for datadir in "val" "test"
do
    for conf in random_thin_256 random_medium_256 random_thick_256 random_thin_512 random_medium_512 random_thick_512
    do
        "$BINDIR/gen_mask_dataset_hydra.py" -cn $conf datadir=$datadir location=mml-ws01-celeba-hq \
         location.out_dir=$OUT_DIR cropping.out_square_crop=False

        "$BINDIR/calc_dataset_stats.py" --samples-n 20 "$OUT_DIR/$datadir/$conf" "$OUT_DIR/$datadir/${conf}_stats"
    done
done
