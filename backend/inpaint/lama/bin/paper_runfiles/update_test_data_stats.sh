#!/usr/bin/env bash

# paths to data are valid for mml7

source "$(dirname $0)/env.sh"

#INDIR="/data/inpainting/paper_data/Places365_val_test/test_large_30k"
#
#for dataset in random_medium_256 random_medium_512 random_thick_256 random_thick_512 random_thin_256 random_thin_512
#do
#    "$BINDIR/calc_dataset_stats.py" "$INDIR/$dataset" "$INDIR/${dataset}_stats2"
#done
#
#"$BINDIR/calc_dataset_stats.py" "/data/inpainting/evalset2" "/data/inpainting/evalset2_stats2"


INDIR="/data/inpainting/paper_data/CelebA-HQ_val_test/test"

for dataset in random_medium_256 random_thick_256 random_thin_256
do
    "$BINDIR/calc_dataset_stats.py" "$INDIR/$dataset" "$INDIR/${dataset}_stats2"
done


INDIR="/data/inpainting/paper_data/Paris_StreetView_Dataset_val_256/paris_eval_gt"

for dataset in random_medium_256 random_thick_256 random_thin_256
do
    "$BINDIR/calc_dataset_stats.py" "$INDIR/$dataset" "$INDIR/${dataset}_stats2"
done