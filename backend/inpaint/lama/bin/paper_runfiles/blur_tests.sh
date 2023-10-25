##!/usr/bin/env bash
#
## !!! file set to make test_large_30k from the vanilla test_large: configs/test_large_30k.lst
#
## paths to data are valid for mml7
#PLACES_ROOT="/data/inpainting/Places365"
#OUT_DIR="/data/inpainting/paper_data/Places365_val_test"
#
#source "$(dirname $0)/env.sh"
#
#for datadir in test_large_30k  # val_large
#do
#    for conf in random_thin_256 random_medium_256 random_thick_256 random_thin_512 random_medium_512 random_thick_512
#    do
#        "$BINDIR/gen_mask_dataset.py" "$CONFIGDIR/data_gen/${conf}.yaml" \
#            "$PLACES_ROOT/$datadir" "$OUT_DIR/$datadir/$conf" --n-jobs 8
#
#        "$BINDIR/calc_dataset_stats.py" --samples-n 20 "$OUT_DIR/$datadir/$conf" "$OUT_DIR/$datadir/${conf}_stats"
#    done
#
#    for conf in segm_256 segm_512
#    do
#        "$BINDIR/gen_mask_dataset.py" "$CONFIGDIR/data_gen/${conf}.yaml" \
#            "$PLACES_ROOT/$datadir" "$OUT_DIR/$datadir/$conf" --n-jobs 2
#
#        "$BINDIR/calc_dataset_stats.py" --samples-n 20 "$OUT_DIR/$datadir/$conf" "$OUT_DIR/$datadir/${conf}_stats"
#    done
#done
#
#IN_DIR="/data/inpainting/paper_data/Places365_val_test/test_large_30k/random_medium_512"
#PRED_DIR="/data/inpainting/predictions/final/images/r.suvorov_2021-03-05_17-08-35_train_ablv2_work_resume_epoch37/random_medium_512"
#BLUR_OUT_DIR="/data/inpainting/predictions/final/blur/images"
#
#for b in 0.1
#
#"$BINDIR/blur_predicts.py" "$BASEDIR/../../configs/eval2.yaml" "$CUR_IN_DIR" "$CUR_OUT_DIR" "$CUR_EVAL_DIR"
#
