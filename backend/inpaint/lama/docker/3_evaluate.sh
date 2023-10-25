#!/usr/bin/env bash


if (( $# < 3 ))
then
    echo "Usage: $0 original_dataset_dir predictions_dir output_dir [other arguments to evaluate_predicts.py]"
    exit 1
fi

CURDIR="$(dirname $0)"
SRCDIR="$CURDIR/.."
SRCDIR="$(realpath $SRCDIR)"

ORIG_DATASET_LOCAL_DIR="$(realpath $1)"
PREDICTIONS_LOCAL_DIR="$(realpath $2)"
OUTPUT_LOCAL_DIR="$(realpath $3)"
shift 3

mkdir -p "$OUTPUT_LOCAL_DIR"

docker run \
    -v "$SRCDIR":/home/user/project \
    -v "$ORIG_DATASET_LOCAL_DIR":/data/orig_dataset \
    -v "$PREDICTIONS_LOCAL_DIR":/data/predictions \
    -v "$OUTPUT_LOCAL_DIR":/data/output \
    -u $(id -u):$(id -g) \
    --name="lama-eval" \
    --rm \
    windj007/lama \
    /home/user/project/bin/evaluate_predicts.py \
        /home/user/project/configs/eval2_cpu.yaml \
        /data/orig_dataset \
        /data/predictions \
        /data/output/metrics.yaml \
        $@
