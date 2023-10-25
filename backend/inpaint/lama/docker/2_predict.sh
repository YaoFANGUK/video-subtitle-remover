#!/usr/bin/env bash


if (( $# < 3 ))
then
    echo "Usage: $0 model_dir input_dir output_dir [other arguments to predict.py]"
    exit 1
fi

CURDIR="$(dirname $0)"
SRCDIR="$CURDIR/.."
SRCDIR="$(realpath $SRCDIR)"

MODEL_LOCAL_DIR="$(realpath $1)"
INPUT_LOCAL_DIR="$(realpath $2)"
OUTPUT_LOCAL_DIR="$(realpath $3)"
shift 3

mkdir -p "$OUTPUT_LOCAL_DIR"

docker run \
    -v "$SRCDIR":/home/user/project \
    -v "$MODEL_LOCAL_DIR":/data/checkpoint \
    -v "$INPUT_LOCAL_DIR":/data/input \
    -v "$OUTPUT_LOCAL_DIR":/data/output \
    -u $(id -u):$(id -g) \
    --name="lama-predict" \
    --rm \
    windj007/lama \
    /home/user/project/bin/predict.py \
        model.path=/data/checkpoint \
        indir=/data/input \
        outdir=/data/output \
        dataset.img_suffix=.png \
        $@
