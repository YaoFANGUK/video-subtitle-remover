#!/usr/bin/env bash


if (( $# < 3 ))
then
    echo "Usage: $0 config_name input_images_dir image_mask_dataset_out_dir [other args to gen_mask_dataset.py]"
    exit 1
fi

CURDIR="$(dirname $0)"
SRCDIR="$CURDIR/.."
SRCDIR="$(realpath $SRCDIR)"

CONFIG_LOCAL_PATH="$(realpath $1)"
INPUT_LOCAL_DIR="$(realpath $2)"
OUTPUT_LOCAL_DIR="$(realpath $3)"
shift 3

mkdir -p "$OUTPUT_LOCAL_DIR"

docker run \
    -v "$SRCDIR":/home/user/project \
    -v "$CONFIG_LOCAL_PATH":/data/config.yaml \
    -v "$INPUT_LOCAL_DIR":/data/input \
    -v "$OUTPUT_LOCAL_DIR":/data/output \
    -u $(id -u):$(id -g) \
    --name="lama-mask-gen" \
    --rm \
    windj007/lama \
    /home/user/project/bin/gen_mask_dataset.py \
        /data/config.yaml /data/input /data/output $@
