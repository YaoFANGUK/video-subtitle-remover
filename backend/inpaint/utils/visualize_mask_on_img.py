import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import glob

from backend.inpaint.utils import load_img_to_array, show_mask


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--input_mask_glob", type=str, required=True,
        help="Glob to input masks",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )

if __name__ == "__main__":
    """Example usage:
    python visual_mask_on_img.py \
        --input_img FA_demo/FA1_dog.png \
        --input_mask_glob "results/FA1_dog/mask*.png" \
        --output_dir results
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    img = load_img_to_array(args.input_img)
    img_stem = Path(args.input_img).stem

    mask_ps = sorted(glob.glob(args.input_mask_glob))

    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for mask_p in mask_ps:
        mask = load_img_to_array(mask_p)
        mask = mask.astype(np.uint8)

        # path to the results
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()
