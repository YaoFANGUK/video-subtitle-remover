import argparse
from enum import Enum

from .constant import InpaintMode

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    return args