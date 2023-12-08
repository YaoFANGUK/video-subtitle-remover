import os
from typing import Union

import torch
import numpy as np
from PIL import Image
from backend.inpaint.utils.lama_util import prepare_img_and_mask
from backend import config


class LamaInpaint:
    def __init__(self, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), model_path=None) -> None:
        if model_path is None:
            model_path = os.path.join(config.LAMA_MODEL_PATH, 'big-lama.pt')
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]):
        image, mask = prepare_img_and_mask(image, mask, self.device)
        with torch.inference_mode():
            inpainted = self.model(image, mask)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            return cur_res

