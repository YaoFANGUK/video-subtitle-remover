import os

import cv2
import numpy as np
import torch
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset

from saicinpainting.evaluation.evaluator import InpaintingEvaluator
from saicinpainting.evaluation.losses.base_loss import SSIMScore, LPIPSScore, FIDScore


class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, image_size=(400, 600)):
        self.root_dir = root_dir
        self.files = sorted(os.listdir(root_dir))
        self.image_size = image_size

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.files[index])
        image = io.imread(img_name)
        image = resize(image, self.image_size, anti_aliasing=True)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        return image

    def __len__(self):
        return len(self.files)


def create_rectangle_mask(height, width):
    mask = np.ones((height, width))
    up_left_corner = width // 4, height // 4
    down_right_corner = (width - up_left_corner[0] - 1, height - up_left_corner[1] - 1)
    cv2.rectangle(mask, up_left_corner, down_right_corner, (0, 0, 0), thickness=cv2.FILLED)
    return mask


class Model():
    def __call__(self, img_batch, mask_batch):
        mean = (img_batch * mask_batch[:, None, :, :]).sum(dim=(2, 3)) / mask_batch.sum(dim=(1, 2))[:, None]
        inpainted = mean[:, :, None, None] * (1 - mask_batch[:, None, :, :]) + img_batch * mask_batch[:, None, :, :]
        return inpainted


class SimpleImageSquareMaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.mask = torch.FloatTensor(create_rectangle_mask(*self.dataset.image_size))
        self.model = Model()

    def __getitem__(self, index):
        img = self.dataset[index]
        mask = self.mask.clone()
        inpainted = self.model(img[None, ...], mask[None, ...])
        return dict(image=img, mask=mask, inpainted=inpainted)

    def __len__(self):
        return len(self.dataset)


dataset = SimpleImageDataset('imgs')
mask_dataset = SimpleImageSquareMaskDataset(dataset)
model = Model()
metrics = {
    'ssim': SSIMScore(),
    'lpips': LPIPSScore(),
    'fid': FIDScore()
}

evaluator = InpaintingEvaluator(
    mask_dataset, scores=metrics, batch_size=3, area_grouping=True
)

results = evaluator.evaluate(model)
print(results)
