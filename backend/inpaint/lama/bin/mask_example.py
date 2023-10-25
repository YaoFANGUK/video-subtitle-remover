import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

from saicinpainting.evaluation.masks.mask import SegmentationMask

im = io.imread('imgs/ex4.jpg')
im = resize(im, (512, 1024), anti_aliasing=True)
mask_seg = SegmentationMask(num_variants_per_mask=10)
mask_examples = mask_seg.get_masks(im)
for i, example in enumerate(mask_examples):
    plt.imshow(example)
    plt.show()
    plt.imsave(f'tmp/img_masks/{i}.png', example)
