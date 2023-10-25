import numpy as np
from skimage import io
from skimage.segmentation import mark_boundaries


def save_item_for_vis(item, out_file):
    mask = item['mask'] > 0.5
    if mask.ndim == 3:
        mask = mask[0]
    img = mark_boundaries(np.transpose(item['image'], (1, 2, 0)),
                          mask,
                          color=(1., 0., 0.),
                          outline_color=(1., 1., 1.),
                          mode='thick')

    if 'inpainted' in item:
        inp_img = mark_boundaries(np.transpose(item['inpainted'], (1, 2, 0)),
                                  mask,
                                  color=(1., 0., 0.),
                                  mode='outer')
        img = np.concatenate((img, inp_img), axis=1)

    img = np.clip(img * 255, 0, 255).astype('uint8')
    io.imsave(out_file, img)


def save_mask_for_sidebyside(item, out_file):
    mask = item['mask']# > 0.5
    if mask.ndim == 3:
        mask = mask[0]
    mask = np.clip(mask * 255, 0, 255).astype('uint8')
    io.imsave(out_file, mask)

def save_img_for_sidebyside(item, out_file):
    img = np.transpose(item['image'], (1, 2, 0))
    img = np.clip(img * 255, 0, 255).astype('uint8')
    io.imsave(out_file, img)