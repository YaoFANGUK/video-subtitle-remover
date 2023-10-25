import cv2
import numpy as np
from typing import Tuple

def resize_and_pad(image: np.ndarray, mask: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes an image and its corresponding mask to have the longer side equal to `target_size` and pads them to make them
    both have the same size. The resulting image and mask have dimensions (target_size, target_size).

    Args:
        image: A numpy array representing the image to resize and pad.
        mask: A numpy array representing the mask to resize and pad.
        target_size: An integer specifying the desired size of the longer side after resizing.

    Returns:
        A tuple containing two numpy arrays - the resized and padded image and the resized and padded mask.
    """
    height, width, _ = image.shape
    max_dim = max(height, width)
    scale = target_size / max_dim
    new_height = int(height * scale)
    new_width = int(width * scale)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_height = target_size - new_height
    pad_width = target_size - new_width
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    image_padded = np.pad(image_resized, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    mask_padded = np.pad(mask_resized, ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant')
    return image_padded, mask_padded, (top_pad, bottom_pad, left_pad, right_pad)

def recover_size(image_padded: np.ndarray, mask_padded: np.ndarray, orig_size: Tuple[int, int], 
                 padding_factors: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes a padded and resized image and mask to the original size.

    Args:
        image_padded: A numpy array representing the padded and resized image.
        mask_padded: A numpy array representing the padded and resized mask.
        orig_size: A tuple containing two integers - the original height and width of the image before resizing and padding.

    Returns:
        A tuple containing two numpy arrays - the recovered image and the recovered mask with dimensions `orig_size`.
    """
    h,w,c = image_padded.shape
    top_pad, bottom_pad, left_pad, right_pad = padding_factors
    image = image_padded[top_pad:h-bottom_pad, left_pad:w-right_pad, :]
    mask = mask_padded[top_pad:h-bottom_pad, left_pad:w-right_pad]
    image_resized = cv2.resize(image, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, orig_size[::-1], interpolation=cv2.INTER_LINEAR)
    return image_resized, mask_resized




if __name__ == '__main__':

    # image = cv2.imread('example/boat.jpg')
    # mask = cv2.imread('example/boat_mask_2.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('example/groceries.jpg')
    # mask = cv2.imread('example/groceries_mask_2.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('example/bridge.jpg')
    # mask = cv2.imread('example/bridge_mask_2.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('example/person_umbrella.jpg')
    # mask = cv2.imread('example/person_umbrella_mask_2.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread('example/hippopotamus.jpg')
    # mask = cv2.imread('example/hippopotamus_mask_1.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('/data1/yutao/projects/IAM/Inpaint-Anything/example/fill-anything/sample5.jpeg')
    mask = cv2.imread('/data1/yutao/projects/IAM/Inpaint-Anything/example/fill-anything/sample5/mask.png', cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    print(mask.shape)
    cv2.imwrite('original_image.jpg', image)
    cv2.imwrite('original_mask.jpg', mask)
    image_padded, mask_padded, padding_factors = resize_and_pad(image, mask)
    cv2.imwrite('padded_image.png', image_padded)
    cv2.imwrite('padded_mask.png', mask_padded)
    print(image_padded.shape, mask_padded.shape, padding_factors)

    # ^ ------------------------------------------------------------------------------------
    # ^ Please conduct inpainting or filling here on the cropped image with the cropped mask
    # ^ ------------------------------------------------------------------------------------

    # resize and pad the image and mask

    # perform some operation on the 512x512 image and mask
    # ...

    # recover the image and mask to the original size
    height, width, _ = image.shape
    image_resized, mask_resized = recover_size(image_padded, mask_padded, (height, width), padding_factors)

    # save the resized and recovered image and mask
    cv2.imwrite('resized_and_padded_image.png', image_padded)
    cv2.imwrite('resized_and_padded_mask.png', mask_padded)
    cv2.imwrite('recovered_image.png', image_resized)
    cv2.imwrite('recovered_mask.png', mask_resized)

        