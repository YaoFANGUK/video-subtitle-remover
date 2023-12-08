import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian import gaussian_blur2d
from .kernels import get_canny_nms_kernel, get_hysteresis_kernel
from .sobel import spatial_gradient

def rgb_to_grayscale(image, rgb_weights = None):
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b


def canny(
    input: torch.Tensor,
    low_threshold: float = 0.1,
    high_threshold: float = 0.2,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: Tuple[float, float] = (1, 1),
    hysteresis: bool = True,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Find edges of the input image and filters them using the Canny algorithm.

    .. image:: _static/img/canny.png

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       canny.html>`__.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = canny(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if low_threshold > high_threshold:
        raise ValueError(
            "Invalid input thresholds. low_threshold should be smaller than the high_threshold. Got: {}>{}".format(
                low_threshold, high_threshold
            )
        )

    if low_threshold < 0 and low_threshold > 1:
        raise ValueError(f"Invalid input threshold. low_threshold should be in range (0,1). Got: {low_threshold}")

    if high_threshold < 0 and high_threshold > 1:
        raise ValueError(f"Invalid input threshold. high_threshold should be in range (0,1). Got: {high_threshold}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    # To Grayscale
    if input.shape[1] == 3:
        input = rgb_to_grayscale(input)

    # Gaussian filter
    blurred: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma)

    # Compute the gradients
    gradients: torch.Tensor = spatial_gradient(blurred, normalized=False)

    # Unpack the edges
    gx: torch.Tensor = gradients[:, :, 0]
    gy: torch.Tensor = gradients[:, :, 1]

    # Compute gradient magnitude and angle
    magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)
    angle: torch.Tensor = torch.atan2(gy, gx)

    # Radians to Degrees
    angle = 180.0 * angle / math.pi

    # Round angle to the nearest 45 degree
    angle = torch.round(angle / 45) * 45

    # Non-maximal suppression
    nms_kernels: torch.Tensor = get_canny_nms_kernel(device, dtype)
    nms_magnitude: torch.Tensor = F.conv2d(magnitude, nms_kernels, padding=nms_kernels.shape[-1] // 2)

    # Get the indices for both directions
    positive_idx: torch.Tensor = (angle / 45) % 8
    positive_idx = positive_idx.long()

    negative_idx: torch.Tensor = ((angle / 45) + 4) % 8
    negative_idx = negative_idx.long()

    # Apply the non-maximum suppression to the different directions
    channel_select_filtered_positive: torch.Tensor = torch.gather(nms_magnitude, 1, positive_idx)
    channel_select_filtered_negative: torch.Tensor = torch.gather(nms_magnitude, 1, negative_idx)

    channel_select_filtered: torch.Tensor = torch.stack(
        [channel_select_filtered_positive, channel_select_filtered_negative], 1
    )

    is_max: torch.Tensor = channel_select_filtered.min(dim=1)[0] > 0.0

    magnitude = magnitude * is_max

    # Threshold
    edges: torch.Tensor = F.threshold(magnitude, low_threshold, 0.0)

    low: torch.Tensor = magnitude > low_threshold
    high: torch.Tensor = magnitude > high_threshold

    edges = low * 0.5 + high * 0.5
    edges = edges.to(dtype)

    # Hysteresis
    if hysteresis:
        edges_old: torch.Tensor = -torch.ones(edges.shape, device=edges.device, dtype=dtype)
        hysteresis_kernels: torch.Tensor = get_hysteresis_kernel(device, dtype)

        while ((edges_old - edges).abs() != 0).any():
            weak: torch.Tensor = (edges == 0.5).float()
            strong: torch.Tensor = (edges == 1).float()

            hysteresis_magnitude: torch.Tensor = F.conv2d(
                edges, hysteresis_kernels, padding=hysteresis_kernels.shape[-1] // 2
            )
            hysteresis_magnitude = (hysteresis_magnitude == 1).any(1, keepdim=True).to(dtype)
            hysteresis_magnitude = hysteresis_magnitude * weak + strong

            edges_old = edges.clone()
            edges = hysteresis_magnitude + (hysteresis_magnitude == 0) * weak * 0.5

        edges = hysteresis_magnitude

    return magnitude, edges


class Canny(nn.Module):
    r"""Module that finds edges of the input image and filters them using the Canny algorithm.

    Args:
        input: input image tensor with shape :math:`(B,C,H,W)`.
        low_threshold: lower threshold for the hysteresis procedure.
        high_threshold: upper threshold for the hysteresis procedure.
        kernel_size: the size of the kernel for the gaussian blur.
        sigma: the standard deviation of the kernel for the gaussian blur.
        hysteresis: if True, applies the hysteresis edge tracking.
            Otherwise, the edges are divided between weak (0.5) and strong (1) edges.
        eps: regularization number to avoid NaN during backprop.

    Returns:
        - the canny edge magnitudes map, shape of :math:`(B,1,H,W)`.
        - the canny edge detection filtered by thresholds and hysteresis, shape of :math:`(B,1,H,W)`.

    Example:
        >>> input = torch.rand(5, 3, 4, 4)
        >>> magnitude, edges = Canny()(input)  # 5x3x4x4
        >>> magnitude.shape
        torch.Size([5, 1, 4, 4])
        >>> edges.shape
        torch.Size([5, 1, 4, 4])
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        kernel_size: Tuple[int, int] = (5, 5),
        sigma: Tuple[float, float] = (1, 1),
        hysteresis: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if low_threshold > high_threshold:
            raise ValueError(
                "Invalid input thresholds. low_threshold should be\
                             smaller than the high_threshold. Got: {}>{}".format(
                    low_threshold, high_threshold
                )
            )

        if low_threshold < 0 or low_threshold > 1:
            raise ValueError(f"Invalid input threshold. low_threshold should be in range (0,1). Got: {low_threshold}")

        if high_threshold < 0 or high_threshold > 1:
            raise ValueError(f"Invalid input threshold. high_threshold should be in range (0,1). Got: {high_threshold}")

        # Gaussian blur parameters
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Double threshold
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # Hysteresis
        self.hysteresis = hysteresis

        self.eps: float = eps

    def __repr__(self) -> str:
        return ''.join(
            (
                f'{type(self).__name__}(',
                ', '.join(
                    f'{name}={getattr(self, name)}' for name in sorted(self.__dict__) if not name.startswith('_')
                ),
                ')',
            )
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return canny(
            input, self.low_threshold, self.high_threshold, self.kernel_size, self.sigma, self.hysteresis, self.eps
        )