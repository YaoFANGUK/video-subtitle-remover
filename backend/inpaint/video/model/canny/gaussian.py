from typing import Tuple

import torch
import torch.nn as nn

from .filter import filter2d, filter2d_separable
from .kernels import get_gaussian_kernel1d, get_gaussian_kernel2d


def gaussian_blur2d(
    input: torch.Tensor,
    kernel_size: Tuple[int, int],
    sigma: Tuple[float, float],
    border_type: str = 'reflect',
    separable: bool = True,
) -> torch.Tensor:
    r"""Create an operator that blurs a tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    if separable:
        kernel_x: torch.Tensor = get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y: torch.Tensor = get_gaussian_kernel1d(kernel_size[0], sigma[0])
        out = filter2d_separable(input, kernel_x[None], kernel_y[None], border_type)
    else:
        kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)
        out = filter2d(input, kernel[None], border_type)
    return out


class GaussianBlur2d(nn.Module):
    r"""Create an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        separable: run as composition of two 1d-convolutions.

    Returns:
        the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        border_type: str = 'reflect',
        separable: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.border_type = border_type
        self.separable = separable

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'sigma='
            + str(self.sigma)
            + ', '
            + 'border_type='
            + self.border_type
            + 'separable='
            + str(self.separable)
            + ')'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type, self.separable)