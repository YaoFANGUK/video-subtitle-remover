# Flow visualization code adapted from https://github.com/tomrunia/OpticalFlow_Visualization
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

@torch.no_grad()
def flow_to_image(flow: torch.Tensor) -> torch.Tensor:

    """
    Converts a flow to an RGB image.

    Args:
        flow (Tensor): Flow of shape (N, 2, H, W) or (2, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 where each color corresponds
            to a given flow direction. Shape is (N, 3, H, W) or (3, H, W) depending on the input.
    """

    if flow.dtype != torch.float:
        raise ValueError(f"Flow should be of dtype torch.float, got {flow.dtype}.")

    orig_shape = flow.shape
    if flow.ndim == 3:
        flow = flow[None]  # Add batch dim

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")

    max_norm = torch.sum(flow**2, dim=1).sqrt().max()
    epsilon = torch.finfo((flow).dtype).eps
    normalized_flow = flow / (max_norm + epsilon)
    img = _normalized_flow_to_image(normalized_flow)

    if len(orig_shape) == 3:
        img = img[0]  # Remove batch dim
    return img

@torch.no_grad()
def _normalized_flow_to_image(normalized_flow: torch.Tensor) -> torch.Tensor:

    """
    Converts a batch of normalized flow to an RGB image.

    Args:
        normalized_flow (torch.Tensor): Normalized flow tensor of shape (N, 2, H, W)
    Returns:
       img (Tensor(N, 3, H, W)): Flow visualization image of dtype uint8.
    """

    N, _, H, W = normalized_flow.shape
    device = normalized_flow.device
    flow_image = torch.zeros((N, 3, H, W), dtype=torch.uint8, device=device)
    colorwheel = _make_colorwheel().to(device)  # shape [55x3]
    num_cols = colorwheel.shape[0]
    norm = torch.sum(normalized_flow**2, dim=1).sqrt()
    a = torch.atan2(-normalized_flow[:, 1, :, :], -normalized_flow[:, 0, :, :]) / torch.pi
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm * (1 - col)
        flow_image[:, c, :, :] = torch.floor(255. * col)
    return flow_image


@torch.no_grad()
def _make_colorwheel() -> torch.Tensor:
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255. * torch.arange(0., RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - torch.floor(255. * torch.arange(0., YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = torch.floor(255. * torch.arange(0., GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - torch.floor(255. * torch.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = torch.floor(255. * torch.arange(0., BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - torch.floor(255. * torch.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel
