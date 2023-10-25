import torch
import torch.nn as nn
from torch.optim import Adam, SGD 
from kornia.filters import gaussian_blur2d
from kornia.geometry.transform import resize
from kornia.morphology import erosion
from torch.nn import functional as F
import numpy as np
import cv2

from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.modules.ffc import FFCResnetBlock
from saicinpainting.training.modules.pix2pixhd import ResnetBlock

from tqdm import tqdm


def _pyrdown(im : torch.Tensor, downsize : tuple=None):
    """downscale the image"""
    if downsize is None:
        downsize = (im.shape[2]//2, im.shape[3]//2)
    assert im.shape[1] == 3, "Expected shape for the input to be (n,3,height,width)"
    im = gaussian_blur2d(im, kernel_size=(5,5), sigma=(1.0,1.0))
    im = F.interpolate(im, size=downsize, mode='bilinear', align_corners=False)
    return im

def _pyrdown_mask(mask : torch.Tensor, downsize : tuple=None, eps : float=1e-8, blur_mask : bool=True, round_up : bool=True):
    """downscale the mask tensor

    Parameters
    ----------
    mask : torch.Tensor
        mask of size (B, 1, H, W)
    downsize : tuple, optional
        size to downscale to. If None, image is downscaled to half, by default None
    eps : float, optional
        threshold value for binarizing the mask, by default 1e-8
    blur_mask : bool, optional
        if True, apply gaussian filter before downscaling, by default True
    round_up : bool, optional
        if True, values above eps are marked 1, else, values below 1-eps are marked 0, by default True

    Returns
    -------
    torch.Tensor
        downscaled mask
    """

    if downsize is None:
        downsize = (mask.shape[2]//2, mask.shape[3]//2)
    assert mask.shape[1] == 1, "Expected shape for the input to be (n,1,height,width)"
    if blur_mask == True:
        mask = gaussian_blur2d(mask, kernel_size=(5,5), sigma=(1.0,1.0))
        mask = F.interpolate(mask, size=downsize,  mode='bilinear', align_corners=False)
    else:
        mask = F.interpolate(mask, size=downsize,  mode='bilinear', align_corners=False)
    if round_up:
        mask[mask>=eps] = 1
        mask[mask<eps] = 0
    else:
        mask[mask>=1.0-eps] = 1
        mask[mask<1.0-eps] = 0
    return mask

def _erode_mask(mask : torch.Tensor, ekernel : torch.Tensor=None, eps : float=1e-8):
    """erode the mask, and set gray pixels to 0"""
    if ekernel is not None:
        mask = erosion(mask, ekernel)
        mask[mask>=1.0-eps] = 1
        mask[mask<1.0-eps] = 0
    return mask


def _l1_loss(
    pred : torch.Tensor, pred_downscaled : torch.Tensor, ref : torch.Tensor, 
    mask : torch.Tensor, mask_downscaled : torch.Tensor, 
    image : torch.Tensor, on_pred : bool=True
    ):
    """l1 loss on src pixels, and downscaled predictions if on_pred=True"""
    loss = torch.mean(torch.abs(pred[mask<1e-8] - image[mask<1e-8]))
    if on_pred: 
        loss += torch.mean(torch.abs(pred_downscaled[mask_downscaled>=1e-8] - ref[mask_downscaled>=1e-8]))                
    return loss

def _infer(
    image : torch.Tensor, mask : torch.Tensor, 
    forward_front : nn.Module, forward_rears : nn.Module, 
    ref_lower_res : torch.Tensor, orig_shape : tuple, devices : list, 
    scale_ind : int, n_iters : int=15, lr : float=0.002):
    """Performs inference with refinement at a given scale.

    Parameters
    ----------
    image : torch.Tensor
        input image to be inpainted, of size (1,3,H,W)
    mask : torch.Tensor
        input inpainting mask, of size (1,1,H,W) 
    forward_front : nn.Module
        the front part of the inpainting network
    forward_rears : nn.Module
        the rear part of the inpainting network
    ref_lower_res : torch.Tensor
        the inpainting at previous scale, used as reference image
    orig_shape : tuple
        shape of the original input image before padding
    devices : list
        list of available devices
    scale_ind : int
        the scale index
    n_iters : int, optional
        number of iterations of refinement, by default 15
    lr : float, optional
        learning rate, by default 0.002

    Returns
    -------
    torch.Tensor
        inpainted image
    """
    masked_image = image * (1 - mask)
    masked_image = torch.cat([masked_image, mask], dim=1)

    mask = mask.repeat(1,3,1,1)
    if ref_lower_res is not None:
        ref_lower_res = ref_lower_res.detach()
    with torch.no_grad():
        z1,z2 = forward_front(masked_image)
    # Inference
    mask = mask.to(devices[-1])
    ekernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)).astype(bool)).float()
    ekernel = ekernel.to(devices[-1])
    image = image.to(devices[-1])
    z1, z2 = z1.detach().to(devices[0]), z2.detach().to(devices[0])
    z1.requires_grad, z2.requires_grad = True, True

    optimizer = Adam([z1,z2], lr=lr)

    pbar = tqdm(range(n_iters), leave=False)
    for idi in pbar:
        optimizer.zero_grad()
        input_feat = (z1,z2)
        for idd, forward_rear in enumerate(forward_rears):
            output_feat = forward_rear(input_feat)
            if idd < len(devices) - 1:
                midz1, midz2 = output_feat
                midz1, midz2 = midz1.to(devices[idd+1]), midz2.to(devices[idd+1])
                input_feat = (midz1, midz2)
            else:        
                pred = output_feat

        if ref_lower_res is None:
            break
        losses = {}
        ######################### multi-scale #############################
        # scaled loss with downsampler
        pred_downscaled = _pyrdown(pred[:,:,:orig_shape[0],:orig_shape[1]])
        mask_downscaled = _pyrdown_mask(mask[:,:1,:orig_shape[0],:orig_shape[1]], blur_mask=False, round_up=False)
        mask_downscaled = _erode_mask(mask_downscaled, ekernel=ekernel)
        mask_downscaled = mask_downscaled.repeat(1,3,1,1)
        losses["ms_l1"] = _l1_loss(pred, pred_downscaled, ref_lower_res, mask, mask_downscaled, image, on_pred=True)

        loss = sum(losses.values())
        pbar.set_description("Refining scale {} using scale {} ...current loss: {:.4f}".format(scale_ind+1, scale_ind, loss.item()))
        if idi < n_iters - 1:
            loss.backward()
            optimizer.step()
            del pred_downscaled
            del loss
            del pred
    # "pred" is the prediction after Plug-n-Play module
    inpainted = mask * pred + (1 - mask) * image
    inpainted = inpainted.detach().cpu()
    return inpainted

def _get_image_mask_pyramid(batch : dict, min_side : int, max_scales : int, px_budget : int):
    """Build the image mask pyramid

    Parameters
    ----------
    batch : dict
        batch containing image, mask, etc
    min_side : int
        minimum side length to limit the number of scales of the pyramid 
    max_scales : int
        maximum number of scales allowed
    px_budget : int
        the product H*W cannot exceed this budget, because of resource constraints

    Returns
    -------
    tuple
        image-mask pyramid in the form of list of images and list of masks
    """

    assert batch['image'].shape[0] == 1, "refiner works on only batches of size 1!"

    h, w = batch['unpad_to_size']
    h, w = h[0].item(), w[0].item()

    image = batch['image'][...,:h,:w]
    mask = batch['mask'][...,:h,:w]
    if h*w > px_budget:
        #resize 
        ratio = np.sqrt(px_budget / float(h*w))
        h_orig, w_orig = h, w
        h,w = int(h*ratio), int(w*ratio)
        print(f"Original image too large for refinement! Resizing {(h_orig,w_orig)} to {(h,w)}...")
        image = resize(image, (h,w),interpolation='bilinear', align_corners=False)
        mask = resize(mask, (h,w),interpolation='bilinear', align_corners=False)
        mask[mask>1e-8] = 1        
    breadth = min(h,w)
    n_scales = min(1 + int(round(max(0,np.log2(breadth / min_side)))), max_scales)        
    ls_images = []
    ls_masks = []
    
    ls_images.append(image)
    ls_masks.append(mask)
    
    for _ in range(n_scales - 1):
        image_p = _pyrdown(ls_images[-1])
        mask_p = _pyrdown_mask(ls_masks[-1])
        ls_images.append(image_p)
        ls_masks.append(mask_p)
    # reverse the lists because we want the lowest resolution image as index 0
    return ls_images[::-1], ls_masks[::-1]

def refine_predict(
    batch : dict, inpainter : nn.Module, gpu_ids : str, 
    modulo : int, n_iters : int, lr : float, min_side : int, 
    max_scales : int, px_budget : int
    ):
    """Refines the inpainting of the network

    Parameters
    ----------
    batch : dict
        image-mask batch, currently we assume the batchsize to be 1
    inpainter : nn.Module
        the inpainting neural network
    gpu_ids : str
        the GPU ids of the machine to use. If only single GPU, use: "0,"
    modulo : int
        pad the image to ensure dimension % modulo == 0
    n_iters : int
        number of iterations of refinement for each scale
    lr : float
        learning rate
    min_side : int
        all sides of image on all scales should be >= min_side / sqrt(2)
    max_scales : int
        max number of downscaling scales for the image-mask pyramid
    px_budget : int
        pixels budget. Any image will be resized to satisfy height*width <= px_budget

    Returns
    -------
    torch.Tensor
        inpainted image of size (1,3,H,W)
    """

    assert not inpainter.training
    assert not inpainter.add_noise_kwargs
    assert inpainter.concat_mask

    gpu_ids = [f'cuda:{gpuid}' for gpuid in gpu_ids.replace(" ","").split(",") if gpuid.isdigit()]
    n_resnet_blocks = 0
    first_resblock_ind = 0
    found_first_resblock = False
    for idl in range(len(inpainter.generator.model)):
        if isinstance(inpainter.generator.model[idl], FFCResnetBlock) or isinstance(inpainter.generator.model[idl], ResnetBlock):
            n_resnet_blocks += 1
            found_first_resblock = True
        elif not found_first_resblock:
            first_resblock_ind += 1
    resblocks_per_gpu = n_resnet_blocks // len(gpu_ids)

    devices = [torch.device(gpu_id) for gpu_id in gpu_ids]
    
    # split the model into front, and rear parts    
    forward_front = inpainter.generator.model[0:first_resblock_ind]
    forward_front.to(devices[0])
    forward_rears = []
    for idd in range(len(gpu_ids)):
        if idd < len(gpu_ids) - 1:
            forward_rears.append(inpainter.generator.model[first_resblock_ind + resblocks_per_gpu*(idd):first_resblock_ind+resblocks_per_gpu*(idd+1)]) 
        else:
            forward_rears.append(inpainter.generator.model[first_resblock_ind + resblocks_per_gpu*(idd):]) 
        forward_rears[idd].to(devices[idd]) 

    ls_images, ls_masks = _get_image_mask_pyramid(
        batch, 
        min_side, 
        max_scales, 
        px_budget
        )
    image_inpainted = None

    for ids, (image, mask) in enumerate(zip(ls_images, ls_masks)):
        orig_shape = image.shape[2:]
        image = pad_tensor_to_modulo(image, modulo)
        mask = pad_tensor_to_modulo(mask, modulo)
        mask[mask >= 1e-8] = 1.0
        mask[mask < 1e-8] = 0.0
        image, mask = move_to_device(image, devices[0]), move_to_device(mask, devices[0])
        if image_inpainted is not None:
            image_inpainted = move_to_device(image_inpainted, devices[-1])
        image_inpainted = _infer(image, mask, forward_front, forward_rears, image_inpainted, orig_shape, devices, ids, n_iters, lr)
        image_inpainted = image_inpainted[:,:,:orig_shape[0], :orig_shape[1]]
        # detach everything to save resources
        image = image.detach().cpu()
        mask = mask.detach().cpu()
    
    return image_inpainted
