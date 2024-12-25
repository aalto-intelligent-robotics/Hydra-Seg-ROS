import torch
from torch import Tensor
import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy

from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox


def masks(
    orig_img: np.ndarray,
    masks: Tensor,
    colors: List[List[int]],
    alpha: float = 0.5,
    retina_masks: bool = False,
    blacked_out_rest: bool = False,
) -> np.ndarray:
    """
    Plot masks on image.

    Args:
        orig_img (tensor): The original image, shape: [h, w, 3]
        masks (tensor): Predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): Colors for predicted masks, [[r, g, b] * n]
        alpha (float): Mask transparency: 0.0 fully transparent, 1.0 opaque
        retina_masks (bool): Whether to use high resolution masks or not. Defaults to False.
        blacked_out_rest (bool): Whether to black out the background class. Defaults to False.
    """
    if len(masks) > 0:
        masked_img = deepcopy(orig_img)
        img = LetterBox(masks.shape[1:])(image=orig_img)
        im_device = (
            torch.as_tensor(img, dtype=torch.float16, device=masks.data.device)
            .permute(2, 0, 1)
            .flip(0)
            .contiguous()
            / 255
        )
        if len(masks) == 0:
            masked_img[:] = im_device.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        if im_device.device != masks.device:
            im_device = im_device.to(masks.device)
        if blacked_out_rest:
            im_device[:] = 0
        colors_tensor = (
            torch.tensor(colors, device=masks.device, dtype=torch.float32) / 255.0
        )  # shape(n,3)
        colors_tensor = colors_tensor[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors_tensor * alpha)  # shape(n,h,w,3)

        inv_alpha_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = masks_color.max(dim=0).values  # shape(n,h,w,3)

        im_device = im_device.flip(dims=[0])  # flip channel
        im_device = im_device.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_device = im_device * inv_alpha_masks[-1] + mcs
        im_mask = im_device * 255
        im_mask_np = im_mask.byte().cpu().numpy()
        masked_img[:] = (
            im_mask_np
            if retina_masks
            else ops.scale_image(im_mask_np, masked_img.shape)
        )
    else:
        masked_img = np.zeros_like(orig_img)
    return masked_img
