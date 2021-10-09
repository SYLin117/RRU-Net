import torch
import torch.nn.functional as F
import numpy as np

from dice_loss import dice_coeff
from tqdm import tqdm


def eval_net(net, dataset, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    for i, b in enumerate(dataset):
        # img = b[0].astype(np.float32)
        # true_mask = b[1].astype(np.float32) / 255
        #
        # img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        img = b['image']
        true_mask = b['mask']
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda().to(torch.float32)

        mask_pred = net(img)[0]
        mask_pred = (torch.sigmoid(mask_pred) > 0.5).to(torch.float32)

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i


def dice_loss(true_masks, mask_preds, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""
    if gpu:
        mask_preds = mask_preds.cuda().to(torch.float32)
        true_mask = true_masks.cuda().to(torch.float32)
    dice_loss = 1 - dice_coeff(mask_preds, true_masks).item()
    return dice_loss