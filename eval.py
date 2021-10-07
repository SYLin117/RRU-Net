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

# def evaluate(net, dataloader, device):
#     net.eval()
#     num_val_batches = len(dataloader)
#     dice_score = 0
#
#     # iterate over the validation set
#     for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#         image, mask_true = batch['image'], batch['mask']
#         # move images and labels to correct device and type
#         image = image.to(device=device, dtype=torch.float32)
#         mask_true = mask_true.to(device=device, dtype=torch.long)
#         mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
#
#         with torch.no_grad():
#             # predict the mask
#             mask_pred = net(image)
#
#             # convert to one-hot format
#             if net.n_classes == 1:
#                 mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
#                 # compute the Dice score
#                 dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
#             else:
#                 mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
#                 # compute the Dice score, ignoring background
#                 dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
#                                                     reduce_batch_first=False)
#
#     net.train()
#     return dice_score / num_val_batches
