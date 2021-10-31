import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset, ForgeDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from transunet import MyTransUNet

# DATASETS_DIR = Path('/media/ian/WD/datasets')
DATASETS_DIR = Path('F:\\datasets')

# dir_img = DATASETS_DIR.joinpath('carvana-image-masking/train')
# dir_mask = DATASETS_DIR.joinpath('carvana-image-masking/train_masks')
# dir_checkpoint = Path('./checkpoints/')
#
dir_img = DATASETS_DIR.joinpath('total_forge', 'CM', 'test_and_train', 'train', 'images')
dir_mask = DATASETS_DIR.joinpath('total_forge', 'CM', 'test_and_train', 'train', 'masks')
dir_checkpoint = Path('./transunet_copymove_checkpoint/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.0001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              train_forge=False,
              resize=False):
    # 1. Create dataset

    if not train_forge:
        try:
            dataset = CarvanaDataset(dir_img, dir_mask, img_scale, resize)
        except (AssertionError, RuntimeError):
            dataset = BasicDataset(dir_img, dir_mask, img_scale)
    else:
        dataset = ForgeDataset(dir_img, dir_mask, 1)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    # loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='forge-trans-U-Net-3070', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # logging.info("true_masks unique value: {}".format(torch.unique(true_masks)))
                with torch.cuda.amp.autocast(enabled=amp):
                    try:
                        masks_pred = net(images)
                        # logging.info("mask_pred unique value: {}".format(torch.unique(masks_pred)))
                        # logging.info("true_masks unique value: {}".format(torch.unique(true_masks)))
                        # logging.info("mask_pred size value: {}".format(masks_pred.size()))
                        # logging.info("true_masks size value: {}".format(true_masks.size()))
                        loss = criterion(masks_pred, true_masks) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                    except RuntimeError as re:
                        print("image size: {}".format(images.size()))
                        print("error: {}".format(re))
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                with torch.no_grad():
                    # Evaluation round
                    if global_step % (n_train // (10 * batch_size)) == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            try:
                                if tag == 'srm.srm_layer.weight':
                                    continue
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())  # 不知道為什麼出錯
                            except AttributeError as ae:
                                print("params: {} got no grad".format(tag))

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                    ## print out srm kernel weight
                    # logging.info("net.srm.srm_layer.weight: {}".format(net.srm.srm_layer.weight))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--train-forge', default=True,
                        help='train on forge dataset')  ## when train my own coco forge used
    parser.add_argument('--resize', default=True, help='resize image to (256,256)')

    return parser.parse_args()


if __name__ == '__main__':
    def init_all(model, init_funcs):
        for name, params in model.named_parameters():
            if params.requires_grad == True:
                params_shape = len(params.shape)
                init_func = init_funcs.get(params_shape, init_funcs["default"])
                init_func(params)


    init_funcs = {
        1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.),  # can be bias
        2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.),  # can be weight
        3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.),  # can be conv1D filter
        4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.),  # can be conv2D filter
        "default": lambda x: torch.nn.init.constant(x, 1.),  # everything else
    }

    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = UNet(n_channels=3, n_classes=2, bilinear=True)
    net = MyTransUNet(in_channels=3, img_dim=256, classes=1)
    # for name, param in net.named_parameters():  # 確定SRM不可以train
    #     if name == 'srm.srm_layer.weight':
    #         param.requires_grad = False
    # init_all(net, init_funcs)  # 初始化所有Weight

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  train_forge=args.train_forge)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
