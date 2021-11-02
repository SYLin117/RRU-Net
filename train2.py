import os.path

import torch.backends.cudnn as cudnn
from torch import optim

from eval import eval_net, dice_loss
from unet.unet_model import *
from utils import *

import matplotlib.pyplot as plt
import time

import wandb
from transunet import MyTransUNet

import re
from tqdm import tqdm


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-5,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              resize=(256, 256),
              train_dataset=None,
              val_dataset=None,
              dir_logs=None,
              model_name='no-model_name',
              resume=False,
              resume_id=None,
              latest_epoch=0, ):
    # training images are square
    # ids = split_ids(get_ids(dir_img))
    # iddataset = split_train_val(ids, val_percent)

    # dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=(300, 300))
    n_val = int(len(train_dataset) * val_percent) if not val_dataset else len(val_dataset)
    n_train = len(train_dataset) - n_val if not val_dataset else len(train_dataset)
    if not val_dataset:
        train_set, val_set = random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    else:
        train_set = train_dataset
        val_set = val_dataset

    train_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **train_args)
    val_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_args)

    id = wandb.util.generate_id() if not resume else resume_id
    experiment = wandb.init(project='Copy-Move-COCO', id=id, resume='allow', anonymous='must')
    experiment.name = model_name
    if not resume:
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=lr,
                                      val_percent=val_percent, resize=resize, checkpoints=save_cp, gpu=gpu))
    else:
        experiment.config.update(dict(epochs=latest_epoch, batch_size=batch_size, learning_rate=lr,
                                      val_percent=val_percent, resize=resize, checkpoints=save_cp, gpu=gpu))
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs,
               batch_size,
               lr,
               n_train,
               n_val,
               str(save_cp),
               str(gpu)))

    # n_train = len(iddataset['train'])
    # optimizer = optim.Adam(net.parameters(),
    #                        lr=lr,
    #                        weight_decay=0)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.BCELoss()

    Train_loss = []
    Valida_dice = []
    EPOCH = []

    global_step = 0
    start_epoch = 0 if not resume else latest_epoch  # 如果是resume從最後一次epoch開始 反之從0
    for epoch in range(start_epoch, epochs):
        net.train()

        start_epoch_time = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale, dataset)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale, dataset)

        epoch_loss = 0

        for i, b in enumerate(tqdm(train_loader)):
            global_step += 1 * batch_size
            start_batch = time.time()
            # imgs = np.array([i[0] for i in b]).astype(np.float32)
            # true_masks = np.array([i[1] for i in b]).astype(np.float32) / 255.
            # imgs = torch.from_numpy(imgs)
            # true_masks = torch.from_numpy(true_masks)

            imgs = b['image']
            true_masks = b['mask']

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            optimizer.zero_grad()

            masks_pred = net(imgs)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1).to(torch.float32)
            true_masks_flat = true_masks.view(-1).to(torch.float32)
            loss = criterion(masks_probs_flat, true_masks_flat) + dice_loss(masks_pred, true_masks)

            # print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / n_train, loss, time.time() - start_batch))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            experiment.log({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })
            if global_step % int(n_train * 0.01) == 0:
                random_index = int(np.random.random() * len(val_set))
                single_val = val_set[random_index]
                val_true_mask = single_val['mask']  # on cpu
                val_imgs = single_val['image']  # on cpu
                val_imgs = val_imgs.unsqueeze(dim=0)
                if gpu:
                    val_imgs = val_imgs.cuda()
                with torch.no_grad():
                    val_pred_mask = net(val_imgs)
                    val_pred_mask = torch.sigmoid(val_pred_mask).squeeze(0)
                experiment.log({
                    'images': wandb.Image(val_imgs[0].cpu()),
                    'masks': {
                        'true': wandb.Image(val_true_mask[0].float().cpu()),
                        'pred': wandb.Image(val_pred_mask[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                })

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        net.eval()

        val_dice = eval_net(net, val_loader, gpu)
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))
        experiment.log({
            'Validation Dice Coeff: {:.4f}': val_dice
        })
        Train_loss.append(epoch_loss / i)
        Valida_dice.append(val_dice)
        EPOCH.append(epoch)

        fig = plt.figure()
        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(EPOCH, Train_loss, c='red')
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')
        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_dice'], loc='best')
        plt.savefig(os.path.join(dir_logs, 'Training Process for lr-{}.png'.format(lr)), dpi=600)

        torch.save(net.state_dict(), os.path.join(dir_logs, 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        print('Spend time: {:.3f}s'.format(time.time() - start_epoch_time))
        print()


if __name__ == '__main__':
    """
    修改train.py並使用Pytorch Dataset 做為資料讀取的方式
    """
    import pathlib

    epochs, batchsize, scale, gpu = 50, 4, 1, True
    lr = 1e-5
    ft = False
    dataset_name = 'large_cm'
    model = 'MyTransUnet2'
    CURRENT_PATH = str(pathlib.Path().resolve())
    resize = (512, 512)

    dir_logs = os.path.join(CURRENT_PATH, 'result', 'logs', dataset_name, model)
    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)
    if model == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif model == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif model == 'Ringed_Res_Unet':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)
    elif model == 'MyTransUnet':
        net = MyTransUNet(in_channels=3, classes=1, img_dim=resize[0])
    elif model == 'MyTransUnet2':
        net = MyTransUNet2(in_channels=3, classes=1, img_dim=resize[0])
    else:
        raise Exception("model not implements.")

    id = None
    latest_epoch = None
    if ft:
        fine_tuning_model, latest_epoch = find_latest_epoch(
            os.path.join(CURRENT_PATH, 'result', 'logs', dataset_name, model, ))
        net.load_state_dict(torch.load(fine_tuning_model))
        epochs = epochs - latest_epoch
        print('Model loaded from {}'.format(fine_tuning_model))
        id = '2f3silq6'

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    DATASETS_DIR = get_dataset_root()
    dir_img = DATASETS_DIR.joinpath('COCO', 'coco2017_large_cm', 'A', 'train')
    dir_mask = DATASETS_DIR.joinpath('COCO', 'coco2017_large_cm', 'B', 'train')
    train_dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=resize)
    dir_img_val = DATASETS_DIR.joinpath('COCO', 'coco2017_large_cm', 'A', 'val')
    dir_mask_val = DATASETS_DIR.joinpath('COCO', 'coco2017_large_cm', 'B', 'val')
    val_dataset = ForgeDataset(dir_img_val, dir_mask_val, 1, mask_suffix='', resize=resize)
    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              val_percent=0.1,
              save_cp=True,
              gpu=gpu,
              resize=resize,
              train_dataset=train_dataset,
              val_dataset=val_dataset,
              dir_logs=dir_logs,
              model_name=model,
              resume=False,
              resume_id=id,
              latest_epoch=latest_epoch, )
