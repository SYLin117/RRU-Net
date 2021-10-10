import os.path

import torch.backends.cudnn as cudnn
from torch import optim

from eval import eval_net, dice_loss
from unet.unet_model import *
from utils import *

import matplotlib.pyplot as plt
import time

import wandb


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-5,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1,
              dataset=None,
              dir_logs=None):
    # training images are square
    # ids = split_ids(get_ids(dir_img))
    # iddataset = split_train_val(ids, val_percent)

    dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=(300, 300))
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    loader_args2 = dict(batch_size=1, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args2)

    experiment = wandb.init(project='Video(RRU-Net)', resume='allow', anonymous='must')
    experiment.name = 'RRU-Net'
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=lr,
                                  val_percent=val_percent, resize=(300, 300), checkpoints=save_cp, gpu=gpu))
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
    for epoch in range(epochs):
        net.train()

        start_epoch = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale, dataset)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale, dataset)

        epoch_loss = 0

        # for i, b in enumerate(batch(train, batch_size)):
        for i, b in enumerate(train_loader):
            global_step += 1
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

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / n_train, loss, time.time() - start_batch))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            experiment.log({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        net.eval()

        val_dice = eval_net(net, val_loader, gpu)
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation Dice': val_dice,
            # 'images': wandb.Image(images[0].cpu()),
            # 'masks': {
            #     'true': wandb.Image(true_masks[0].float().cpu()),
            #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
            # },
            'step': global_step,
            'epoch': epoch,
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
        plt.savefig(dir_logs + 'Training Process for lr-{}.png'.format(lr), dpi=600)

        # torch.save(net.state_dict(),
        #            dir_logs + '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}.pkl'.format(dataset, val_dice, epoch_loss / i))
        torch.save(net.state_dict(), dir_logs + 'checkpoint_epoch{}.pth'.format(epoch + 1))
        print('Spend time: {:.3f}s'.format(time.time() - start_epoch))
        print()


if __name__ == '__main__':
    """
    修改train.py並使用Pytorch Dataset 做為資料讀取的方式
    """
    import pathlib

    epochs, batchsize, scale, gpu = 50, 6, 1, True
    lr = 1e-5
    ft = False
    dataset = 'Rewind'
    # dataset = 'total_split'

    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    model = 'Ringed_Res_Unet'
    current_path = str(pathlib.Path().resolve())
    # dir_img = './data/data_{}/train/tam/'.format(dataset)
    # dir_mask = './data/data_{}/train/mask/'.format(dataset)
    # dir_img = '/media/ian/WD/datasets/total_forge/train_and_test/train/images'
    # dir_mask = '/media/ian/WD/datasets/total_forge/train_and_test/train/masks'
    # dir_img = r'E:\data\train_and_test\train_and_test\train\images'
    # dir_mask = r'E:\data\train_and_test\train_and_test\train\masks'
    # dir_img = r'E:\data\train_and_test\train_and_test\test\images'
    # dir_mask = r'E:\data\train_and_test\train_and_test\test\masks'

    ############################  REWIND DATSET  ##########################################
    # dir_img = r'D:\VTD\video_tampering_dataset\videos\h264_lossless\test_and_train\train\images'
    # dir_mask = r'D:\VTD\video_tampering_dataset\videos\h264_lossless\test_and_train\train\masks'
    dir_img = r'/media/ian/WD/datasets/video tempered dataset/REWIND/video_tampering_dataset/videos/h264_lossless(processed)/test_and_train/train/images'
    dir_mask = r'/media/ian/WD/datasets/video tempered dataset/REWIND/video_tampering_dataset/videos/h264_lossless(processed)/test_and_train/train/masks'

    dir_logs = os.path.join(current_path, 'result', 'logs', dataset, model)
    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)
    if model == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif model == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif model == 'Ringed_Res_Unet':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if ft:
        # fine_tuning_model = './result/logs/{}/{}/test.pkl'.format(dataset, model)
        fine_tuning_model = os.path.join(current_path, 'result', 'logs', dataset, model, 'test.pkl')
        net.load_state_dict(torch.load(fine_tuning_model))
        print('Model loaded from {}'.format(fine_tuning_model))

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              val_percent=0.1,
              save_cp=True,
              gpu=gpu,
              img_scale=scale,
              dataset=dataset,
              dir_logs=dir_logs)
