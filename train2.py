import torch.backends.cudnn as cudnn
from torch import optim

from eval import eval_net
from unet.unet_model import *
from utils import *

import matplotlib.pyplot as plt
import time


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-2,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1,
              dataset=None):
    # training images are square
    # ids = split_ids(get_ids(dir_img))
    # iddataset = split_train_val(ids, val_percent)

    dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=(300, 300))
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
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
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=0)
    criterion = nn.BCELoss()

    Train_loss = []
    Valida_dice = []
    EPOCH = []

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
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / n_train, loss, time.time() - start_batch))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        net.eval()

        val_dice = eval_net(net, val_loader, gpu)
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))

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

        torch.save(net.state_dict(),
                   dir_logs + '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}.pkl'.format(dataset, val_dice, epoch_loss / i))
        print('Spend time: {:.3f}s'.format(time.time() - start_epoch))
        print()


if __name__ == '__main__':
    epochs, batchsize, scale, gpu = 50, 6, 1, False
    lr = 1e-3
    ft = False
    dataset = 'Total'
    # dataset = 'total_split'

    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    model = 'Ringed_Res_Unet'

    # dir_img = './data/data_{}/train/tam/'.format(dataset)
    # dir_mask = './data/data_{}/train/mask/'.format(dataset)
    dir_img = '/media/ian/WD/datasets/total_forge/train_and_test/train/images'
    dir_mask = '/media/ian/WD/datasets/total_forge/train_and_test/train/masks'
    dir_logs = './result/logs/{}/{}/'.format(dataset, model)

    if model == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif model == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif model == 'Ringed_Res_Unet':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if ft:
        fine_tuning_model = './result/logs/{}/{}/test.pkl'.format(dataset, model)
        net.load_state_dict(torch.load(fine_tuning_model))
        print('Model loaded from {}'.format(fine_tuning_model))

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              gpu=gpu,
              img_scale=scale,
              dataset=dataset)
