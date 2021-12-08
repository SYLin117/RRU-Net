import os
import os.path

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import math
import matplotlib.pyplot as plt
import time
import wandb
import torch.backends.cudnn as cudnn
from torch import optim
from eval import eval_net, dice_loss
from utils import *
from transunet import MyTransUNet
from tqdm import tqdm
from u2net_model import U2NET, U2NETP
from unet import FCNs, VGGNet


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
              model_name='no-model-name',
              resume=False,
              resume_id=None,
              latest_epoch=0,
              project_name=None):
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
    experiment = wandb.init(project=project_name.replace("_", "-"), id=id, resume='allow', anonymous='must')
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
    net.apply(init_weights)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
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

            # print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / n_train, loss, time.time() - start_batch))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            experiment.log({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })

            if global_step % 1000 == 0:
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


    def get_model(model):
        net = None
        if model == 'FCN':
            vgg_model = VGGNet()
            net = FCNs(pretrained_net=vgg_model, n_class=1)
        elif model == 'Unet':
            net = Unet(n_channels=3, n_classes=1)
        elif model == 'SRM_Unet':
            net = SRM_Unet(n_channels=3, n_classes=1)
        elif model == 'Res_Unet':
            net = Res_Unet(n_channels=3, n_classes=1)
        elif model == 'SRM_Res_Unet':
            net = SRM_Res_Unet(n_channels=3, n_classes=1)
        elif model == 'Ringed_Res_Unet':
            net = Ringed_Res_Unet(n_channels=3, n_classes=1)
        elif model == 'SRM_Ringed_Res_Unet':
            net = SRM_Ringed_Res_Unet(n_channels=3, n_classes=1)
        elif model == 'U2Net':
            net = U2NET(in_ch=3, out_ch=1)
        elif model == 'U2NetP':
            net = U2NETP(in_ch=3, out_ch=1)
        elif model == 'MyTransUnet':
            net = MyTransUNet(in_channels=3, classes=1, img_dim=resize[0])
        elif model == 'MyTransUnet2':
            net = MyTransUNet2(in_channels=3, classes=1, img_dim=resize[0])
        else:
            raise Exception("model not implements.")
        assert net != None
        return net
    def get_dataset(dataset_name):
        DATASETS_DIR = get_dataset_root()
        if dataset_name == 'superlarge_cm':
            dir_img = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_cm', 'A', 'train')
            dir_mask = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_cm', 'B', 'train')
            train_dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=resize)
            dir_img_val = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_cm', 'A', 'val')
            dir_mask_val = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_cm', 'B', 'val')
            val_dataset = ForgeDataset(dir_img_val, dir_mask_val, 1, mask_suffix='', resize=resize)
        elif dataset_name == 'superlarge_sp':
            dir_img = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_sp', 'A', 'train')
            dir_mask = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_sp', 'B', 'train')
            train_dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=resize)
            dir_img_val = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_sp', 'A', 'val')
            dir_mask_val = DATASETS_DIR.joinpath('COCO', 'coco2017_superlarge_sp', 'B', 'val')
            val_dataset = ForgeDataset(dir_img_val, dir_mask_val, 1, mask_suffix='', resize=resize)
        elif dataset_name == 'new_sp':
            dir_img = DATASETS_DIR.joinpath('COCO', 'coco2017_new_sp', 'A', 'train')
            dir_mask = DATASETS_DIR.joinpath('COCO', 'coco2017_new_sp', 'B', 'train')
            train_dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=resize)
            dir_img_val = DATASETS_DIR.joinpath('COCO', 'coco2017_new_sp', 'A', 'val')
            dir_mask_val = DATASETS_DIR.joinpath('COCO', 'coco2017_new_sp', 'B', 'val')
            val_dataset = ForgeDataset(dir_img_val, dir_mask_val, 1, mask_suffix='', resize=resize)
        elif dataset_name == 'new_cm':
            dir_img = DATASETS_DIR.joinpath('COCO', 'coco2017_new_cm', 'A', 'train')
            dir_mask = DATASETS_DIR.joinpath('COCO', 'coco2017_new_cm', 'B', 'train')
            train_dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=resize)
            dir_img_val = DATASETS_DIR.joinpath('COCO', 'coco2017_new_cm', 'A', 'val')
            dir_mask_val = DATASETS_DIR.joinpath('COCO', 'coco2017_new_cm', 'B', 'val')
            val_dataset = ForgeDataset(dir_img_val, dir_mask_val, 1, mask_suffix='', resize=resize)
        elif dataset_name == 'casia2':
            dir_img = DATASETS_DIR.joinpath('CASIA2', 'split', 'train', 'images')
            dir_mask = DATASETS_DIR.joinpath('CASIA2', 'split', 'train', 'masks')
            train_dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=resize)
            val_dataset = None
        else:
            raise Exception("dataset not include")
        return train_dataset, val_dataset



    import pathlib

    epochs, batchsize, scale, gpu = 50, 2, 1, True
    lr = 1e-5
    ft = False
    dataset_name1 = 'new_cm'
    dataset_name2 = 'new_sp'
    model1 = 'SRM_Ringed_Res_Unet'
    model2 = 'UNet'
    CURRENT_PATH = str(pathlib.Path().resolve())
    resize = (300, 300)

    dir_logs1 = os.path.join(CURRENT_PATH, 'result', 'logs', dataset_name1, model1)
    if not os.path.exists(dir_logs1):
        os.makedirs(dir_logs1)
    dir_logs2 = os.path.join(CURRENT_PATH, 'result', 'logs', dataset_name2, model2)
    if not os.path.exists(dir_logs2):
        os.makedirs(dir_logs2)

    net1 = get_model(model1)
    net2 = get_model(model2)

    pretrained_model, _ = find_latest_epoch(dir_logs1)
    net1.load_state_dict(torch.load(pretrained_model))

    pretrained_model, _ = find_latest_epoch(dir_logs2)
    net2.load_state_dict(torch.load(pretrained_model))



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
              resize=resize,
              train_dataset=train_dataset,
              val_dataset=val_dataset,
              dir_logs=dir_logs,
              model_name=model,
              resume=False,
              resume_id=id,
              latest_epoch=latest_epoch,
              project_name=dataset_name)
