import os
import cv2
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from torchvision.models import resnet50, vgg16, efficientnet_b2

from sklearn.model_selection import train_test_split

from PIL import Image

import matplotlib.pyplot as plt
from tqdm import tqdm
import pathlib
import wandb
from self_attention_cv import ViT
from utils import get_dataset_root, find_latest_epoch

DATASET_NAME = 'large_cm_sp'
DATASETS_DIR = get_dataset_root()
DIR_TRAIN = os.path.join(DATASETS_DIR, 'COCO', DATASET_NAME, 'train')
DIR_TEST = os.path.join(DATASETS_DIR, 'COCO', DATASET_NAME, 'test')
CURRENT_PATH = str(pathlib.Path().resolve())
MODEL_NAME = 'EFFICIENTNET_B2'
DIR_LOGS = os.path.join(CURRENT_PATH, 'result', 'logs', 'large_cm_sp', MODEL_NAME)
if not os.path.exists(DIR_LOGS):
    os.makedirs(DIR_LOGS)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ForgeDataset(Dataset):

    def __init__(self, imgs, class_to_int, mode="train", img_size=300, transforms=None):

        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        self.resize = pair(img_size)

    def __getitem__(self, idx):

        image_name = self.imgs[idx]

        ### Reading, converting and normalizing image
        # img = cv2.imread(DIR_TRAIN + image_name, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (224,224))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img /= 255.
        img = Image.open(os.path.join(DIR_TRAIN, image_name))
        img = img.resize(self.resize)

        if self.mode == "train" or self.mode == "val":

            ### Preparing class label
            label = self.class_to_int[image_name.split("_")[0]]
            label = torch.tensor(label, dtype=torch.float32)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label
            # return {
            #     'image': img,
            #     'label': label
            # }

        elif self.mode == "test":

            ### Apply Transforms on image
            img = self.transforms(img)

    def __len__(self):
        return len(self.imgs)


def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        # T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize((0, 0, 0), (1, 1, 1))
    ])


def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0), (1, 1, 1))
    ])


def accuracy(preds, trues):
    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]

    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]

    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)

    return (acc * 100)


if __name__ == "__main__":
    def train_one_epoch(train_data_loader):
        ### Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        ###Iterating over data loader
        for index, batch in enumerate(tqdm(train_data_loader)):
            # Loading images and labels to device
            images = batch[0]
            labels = batch[1]
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.reshape((labels.shape[0], 1))  # [N, 1] - to match with preds shape

            # Reseting Gradients
            optimizer.zero_grad()

            # Forward
            preds = model(images)

            # Calculating Loss
            _loss = criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = accuracy(preds, labels)
            epoch_acc.append(acc)

            # Backward
            _loss.backward()
            optimizer.step()

            global global_step, experiment, epoch
            global_step += 1
            if global_step % 100 == 0:
                experiment.log({
                    'train loss': loss,
                    'step': global_step,
                    'epoch': epoch
                })

        ###Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        ###Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        ###Storing results to logs
        train_logs["loss"].append(epoch_loss)
        train_logs["accuracy"].append(epoch_acc)
        train_logs["time"].append(total_time)

        return epoch_loss, epoch_acc, total_time


    def val_one_epoch(val_data_loader, best_val_acc):
        ### Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()

        ###Iterating over data loader
        for images, labels in val_data_loader:
            # Loading images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.reshape((labels.shape[0], 1))  # [N, 1] - to match with preds shape

            # Forward
            preds = model(images)

            # Calculating Loss
            _loss = criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = accuracy(preds, labels)
            epoch_acc.append(acc)

        ###Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time

        ###Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)

        ###Storing results to logs
        val_logs["loss"].append(epoch_loss)
        val_logs["accuracy"].append(epoch_acc)
        val_logs["time"].append(total_time)

        ###Saving best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            # torch.save(model.state_dict(), "resnet50_best.pth")
            torch.save(model.state_dict(), os.path.join(DIR_LOGS, 'checkpoint_epoch{}.pth'.format(epoch + 1)))

        return epoch_loss, epoch_acc, total_time, best_val_acc


    imgs = os.listdir(DIR_TRAIN)
    test_imgs = os.listdir(DIR_TEST)

    cm_list = [img for img in imgs if img.split("_")[0] == "cm"]
    sp_list = [img for img in imgs if img.split("_")[0] == "sp"]

    print("No of CM Images: ", len(cm_list))
    print("No of SP Images: ", len(sp_list))

    class_to_int = {"cm": 0, "sp": 1}
    int_to_class = {0: "cm", 1: "sp"}

    train_imgs, val_imgs = train_test_split(imgs, test_size=0.25)
    train_dataset = ForgeDataset(train_imgs, class_to_int, mode="train", img_size=300, transforms=get_train_transform())
    val_dataset = ForgeDataset(val_imgs, class_to_int, mode="val", img_size=300, transforms=get_val_transform())
    test_dataset = ForgeDataset(test_imgs, class_to_int, mode="test", img_size=300, transforms=get_val_transform())

    train_data_loader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=16,
        shuffle=True
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        num_workers=4,
        batch_size=8,
        shuffle=True
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=8,
        shuffle=True
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # for images, labels in train_data_loader:
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.imshow(make_grid(images, 4).permute(1, 2, 0))
    #     break

    fine_tune = False
    start_epoch = 0
    if MODEL_NAME == 'resnet50':
        model = resnet50(pretrained=True)
        # Modifying Head - classifier
        model.fc = nn.Sequential(
            nn.Linear(2048, 1, bias=True),
            nn.Sigmoid()
        )
    elif MODEL_NAME == "ViT16":
        model = ViT(img_dim=512, patch_dim=16, num_classes=1, classification=True)
        model.mlp_head = nn.Sequential(
            nn.Linear(model.dim, 1, bias=True),
            nn.Sigmoid()
        )
    elif MODEL_NAME == "EFFICIENTNET_B2":
        model = efficientnet_b2(pretrained=True, progress=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=1, bias=True),
            nn.Sigmoid(),
        )
    else:
        raise RuntimeError('model not include')
    if fine_tune:
        fine_tuning_model, latest_epoch = find_latest_epoch(
            os.path.join(CURRENT_PATH, 'result', 'logs', DATASET_NAME, MODEL_NAME, ))
        model.load_state_dict(torch.load(fine_tuning_model))
        start_epoch = latest_epoch
    ## setting up wandb
    wandb_id = wandb.util.generate_id()
    experiment = wandb.init(project='Classify-cm-sp', id=wandb_id, resume='allow', anonymous='must')
    experiment.name = MODEL_NAME
    print('model name: ', MODEL_NAME)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Loss Function
    criterion = nn.BCELoss()

    # Logs - Helpful for plotting after training finishes
    train_logs = {"loss": [], "accuracy": [], "time": []}
    val_logs = {"loss": [], "accuracy": [], "time": []}

    # Loading model to device
    model.to(device)

    # No of epochs
    epochs = 50

    global_step = 0
    best_val_acc = 0
    for epoch in range(start_epoch, epochs):
        ###Training
        train_loss, train_acc, train_time = train_one_epoch(train_data_loader)

        # Print Epoch Details
        print("\nTraining")
        print("Epoch {}".format(epoch + 1))
        print("Loss : {}".format(round(train_loss, 4)))
        print("Acc : {}".format(round(train_acc, 4)))
        print("Time : {}".format(round(train_time, 4)))

        ###Validation
        val_loss, val_acc, val_time, best_val_acc = val_one_epoch(val_data_loader, best_val_acc)

        # Print Epoch Details
        print("\nValidating")
        print("Epoch {}".format(epoch + 1))
        print("Loss : {}".format(round(val_loss, 4)))
        print("Acc : {}".format(round(val_acc, 4)))
        print("Time : {}".format(round(val_time, 4)))

        experiment.log({
            'train epoch loss': train_loss,
            'validation epoch loss': val_loss,
            'validation epoch acc': val_acc,
            'epoch': epoch
        })

    ### Plotting Results

    # Loss
    plt.title("Loss")
    plt.plot(np.arange(start_epoch, epochs, 1), train_logs["loss"], color='blue')
    plt.plot(np.arange(start_epoch, epochs, 1), val_logs["loss"], color='yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # Accuracy
    plt.title("Accuracy")
    # plt.plot(np.arange(1, 11, 1), train_logs["accuracy"], color='blue')
    # plt.plot(np.arange(1, 11, 1), val_logs["accuracy"], color='yellow')
    plt.plot(np.arange(start_epoch, epochs, 1), train_logs["accuracy"], color='blue')
    plt.plot(np.arange(start_epoch, epochs, 1), val_logs["accuracy"], color='yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
