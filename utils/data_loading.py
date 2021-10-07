import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', resize=None,
                 transform=None):
        """
        resize: tuple, eg:(128,128)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        self.resize = resize

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(tmp, scale, is_mask, resize=None):
        img = tmp.copy()
        img_format = img.format
        w, h = img.size
        if resize is None:
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            img = img.resize((newW, newH))
        else:
            img = img.resize(resize)

        if is_mask:
            if img_format == 'PNG' or img_format == 'TIFF' or img_format == 'JPEG':  # 將bool array轉為0,1 array
                # img = img.convert('1')
                img_ndarray = np.asarray(img.convert('1')) * 1
            else:
                img_ndarray = np.asarray(img)
        else:
            img_ndarray = np.asarray(img)

        # only 2 dim
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def preprocess2(tmp, scale, is_mask, resize=None):
        img = copy.deepcopy(tmp)
        if is_mask:
            w, h = img.shape
        else:
            w, h, c = img.shape
        if resize is None:
            newW, newH = int(scale * w), int(scale * h)
            assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
            img = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)

        if is_mask:
            img = np.where(img >= 1, 1, 0)

        # only 2 dim
        if img.ndim == 2 and not is_mask:
            img = img[np.newaxis, ...]
        elif not is_mask:
            img = img.transpose((2, 0, 1))
        if not is_mask:
            img = img / 255
        return img

    @staticmethod
    def load(filename, is_mask=False):
        ext = splitext(filename)[1]
        try:
            if ext in ['.npz', '.npy']:
                return Image.fromarray(np.load(filename))
            elif ext in ['.pt', '.pth']:
                return Image.fromarray(torch.load(filename).numpy())
            elif ext in ['.gif']:
                if not is_mask:
                    return np.asarray(Image.open(filename))
                else:
                    return np.asarray(Image.open(filename).convert('L'))
            else:
                if not is_mask:
                    tmp = cv2.imread(str(filename), cv2.IMREAD_COLOR)
                    # cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
                    tmp = tmp[:, :, [2, 1, 0]]
                    return tmp
                else:
                    return cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        except:
            logging.error("image {} loading encounter error".format(splitext(filename)[0]))

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        # print("mask dir: {}".format(self.masks_dir))
        # print("image dir: {}".format(self.images_dir))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        pil_img = self.load(img_file[0])
        pil_mask = self.load(mask_file[0], is_mask=True)
        img = None
        mask = None
        try:
            img = self.preprocess2(pil_img, self.scale, is_mask=False, resize=self.resize, )
            mask = self.preprocess2(pil_mask, self.scale, is_mask=True, resize=self.resize, )
        except:
            print("encounter error during preprocess : {}".format(name))
            raise RuntimeError

        rtn_image = torch.as_tensor(copy.deepcopy(img)).float().contiguous()
        rtn_mask = torch.as_tensor(copy.deepcopy(mask)).long().contiguous()

        diff = np.setdiff1d(np.unique(mask), np.array([0, 1]))
        assert diff.size == 0, "value not match, diff:{}".format(diff)
        return {
            'image': rtn_image,
            'mask': rtn_mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, resize=False):
        if not resize:
            super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
        else:
            super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask', resize=(256, 256))


class ForgeDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1, mask_suffix='', resize=(256, 256)):
        super().__init__(images_dir, masks_dir, scale, mask_suffix=mask_suffix, resize=resize)


if __name__ == "__main__":
    print("___main___")
    # ## test dataset valid
    # DATASETS_DIR = Path(r'I:\datasets')
    # # DATASETS_DIR = Path('/media/ian/WD/datasets')
    # dir_img = DATASETS_DIR.joinpath('big_coco_forge', 'images')
    # dir_mask = DATASETS_DIR.joinpath('big_coco_forge', 'masks')
    # dir_checkpoint = Path(r'.\checkpoints_big_coco_forge(UNet)')
    # dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='', resize=(256, 256))
    # loader_args = dict(batch_size=4, num_workers=4, pin_memory=True)
    # dataloader = DataLoader(dataset, shuffle=True, **loader_args)
    # dataiter = iter(dataloader)
    # idx = 0
    # while True:
    #     try:
    #         features, labels = next(dataiter)
    #         print('number: {}'.format(idx))
    #         idx += 1
    #     except StopIteration:
    #         break
    #################################################################################################################
    # masks = list(Path(dir_mask).glob("*.*"))
    # print(len(masks))
    # print(os.system('who am i'))
    # output = os.popen('whoami')
    #################################################################################################################
    # f = open("/media/ian/WD/datasets/carvana-image-masking/train_masks/aa1", "r")
    # print(f.read())
    #################################################################################################################
    # print(output.read())
    # dirname = os.path.dirname(__file__)
    # FILE_PATH3 = '/tmp/f1eb080c7182_15_mask.gif'
    # FILE_PATH = '/media/ian/WD/datasets/carvana-image-masking/train_masks/fff9b3a5373f_16_mask.gif'
    # FILE_PATH2 = '/media/ian/WD/datasets/carvana-image-masking/train/f1eb080c7182_15.gif'
    # print(os.path.exists(os.path.join(dirname, FILE_PATH)))
    # img = cv2.imread(FILE_PATH, cv2.IMREAD_COLOR)
    # img = Image.open("/tmp/f1eb080c7182_15_mask.gif")
    # print(img)
    #################################################################################################################
    test_img = r'I:\datasets\big_coco_forge\images\000000.jpg'
    pil_img = np.asarray(Image.open(test_img))
    cv_img = cv2.imread(test_img)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv_img[:, :, [2, 1, 0]]
    unique, counts = np.unique((pil_img == cv_img), return_counts=True)
    print(dict(zip(unique, counts)))
