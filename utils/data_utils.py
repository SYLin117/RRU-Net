import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import re
import imutils
from shutil import copyfile
from random import randint, shuffle, random
from tqdm import tqdm
from PIL import Image
from os import path
import torch
import xlsxwriter
import shutil

from utils import get_dataset_root

DATASET_ROOT = get_dataset_root()


def split_train_test(orig_folder, gt_folder, new_folder):
    TRAIN = 0
    TEST = 1
    train_orig_folder = os.path.join(new_folder, "train")  # 新的train資料夾
    test_orig_folder = os.path.join(new_folder, "test")  # 新的test資料夾
    train_image_folder = os.path.join(train_orig_folder, "images")
    train_mask_folder = os.path.join(train_orig_folder, "masks")
    test_image_folder = os.path.join(test_orig_folder, "images")
    test_mask_folder = os.path.join(test_orig_folder, "masks")

    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)
    if not os.path.exists(train_mask_folder):
        os.makedirs(train_mask_folder)
    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder)
    if not os.path.exists(test_mask_folder):
        os.makedirs(test_mask_folder)

    def get_type():
        tmp = random()
        if tmp >= 0.10:
            return TRAIN
        else:
            return TEST

    img_list = [f for f in os.listdir(orig_folder)]
    mask_list = [f for f in os.listdir(gt_folder)]
    img_list.sort()
    mask_list.sort()
    assert len(img_list) == len(mask_list), "image and ground truth not match"
    img_pbar = tqdm(img_list)
    for img in img_pbar:
        img_filename = os.path.splitext(os.path.basename(img))[0]
        matching_mask = [s for s in mask_list if img_filename in s]
        assert len(matching_mask) == 1, "should got 1 and only one match, but got:{}".format(matching_mask)
        type = get_type()
        if type == TRAIN:
            copyfile(os.path.join(orig_folder, img), os.path.join(train_image_folder, img))
            copyfile(os.path.join(gt_folder, matching_mask[0]), os.path.join(train_mask_folder, matching_mask[0]))
        elif type == TEST:
            copyfile(os.path.join(orig_folder, img), os.path.join(test_image_folder, img))
            copyfile(os.path.join(gt_folder, matching_mask[0]), os.path.join(test_mask_folder, matching_mask[0]))


def make_video_to_image(folder):
    videos = [f for f in os.listdir(folder) if re.match(r'[0-9]+_forged\.mp4', f)]
    no_regex = re.compile(r'([0-9]+)_forged\.mp4')
    for i in range(len(videos)):
        video_no = no_regex.match(videos[i]).groups()[0]
        orig_list = []
        original_video_cap = cv2.VideoCapture(os.path.join(folder, videos[i]))
        while (original_video_cap.isOpened()):
            ret, frame = original_video_cap.read()
            if ret == False:
                break
            orig_list.append(frame)
        original_video_cap.release()

        original_dir = os.path.join(folder, 'forged_{}'.format(f'{int(video_no):02}'))
        if not os.path.exists(original_dir):
            os.makedirs(original_dir)
        for i in range(len(orig_list)):
            cv2.imwrite(os.path.join(original_dir, '{}.jpg'.format(i)), orig_list[i])


def combine_video_image(folder):
    def extract_number(file_list):
        # file_list = [f for f in os.listdir(dir)]
        s = re.findall("([0-9]+)\.jpg$", file_list)
        return (int(s[0]) if s else -1, file_list)

    NEW_FORGE_DIR = os.path.join(folder, 'final_forge')
    NEW_MASK_DIR = os.path.join(folder, 'final_mask')
    if not os.path.exists(NEW_FORGE_DIR):
        os.makedirs(NEW_FORGE_DIR)
    if not os.path.exists(NEW_MASK_DIR):
        os.makedirs(NEW_MASK_DIR)
    file_re = re.compile(r'([0-9]+)\.jpg')
    forge_folder_re = re.compile(r'forged_([0-9]+)')
    mask_folder_re = re.compile(r'mask_([0-9]+)')
    forge_folders = [f for f in os.listdir(folder) if re.match(r'forged_[0-9]+', f)]
    mask_folders = [f for f in os.listdir(folder) if re.match(r'mask_[0-9]+', f)]
    forge_folders.sort()
    mask_folders.sort()

    idx = 0
    for i in range(len(forge_folders)):  # loop folders
        forge_folder = forge_folders[i]
        mask_folder = mask_folders[i]
        forge_folder_no = forge_folder_re.match(forge_folder).groups()[0]
        mask_folder_no = mask_folder_re.match(mask_folder).groups()[0]
        assert forge_folder_no == mask_folder_no, "資料夾編號不同"  # 確認folder編號相同
        forge_list = [f for f in os.listdir(os.path.join(folder, forge_folder))]
        mask_list = [f for f in os.listdir(os.path.join(folder, mask_folder))]
        forge_list.sort()
        mask_list.sort()
        assert len(forge_list) == len(mask_list), "檔案數不同"

        new_forge_list = [f for f in os.listdir(NEW_FORGE_DIR)]
        new_mask_list = [f for f in os.listdir(NEW_MASK_DIR)]
        assert len(new_forge_list) == len(new_mask_list), "新資料夾檔案數不同"
        new_forge_list.sort()
        new_mask_list.sort()
        if not new_forge_list:  # empty
            count = 0
        else:
            max_filename = max(new_forge_list, key=extract_number)
            tmp1 = os.path.splitext(os.path.basename(max_filename))[0]
            tmp2 = os.path.splitext(os.path.basename(max_filename))[1]
            count = int(tmp1) + 1
        for j in range(len(forge_list)):
            forge_file = forge_list[j]
            mask_file = mask_list[j]
            forge_no = file_re.match(forge_file).groups()[0]
            mask_no = file_re.match(mask_file).groups()[0]
            assert int(forge_no) == int(mask_no), "檔案編號不同"

            copyfile(os.path.join(folder, forge_folder, forge_file),
                     os.path.join(NEW_FORGE_DIR, "{}{}".format(f'{idx + count:06}', '.jpg')))
            copyfile(os.path.join(folder, mask_folder, mask_file),
                     os.path.join(NEW_MASK_DIR, "{}{}".format(f'{idx + count:06}', '.jpg')))
            idx += 1


def create_video_mask(orig_path, forge_path, video_index):
    original_video_cap = cv2.VideoCapture(orig_path)
    forge_video_cap = cv2.VideoCapture(forge_path)

    orig_fps = original_video_cap.get(cv2.CAP_PROP_FPS)
    forge_fps = forge_video_cap.get(cv2.CAP_PROP_FPS)

    width = int(original_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(original_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(original_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert orig_fps == forge_fps, \
        '{} got {} frames and {} got {} frames'.format(orig_path, orig_fps, forge_path, forge_fps)

    orig_list = []
    forge_list = []
    diff_list = []
    mask_list = []
    idx = 1
    while (original_video_cap.isOpened()):
        ret, frame = original_video_cap.read()
        if ret == False:
            break
        orig_list.append(frame)
    while (forge_video_cap.isOpened()):
        ret, frame = forge_video_cap.read()
        if ret == False:
            break
        forge_list.append(frame)
    original_video_cap.release()
    forge_video_cap.release()
    for i in range(len(orig_list)):
        new = forge_list[i].copy()
        diff = orig_list[i].copy()
        cv2.absdiff(orig_list[i], forge_list[i], diff)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # for i in range(0, 3):
        #     dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
        (T, thresh) = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)
        mask_list.append(thresh)

        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            # fit a bounding box to the contour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(new, (x, y), (x + w, y + h), (0, 255, 0), 2)
        diff_list.append(new)

    out = cv2.VideoWriter('{}.mp4'.format(f'{video_index:02}'), cv2.VideoWriter_fourcc(*'mp4v'), orig_fps,
                          (width, height))
    for i in range(len(diff_list)):
        out.write(diff_list[i])
    out.release()

    mask_dir = os.path.relpath(r'mask_{}'.format(f'{video_index:02}'))
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    for i in range(len(mask_list)):
        cv2.imwrite(os.path.join(mask_dir, '{}.jpg'.format(i)), mask_list[i])


def create_REWIND_mask():
    VIDEO_DIR = r'E:\VTD\video_tampering_dataset\videos\h264_lossless'
    original = [f for f in os.listdir(VIDEO_DIR) if re.search(r'(_original)\.mp4$', f)]
    forged = [f for f in os.listdir(VIDEO_DIR) if re.search(r'(_forged)\.mp4$', f)]

    for i in range(len(original)):
        create_video_mask(os.path.join(VIDEO_DIR, original[i]), os.path.join(VIDEO_DIR, forged[i]), video_index=i + 1)


def casia_v1_mask():
    AU_DIR = r'/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Au'
    CM_DIR = r'/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM'
    SP_DIR = r'/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/Sp'
    CM_MASK_DIR = r'/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM_mask'
    SP_MASK_DIR = r'/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/Sp_mask'
    if not os.path.exists(CM_MASK_DIR):
        os.mkdir(CM_MASK_DIR)
    if not os.path.exists(SP_MASK_DIR):
        os.mkdir(SP_MASK_DIR)
    cm_file_list = [f for f in os.listdir(CM_DIR)]
    sp_file_list = [f for f in os.listdir(SP_DIR)]
    print(len(cm_file_list))  # 459
    print(len(sp_file_list))  # 462

    regex = re.compile("([a-zA-Z]+)([0-9]+)")
    for i in range(len(cm_file_list)):
        f = cm_file_list[i]
        tp_file = cv2.imread(os.path.join(CM_DIR, f))
        filename = os.path.splitext(os.path.basename(f))[0]
        filename_split = filename.split('_')
        target_file = filename_split[5]
        res = regex.match(target_file).groups()
        au_file_path = os.path.join(AU_DIR, "Au_{}_{}.jpg".format(res[0], res[1]))
        au_file = cv2.imread(au_file_path)

        diff = tp_file.copy()
        try:
            cv2.absdiff(au_file, tp_file, diff)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            (T, thresh) = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(CM_MASK_DIR, '{}.jpg'.format(filename)), thresh)
        except:
            print("au_file:{} and tempered file:{} got error".format(au_file_path, f))
    for i in range(len(sp_file_list)):
        f = sp_file_list[i]
        tp_file = cv2.imread(os.path.join(SP_DIR, f))
        filename = os.path.splitext(os.path.basename(f))[0]
        filename_split = filename.split('_')
        target_file = filename_split[4]
        res = regex.match(target_file).groups()
        au_file_path = os.path.join(AU_DIR, "Au_{}_{}.jpg".format(res[0], res[1]))
        au_file = cv2.imread(au_file_path)

        diff = tp_file.copy()
        try:
            cv2.absdiff(au_file, tp_file, diff)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            (T, thresh) = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(SP_MASK_DIR, '{}.jpg'.format(filename)), thresh)
        except:
            print("au_file:{} and tempered file:{} got error".format(au_file_path, f))


def split_CASIA2():
    DIR = r"/media/ian/WD/datasets/CASIA/casia2groundtruth-master/CASIA2.0_Groundtruth"
    CM_DIR = r"/media/ian/WD/datasets/CASIA/casia2groundtruth-master/CM_MASK"
    SP_DIR = r"/media/ian/WD/datasets/CASIA/casia2groundtruth-master/SP_MASK"
    if not os.path.exists(CM_DIR):
        os.mkdir(CM_DIR)
    if not os.path.exists(SP_DIR):
        os.mkdir(SP_DIR)
    file_list = [f for f in os.listdir(DIR)]
    for file in file_list:
        filename = os.path.splitext(os.path.basename(file))[0]
        filename_split = filename.split('_')
        type = filename_split[1]
        if type == "D":
            copyfile(os.path.join(DIR, file), os.path.join(SP_DIR, file))
        elif type == "S":
            copyfile(os.path.join(DIR, file), os.path.join(CM_DIR, file))
        else:
            print("got type {}".format(type))


def get_COMOFOD():
    mask_regex = re.compile("([0-9]+)_B\.png")
    DIR = r"/media/ian/WD/datasets/CoMoFoD/CoMoFoD_small_v2"
    MASK_DIR = r"/media/ian/WD/datasets/CoMoFoD/MASK"
    FORGE_DIR = r"/media/ian/WD/datasets/CoMoFoD/FORGE"
    if not os.path.exists(MASK_DIR):
        os.mkdir(MASK_DIR)
    if not os.path.exists(FORGE_DIR):
        os.mkdir(FORGE_DIR)
    mask_files = [f for f in os.listdir(DIR) if re.match(r'([0-9]+)_B\.png', f)]
    forge_files = [f for f in os.listdir(DIR) if re.match(r'([0-9]+)_F\.png', f)]
    assert len(mask_files) == len(forge_files), "files number not match"
    for file in mask_files:
        file_no = mask_regex.match(file).groups()
        copyfile(os.path.join(DIR, file), os.path.join(MASK_DIR, "{}_F.png".format(file_no[0])))
    for file in forge_files:
        copyfile(os.path.join(DIR, file), os.path.join(FORGE_DIR, file))


def get_COVERAGE():
    mask_regex = re.compile("([0-9]+)forged\.tif")
    img_regex = re.compile("([0-9]+)t\.tif")
    MASK_DIR = r"/media/ian/WD/datasets/COVERAGE/mask"
    FORGE_DIR = r"/media/ian/WD/datasets/COVERAGE/image"
    NEW_MASK_DIR = r"/media/ian/WD/datasets/COVERAGE/new_mask"
    NEW_FORGE_DIR = r"/media/ian/WD/datasets/COVERAGE/new_forge"
    if not os.path.exists(NEW_MASK_DIR):
        os.mkdir(NEW_MASK_DIR)
    if not os.path.exists(NEW_FORGE_DIR):
        os.mkdir(NEW_FORGE_DIR)
    forge_files = [f for f in os.listdir(FORGE_DIR) if re.match(r'[0-9]+t\.tif', f)]
    mask_files = [f for f in os.listdir(MASK_DIR) if re.match(r'[0-9]+forged\.tif', f)]
    assert len(mask_files) == len(forge_files), "files number not match"
    for file in mask_files:
        file_no = mask_regex.match(file).groups()
        copyfile(os.path.join(MASK_DIR, file), os.path.join(NEW_MASK_DIR, "{}.tif".format(file_no[0])))
    for file in forge_files:
        file_no = img_regex.match(file).groups()
        copyfile(os.path.join(FORGE_DIR, file), os.path.join(NEW_FORGE_DIR, "{}.tif".format(file_no[0])))


def combine_multiple_dataset():
    def extract_number(file_list):
        # file_list = [f for f in os.listdir(dir)]
        s = re.findall("([0-9]+).*$", file_list)
        return (int(s[0]) if s else -1, file_list)

    # excel_file = open(r'/media/ian/WD/datasets/forge_combine/forge_combine.xlsx', )
    workbook = xlsxwriter.Workbook(r'/media/ian/WD/datasets/forge_combine/forge_combine.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'file')
    worksheet.write('B1', 'original_path')
    worksheet.write('C1', 'class')
    worksheet.write('D1', 'predict')
    xml_idx = 2

    TYPE = ["SP", "CM"]
    SP_DATASET_NAME_LIST = ["CASIA1", "CASIA2", "COCO"]
    SP_IMG_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/Sp",
                       r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/SP_Tp",
                       r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_sp/train2017"]
    SP_MASK_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/Sp_mask",
                        r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/SP_MASK",
                        r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_sp/train2017_mask"]

    CM_DATASET_NAME_LIST = ["CASIA1", "CASIA2", "CoMoFoD", "COVERAGE", "COCO"]
    CM_IMG_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM",
                       r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/CM_Tp",
                       r"/media/ian/WD/datasets/CoMoFoD/FORGE",
                       r"/media/ian/WD/datasets/COVERAGE/new_forge",
                       r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_cm/train2017"]
    CM_MASK_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM_mask",
                        r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/CM_MASK",
                        r"/media/ian/WD/datasets/CoMoFoD/MASK",
                        r"/media/ian/WD/datasets/COVERAGE/new_mask",
                        r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_cm/train2017_mask"]
    FINAL_IMG_DIR = r"/media/ian/WD/datasets/forge_combine/forge"
    FINAL_MSK_DIR = r"/media/ian/WD/datasets/forge_combine/mask"
    ##############################################################################################################################
    # SP_IMG_DIR_LIST = [r"/media/ian/WD/datasets/COCO/coco2017_big_forge_spliced/train2017", ]
    # SP_MASK_DIR_LIST = [r"/media/ian/WD/datasets/COCO/coco2017_big_forge_spliced/train2017_mask", ]
    # CM_IMG_DIR_LIST = [r"/media/ian/WD/datasets/COCO/coco2017_big_forge_copymove/train2017", ]
    # CM_MASK_DIR_LIST = [r"/media/ian/WD/datasets/COCO/coco2017_big_forge_copymove/train2017_mask", ]
    # FINAL_IMG_DIR = r"/media/ian/WD/datasets/big_coco_forge/images"
    # FINAL_MSK_DIR = r"/media/ian/WD/datasets/big_coco_forge/masks"
    ##############################################################################################################################

    if not os.path.exists(FINAL_IMG_DIR):
        os.makedirs(FINAL_IMG_DIR)
    if not os.path.exists(FINAL_MSK_DIR):
        os.makedirs(FINAL_MSK_DIR)
    for type in TYPE:
        if type == "SP":
            for i in range(len(SP_IMG_DIR_LIST)):
                forge_list = [f for f in os.listdir(SP_IMG_DIR_LIST[i])]
                mask_list = [f for f in os.listdir(SP_MASK_DIR_LIST[i])]
                assert len(forge_list) == len(mask_list), "mask and forge images not match"
                forge_list.sort()
                mask_list.sort()
                new_img_list = [f for f in os.listdir(FINAL_IMG_DIR)]
                new_mask_list = [f for f in os.listdir(FINAL_MSK_DIR)]
                new_img_list.sort()
                new_mask_list.sort()

                assert len(new_mask_list) == len(new_img_list), "new mask and image number not match"
                if not new_img_list:  # empty
                    count = 0
                else:
                    max_filename = max(new_img_list, key=extract_number)
                    tmp1 = os.path.splitext(os.path.basename(max_filename))[0]
                    tmp2 = os.path.splitext(os.path.basename(max_filename))[1]
                    count = int(tmp1) + 1
                idx = 0
                for j in tqdm(range(len(forge_list))):
                    img_file = forge_list[j]
                    msk_file = mask_list[j]
                    filename = os.path.splitext(os.path.basename(img_file))[0]
                    file_ext = os.path.splitext(os.path.basename(img_file))[1]
                    img_cv = cv2.imread(os.path.join(SP_IMG_DIR_LIST[i], img_file), cv2.IMREAD_COLOR)
                    mask_cv = cv2.imread(os.path.join(SP_MASK_DIR_LIST[i], msk_file), cv2.IMREAD_GRAYSCALE)
                    _, mask_cv = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)

                    img_h, img_w, _ = img_cv.shape
                    msk_h, msk_w = mask_cv.shape
                    # assert img_h == msk_h and img_w == msk_w, "image {} got mismatch with mask".format(os.path.join(SP_IMG_DIR_LIST[i], img_file))
                    if img_h != msk_h or img_w != msk_w:
                        mask_cv = cv2.resize(mask_cv, (img_w, img_h))

                    cv2.imwrite(os.path.join(FINAL_IMG_DIR, "{}{}".format(f'{idx + count:06}', '.jpg')), img_cv)
                    cv2.imwrite(os.path.join(FINAL_MSK_DIR, "{}{}".format(f'{idx + count:06}', '.png')), mask_cv)

                    worksheet.write('A{}'.format(xml_idx), "{}{}".format(f'{idx + count:06}', '.jpg'))
                    worksheet.write('B{}'.format(xml_idx), os.path.join(SP_IMG_DIR_LIST[i], img_file))
                    worksheet.write('C{}'.format(xml_idx), 'sp')
                    xml_idx += 1
                    # copyfile(os.path.join(SP_IMG_DIR_LIST[i], img_file),
                    #          os.path.join(FINAL_IMG_DIR, "{}{}".format(f'{idx + count:06}', file_ext)))
                    # copyfile(os.path.join(SP_MASK_DIR_LIST[i], msk_file),
                    #          os.path.join(FINAL_MSK_DIR, "{}{}".format(f'{idx + count:06}', file_ext)))
                    idx += 1
        elif type == "CM":
            for i in range(len(CM_IMG_DIR_LIST)):
                forge_list = [f for f in os.listdir(CM_IMG_DIR_LIST[i])]
                mask_list = [f for f in os.listdir(CM_MASK_DIR_LIST[i])]
                assert len(forge_list) == len(mask_list), "mask and forge images not match, {} and {} folders".format(
                    CM_IMG_DIR_LIST[i], CM_MASK_DIR_LIST[i])
                forge_list.sort()
                mask_list.sort()
                new_img_list = [f for f in os.listdir(FINAL_IMG_DIR)]
                new_mask_list = [f for f in os.listdir(FINAL_MSK_DIR)]
                new_img_list.sort()
                new_mask_list.sort()

                assert len(new_mask_list) == len(new_img_list), "new mask and image number not match"
                if not new_img_list:  # empty
                    count = 0
                else:
                    max_filename = max(new_img_list, key=extract_number)
                    tmp1 = os.path.splitext(os.path.basename(max_filename))[0]
                    tmp2 = os.path.splitext(os.path.basename(max_filename))[1]
                    count = int(tmp1) + 1
                idx = 0
                for j in tqdm(range(len(forge_list))):
                    img_file = forge_list[j]
                    msk_file = mask_list[j]
                    filename = os.path.splitext(os.path.basename(img_file))[0]
                    file_ext = os.path.splitext(os.path.basename(img_file))[1]

                    img_cv = cv2.imread(os.path.join(CM_IMG_DIR_LIST[i], img_file), cv2.IMREAD_COLOR)
                    mask_cv = cv2.imread(os.path.join(CM_MASK_DIR_LIST[i], msk_file), cv2.IMREAD_GRAYSCALE)
                    _, mask_cv = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)

                    img_h, img_w, _ = img_cv.shape
                    msk_h, msk_w = mask_cv.shape
                    # assert img_h == msk_h and img_w == msk_w, "image {} got mismatch with mask".format(os.path.join(SP_IMG_DIR_LIST[i], img_file))
                    if img_h != msk_h or img_w != msk_w:
                        mask_cv = cv2.resize(mask_cv, (img_w, img_h))

                    cv2.imwrite(os.path.join(FINAL_IMG_DIR, "{}{}".format(f'{idx + count:06}', '.jpg')), img_cv)
                    cv2.imwrite(os.path.join(FINAL_MSK_DIR, "{}{}".format(f'{idx + count:06}', '.png')), mask_cv)

                    worksheet.write('A{}'.format(xml_idx), "{}{}".format(f'{idx + count:06}', '.jpg'))
                    worksheet.write('B{}'.format(xml_idx), os.path.join(CM_MASK_DIR_LIST[i], img_file))
                    worksheet.write('C{}'.format(xml_idx), 'cm')
                    xml_idx += 1

                    # copyfile(os.path.join(CM_IMG_DIR_LIST[i], img_file),
                    #          os.path.join(FINAL_IMG_DIR, "{}{}".format(f'{idx + count:06}', file_ext)))
                    # copyfile(os.path.join(CM_MASK_DIR_LIST[i], msk_file),
                    #          os.path.join(FINAL_MSK_DIR, "{}{}".format(f'{idx + count:06}', file_ext)))
                    idx += 1
    workbook.close()


def find_all_black_mask(dirs):
    for sub_dir in dirs:
        masks = glob.glob(os.path.join(sub_dir, "*.jpg"))
        for mask in masks:
            mask_cv = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)
            unique_val = np.unique(thresh)
            if len(unique_val) <= 1:
                print("mask: {} got one value".format(mask))
            elif len(unique_val) > 2:
                print("mask: {} got more than two value".format( mask))


def find_img_with_4_channel(dirs):
    for sub_dir in dirs:
        imgs = glob.glob(os.path.join(sub_dir, "*.*"))
        print("\nsearch {}\n".format(sub_dir))
        for img in tqdm(imgs):
            im_cv = cv2.imread(img, cv2.IMREAD_COLOR)
            height, width, channels = im_cv.shape
            if channels != 3:
                print("image: {}, got {} channels".format(img, channels))


def devide_dataset_to_small_patches(images_dir, masks_dir, new_dir):
    def load(filename, is_mask=False):
        ext = path.splitext(filename)[1]
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
            print("image {} loading encounter error".format(path.splitext(filename)[0]))
            raise RuntimeError

    def crop(im, mask, height=128, width=128):
        imgheight, imgwidth = mask.shape
        for i in range(imgheight // height):
            for j in range(imgwidth // width):
                # box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                crop_mask = mask[i * height:(i + 1) * height, j * width:(j + 1) * width]
                unique, counts = np.unique(crop_mask, return_counts=True)
                result = dict(zip(unique, counts))
                try:
                    if (result[255] / (128 * 128)) >= 0 and (result[255] / (128 * 128)) <= 0.75:
                        crop_img = im[i * height:(i + 1) * height, j * width:(j + 1) * width]
                        yield crop_img, crop_mask
                except KeyError as ke:
                    continue  ##when there is no pixel 1

    os.makedirs(path.join(new_dir, "images"))
    os.makedirs(path.join(new_dir, "masks"))
    file_re = re.compile(r'([0-9]+)\..*')
    images = os.listdir(images_dir)
    masks = os.listdir(masks_dir)
    images.sort()
    masks.sort()
    assert len(images) == len(masks)
    count = 0
    for i in range(len(images)):
        img_name = images[i]
        mask_name = masks[i]
        img_no = file_re.match(img_name).groups()[0]
        mask_no = file_re.match(mask_name).groups()[0]
        assert int(img_no) == int(mask_no)
        img = load(path.join(images_dir, img_name), )
        mask = load(path.join(masks_dir, mask_name), is_mask=True)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_CUBIC)
        (_, mask) = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        for sub_img, sub_mask in crop(img, mask):
            cv2.imwrite(path.join(new_dir, "images", "{}.jpg".format(f'{int(count + 1):05}')),
                        cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(path.join(new_dir, "masks", "{}.jpg".format(f'{int(count + 1):05}')), sub_mask)
            count += 1


def check_casia_mask_image_match(img_dir, mask_dir, ):
    image_files = glob.glob(os.path.join(img_dir, '*.*'))
    mask_files = glob.glob(os.path.join(mask_dir, '*.*'))
    assert len(image_files) == len(image_files)
    image_names = [os.path.basename(element).split('.')[0] for element in image_files]
    mask_names = [os.path.basename(element).split('.')[0].replace('_gt', '') for element in mask_files]
    difference = set(image_names).symmetric_difference(set(mask_names))
    assert len(difference) == 0
    print('images and masks all match')


def check_datatype(dir):
    files = glob.glob(os.path.join(dir, '*.*'))
    exts = [os.path.basename(element).split('.')[1] for element in files]
    print(set(exts))


def move_casia_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    files = glob.glob(os.path.join(src_dir, '*.*'))
    for file in tqdm(files):
        filename = os.path.basename(file)
        filename, file_ext = filename.split('.')[0], filename.split('.')[1]
        if file_ext == 'tif':
            im = Image.open(file)
            out = im.convert('RGB')
            out.save(os.path.join(dst_dir, "{}.{}".format(filename, 'jpg')), 'JPEG', quality=95)
        else:
            copyfile(file, os.path.join(dst_dir, "{}.{}".format(filename, 'jpg')))


def move_casia_masks(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    files = glob.glob(os.path.join(src_dir, '*.*'))
    for file in tqdm(files):
        filename = os.path.basename(file)
        file_name, file_exit = filename.split('.')[0], filename.split('.')[1]
        type = file_name.split('_')[1]
        im = Image.open(file)
        w, h = im.size
        mask = None
        if type == 'S':
            mask = Image.open(os.path.join(DATASET_ROOT, 'CASIA2', 'CM_MASK', "{}_gt.png".format(file_name)))
        elif type == 'D':
            mask = Image.open(os.path.join(DATASET_ROOT, 'CASIA2', 'SP_MASK', "{}_gt.png".format(file_name)))
        else:
            raise Exception('got other type of forgery, {}'.format(type))
        assert mask != None
        mask_w, mask_h = mask.size
        if mask_w != w or mask_h != h:
            mask = mask.resize((w, h))
            mask.save(os.path.join(dst_dir, "{}.{}".format(file_name, 'png')), 'PNG')
        else:
            mask.save(os.path.join(dst_dir, "{}.{}".format(file_name, 'png')), 'PNG')


def combine_forge_and_rename(destinate_dir, *dataset_dirs):
    class_to_int = {"cm": 0, "sp": 1}
    dataset_number = len(dataset_dirs)
    files = list()
    for i, dir in enumerate(dataset_dirs):
        print(dir)
        locals()['folder{}'.format(i + 1)] = dir
        files = files + glob.glob(os.path.join(dir, '*.*'))
    shuffle(files)

    workbook = xlsxwriter.Workbook('test.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'file')
    worksheet.write('B1', 'original_path')
    worksheet.write('C1', 'class')
    worksheet.write('D1', 'predict')
    for i, file in enumerate(files):
        img_name = "{}.jpg".format(f'{i:06}')
        class_type = os.path.split(file)[1].split("_")[0]
        shutil.copyfile(file, os.path.join(destinate_dir, img_name))

        worksheet.write('A{}'.format(i + 2), img_name)
        worksheet.write('B{}'.format(i + 2), file)
        worksheet.write('C{}'.format(i + 2), class_to_int[class_type])

    print(len(files))
    workbook.close()


if __name__ == '__main__':
    print("__main__")
    # casia_v1_mask()
    # split_CASIA2()
    # get_COMOFOD()
    # get_COVERAGE()
    # combine_multiple_dataset()

    # DATASETS_DIR = r'/media/ian/WD/datasets'
    # img_folder = os.path.join(DATASETS_DIR, 'forge_combine', 'forge')
    # gt_folder = os.path.join(DATASETS_DIR, 'forge_combine', 'mask')
    # split_folder = os.path.join(DATASETS_DIR, 'forge_combine', 'split')
    # split_train_test(img_folder, gt_folder, split_folder)

    find_all_black_mask(['/media/ian/WD/datasets/forge_combine/test_masks'])
    ######################################################################################
    # #測試x是否存在在item中
    # x = 'aaa'
    # L = ['aaa-12', 'bbbaaa', 'cccaa']
    # res = [y for y in L if x in y]
    # print(res)
    ######################################################################################
    # DATASETS_DIR = r'D:\media\ian\WD\datasets'
    # img_folder = os.path.join(DATASETS_DIR, r'total_forge\forge')
    # gt_folder = os.path.join(DATASETS_DIR, r'total_forge\mask')
    # split_folder = os.path.join(DATASETS_DIR, r'total_forge\train_and_test')
    # split_train_test(img_folder, gt_folder, split_folder)
    ######################################################################################
    # video_folder = os.path.join('D:\\', 'VTD', 'video_tampering_dataset', 'videos', 'h264_lossless')
    # make_video_to_image(video_folder)
    # combine_video_image(video_folder)
    # split_train_test(os.path.join(video_folder, 'final_forge'), os.path.join(video_folder, 'final_mask'), os.path.join(video_folder, 'test_and_train'))
    ######################################################################################
    # devide_dataset_to_small_patches(r'/media/ian/WD/datasets/total_forge/train_and_test/train/images',
    #                                 r'/media/ian/WD/datasets/total_forge/train_and_test/train/masks',
    #                                 r'/media/ian/WD/datasets/total_forge/train_and_test/train_small_patch(2nd)')
    ######################################################################################
    # copy_move_folder = os.path.join('/media', 'ian', 'WD', 'datasets', 'total_forge', 'SP', )
    ## make_video_to_image(video_folder)
    ## combine_video_image(video_folder)
    # split_train_test(os.path.join(copy_move_folder, 'forge'), os.path.join(copy_move_folder, 'mask'),
    #                  os.path.join(copy_move_folder, 'test_and_train'))
    ######################################################################################
    ######################################################################################
    # combine_forge_and_rename(DATASET_ROOT.joinpath('COCO', 'large_cm_sp', 'test'),
    #                          DATASET_ROOT.joinpath('COCO', 'coco2017_large_sp', 'A', 'test'),
    #                          DATASET_ROOT.joinpath('COCO', 'coco2017_no_overlap_cm', 'A', 'test'))

    ######################################################################################
    # check_casia_mask_image_match(DATASET_ROOT.joinpath('CASIA2', 'SP_Tp'),
    #                              DATASET_ROOT.joinpath('CASIA2', 'SP_MASK'))
    # check_casia_mask_image_match(DATASET_ROOT.joinpath('CASIA2', 'CM_Tp'),
    #                              DATASET_ROOT.joinpath('CASIA2', 'CM_MASK'))
    ######################################################################################
    # check_datatype(DATASET_ROOT.joinpath('CASIA2', 'total','images'))
    # check_datatype(DATASET_ROOT.joinpath('CASIA2', 'total', 'masks'))
    ######################################################################################
    # move_casia_images(str(DATASET_ROOT.joinpath('CASIA2', 'CM_Tp')),
    #                   str(DATASET_ROOT.joinpath('CASIA2', 'total', 'images')))
    # move_casia_images(str(DATASET_ROOT.joinpath('CASIA2', 'SP_Tp')),
    #                   str(DATASET_ROOT.joinpath('CASIA2', 'total', 'images')))
    ######################################################################################
    # move_casia_masks(
    #     str(DATASET_ROOT.joinpath('CASIA2', 'total', 'images')), str(DATASET_ROOT.joinpath('CASIA2', 'total', 'masks')))
    # split_train_test(str(DATASET_ROOT.joinpath('CASIA2', 'total', 'images')),
    #                  str(DATASET_ROOT.joinpath('CASIA2', 'total', 'masks')),
    #                  str(DATASET_ROOT.joinpath('CASIA2', 'split')))
    ######################################################################################
    # img = cv2.imread(str(DATASET_ROOT.joinpath('CASIA2', 'SP_Tp', 'Tp_D_CND_M_N_ani00018_sec00096_00138.tif')))
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('img_rgb', img)
    # cv2.waitKey()
    # cv2.destroyWindow('img_rgb')
    # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyWindow('img')
    # print('')
    ######################################################################################
    # check any mask got all black

    # SP_IMG_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/Sp",
    #                    r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/SP_Tp",
    #                    r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_sp/train2017"]
    # SP_MASK_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/Sp_mask",
    #                     r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/SP_MASK",
    #                     r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_sp/train2017_mask"]
    #
    # CM_DATASET_NAME_LIST = ["CASIA1", "CASIA2", "CoMoFoD", "COVERAGE", "COCO"]
    # CM_IMG_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM",
    #                    r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/CM_Tp",
    #                    r"/media/ian/WD/datasets/CoMoFoD/FORGE",
    #                    r"/media/ian/WD/datasets/COVERAGE/new_forge",
    #                    r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_cm/train2017"]
    # CM_MASK_DIR_LIST = [r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM_mask",
    #                     r"/media/ian/WD/datasets/CASIA/CASIA2.0_revised/CM_MASK",
    #                     r"/media/ian/WD/datasets/CoMoFoD/MASK",
    #                     r"/media/ian/WD/datasets/COVERAGE/new_mask",
    #                     r"/media/ian/WD/datasets/COCO/coco2017_forge_combine_cm/train2017_mask"]
    # check_list = SP_IMG_DIR_LIST + CM_IMG_DIR_LIST

    ################################################################################################################################
    ## chekc
    # find_img_with_4_channel(check_list)
    # tmp = cv2.imread(r"/media/ian/WD/datasets/CASIA/CASIA 1.0 dataset/Modified Tp/Tp/CM_mask/Sp_S_CNN_A_nat0078_nat0078_0004.jpg", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('tmp', tmp)
    # print(np.unique(tmp))
    # _, tmp = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('tmp2', tmp)
    # cv2.waitKey()
    # cv2.destroyWindow('tmp')
    # print(np.unique(tmp))
