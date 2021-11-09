import numpy as np
import cv2
import re
import os
from sklearn.metrics import confusion_matrix
from pathlib import Path
from utils import *


def evaluate_by_folder(gt_folder, pred_folder):
    THRESHOLD = 1
    gt_regex = re.compile("([0-9]+)\.")
    pred_regex = re.compile("([0-9]+)_OUT\.")
    gt_list = [f for f in os.listdir(gt_folder)]
    pred_list = [f for f in os.listdir(pred_folder)]
    gt_list.sort(), pred_list.sort()
    assert len(gt_list) == len(pred_list), 'files number not match, gt:{} and pred:{}'.format(len(gt_list),
                                                                                              len(pred_list))

    tns = np.zeros(len(gt_list))
    fps = np.zeros(len(gt_list))
    fns = np.zeros(len(gt_list))
    tps = np.zeros(len(gt_list))
    total_pix = np.zeros(len(gt_list))
    for i in range(len(gt_list)):
        gt_filename = gt_list[i]
        pred_filename = pred_list[i]
        try:
            gt_no = gt_regex.match(gt_filename).groups()[0]
            pred_no = pred_regex.match(pred_filename).groups()[0]
            assert int(gt_no) == int(pred_no), "gt and pred file number not match"
            gt_mask = cv2.imread(os.path.join(gt_folder, gt_filename), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.morphologyEx(gt_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
            pred_mask = cv2.imread(os.path.join(pred_folder, pred_filename), cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))  # (width, height)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3)
            assert gt_mask.shape == pred_mask.shape, "gt and pred got different size, file: {}, gt.shape:{}, pred.shape:{}".format(
                gt_filename, gt_mask.shape, pred_mask.shape)
            (_, gt_mask) = cv2.threshold(gt_mask, THRESHOLD, 1, cv2.THRESH_BINARY)
            (_, pred_mask) = cv2.threshold(pred_mask, THRESHOLD, 1, cv2.THRESH_BINARY)
            gt_mask_1d = gt_mask.flatten()
            pred_mask_1d = pred_mask.flatten()
            tn, fp, fn, tp = confusion_matrix(gt_mask_1d, pred_mask_1d, labels=[0, 1]).ravel()
            tns[i] = tn
            fps[i] = fp
            fns[i] = fn
            tps[i] = tp
            total_pix[i] = tn + fp + fn + tp
        except AssertionError as ae:
            print("file: {} not match, error msg: {}".format(gt_filename, str(ae)))
            continue
    tmp = (tps + fps)
    precision = np.where(tmp != 0, tps / (tps + fps), 0)
    recall = np.where((tps + fns) != 0, tps / (tps + fns), 0)
    acc = (tps + tns) / (tps + fps + fns + tns)
    f1 = np.where(precision + recall != 0, 2 * np.multiply(precision, recall) / np.add(precision, recall), 0)
    return precision, recall, acc, f1


if __name__ == '__main__':
    print("---main---")
    np.set_printoptions(suppress=True)
    DATASETS_DIR = get_dataset_root()
    # dir_img = DATASETS_DIR.joinpath('total_forge/forge')
    # dir_mask = DATASETS_DIR.joinpath('total_forge/mask')
    # val_percent = 0.1
    # dataset = ForgeDataset(dir_img, dir_mask, 1, mask_suffix='')
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    #############################################################################################################################
    # pred_mask = cv2.imread('./data/test/predict/000064_OUT.jpg', cv2.IMREAD_GRAYSCALE)
    # gt_mask = cv2.imread('./data/test/masks/000064.jpg', cv2.IMREAD_GRAYSCALE)
    # (_, gt_mask) = cv2.threshold(gt_mask, 10, 1, cv2.THRESH_BINARY)
    # (_, pred_mask) = cv2.threshold(pred_mask, 10, 1, cv2.THRESH_BINARY)
    # print(np.unique(gt_mask))
    # print(np.unique(pred_mask))
    # assert pred_mask.shape == gt_mask.shape, "shape doesn't match"
    # print((gt_mask == pred_mask))
    # unique, counts = np.unique((gt_mask == pred_mask), return_counts=True)
    # print(dict(zip(unique, counts)))
    #############################################TOTAL-FORGE DATASET###############################################################
    precision, recall, acc, f1 = evaluate_by_folder(
        os.path.join(DATASETS_DIR, 'total_forge', 'train_and_test', 'test', 'masks'),
        os.path.join(DATASETS_DIR, 'total_forge',  'train_and_test', 'test', 'predict'))
    gt_list = [f for f in os.listdir(
        os.path.join(DATASETS_DIR, 'total_forge',  'train_and_test', 'test', 'masks'))]
    #############################################################################################################################
    # precision, recall, acc, f1 = evaluate_by_folder(os.path.join('/media', 'ian', 'WD', 'PythonProject', 'RRU-Net',
    #                                                              'data', 'video_test', 'masks'),
    #                                                 os.path.join('/media', 'ian', 'WD', 'PythonProject', 'RRU-Net',
    #                                                              'data', 'video_test', 'predict(RRU)2'))
    # gt_list = [f for f in os.listdir(os.path.join('/media', 'ian', 'WD', 'PythonProject', 'RRU-Net',
    #                                               'data', 'test', 'masks'))]
    #############################################COPY-MOVE DATASET###############################################################
    # precision, recall, acc, f1 = evaluate_by_folder(
    #     os.path.join(DATASETS_DIR, 'total_forge', 'CM', 'test_and_train', 'test', 'masks'),
    #     os.path.join(DATASETS_DIR, 'total_forge', 'CM', 'test_and_train', 'test', 'predict'))
    # gt_list = [f for f in os.listdir(
    #     os.path.join(DATASETS_DIR, 'total_forge', 'CM', 'test_and_train', 'test', 'masks'))]
    #############################################SPLICING DATASET###############################################################
    # precision, recall, acc, f1 = evaluate_by_folder(
    #     os.path.join('/media', 'ian', 'WD', 'datasets', 'total_forge', 'SP', 'test_and_train', 'test', 'masks'),
    #     os.path.join('/media', 'ian', 'WD', 'datasets', 'total_forge', 'SP', 'test_and_train', 'test', 'predict'))
    # gt_list = [f for f in os.listdir(
    #     os.path.join('/media', 'ian', 'WD', 'datasets', 'total_forge', 'SP', 'test_and_train', 'test', 'masks'))]
    #############################################################################################################################
    gt_list.sort()
    avg_prec = np.average(precision)
    avg_recall = np.average(recall)
    avg_acc = np.average(acc)
    avg_f1 = np.average(f1)
    print("average precision: {}".format(avg_prec))
    print("average recall: {}".format(avg_recall))
    print("average accuracy: {}".format(avg_acc))
    print("average f1 score: {}".format(avg_f1))
    k = 10
    top_prec = np.argpartition(precision, -k)[-k:]
    bot_prec = np.argpartition(precision, k)[:k]
    print('top prec file: {}'.format([gt_list[idx] for idx in top_prec]))
    print('bot prec file: {}'.format([gt_list[idx] for idx in bot_prec]))
    print('top prec\'s recall : {}'.format(recall[top_prec]))

    top_recall = np.argpartition(recall, -k)[-k:]
    bot_recall = np.argpartition(recall, k)[:k]
    print('top recall file: {}'.format([gt_list[idx] for idx in top_recall]))
    print('bot recall file: {}'.format([gt_list[idx] for idx in bot_recall]))
    print('top recall\'s prec : {}'.format(recall[top_recall]))

    top_f1 = np.argpartition(f1, -k)[-k:]
    bot_f1 = np.argpartition(f1, k)[:k]
    print('top f1 file: {}'.format([gt_list[idx] for idx in top_f1]))
    print('bot f1 file: {}'.format([gt_list[idx] for idx in bot_f1]))
    print('top f1\'s prec : {}'.format(precision[top_prec]))
    ##################################### test confusion matrix #######################################################
    # cm = confusion_matrix([0, 0, 1, 1], [1, 0, 0, 1], labels=[0, 1]).ravel()
    # print(cm)
