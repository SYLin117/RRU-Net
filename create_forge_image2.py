import os
import sys
import json
import random
import time

from utils import get_dataset_root

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
DATASET_ROOT = get_dataset_root()
# DATASET_ROOT = os.path.abspath("/run/user/1000/gvfs/smb-share:server=ubuntu1070.local,share=samba-share")
COCO_2017_DIR = os.path.join(DATASET_ROOT, "COCO", "coco2017")
COCO_DIR = os.path.join(DATASET_ROOT, "COCO")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from random import randint
from PIL import Image
import numpy as np

from datetime import datetime

import shutil

from tqdm import tqdm
import copy
import cv2
from utils import bb_intersection_over_union

image_index = []
seg_index = []

FORGE_TYPE = 'cm'
gene_dir_path = os.sep.join([COCO_DIR, 'coco2017_new_{}'.format(FORGE_TYPE)])
gene_train_image_path = os.sep.join([gene_dir_path, 'train2017'])
gene_val_image_path = os.sep.join([gene_dir_path, 'val2017'])
gene_test_image_path = os.sep.join([gene_dir_path, 'test2017'])
gene_annotation_path = os.sep.join([gene_dir_path, 'annotations'])

COCO_URL = 'http://images.cocodataset.org'
LOAD_DATA_TYPE = 'train'
COCO_TRAIN = 'train'
COCO_TEST = 'test'
COCO_VAL = 'val'

# DATASET_SIZE = int(len(image_index) / 2)
DATASET_SIZE = 10000
LOOP_CNT = 10
OVERLAP_CNT = 10


def random_seg_idx():
    return randint(0, len(seg_index) - 1)


def random_obj_idx(s):
    return randint(1, len(s) - 2)  # 去掉0，255


def random_obj_loc(img_h, img_w, obj_h, obj_w, pad=0):
    return randint(pad, img_h - obj_h - pad), randint(pad, img_w - obj_w - pad)


def random_img_idx():
    return randint(0, len(image_index) - 1)


def find_obj_vertex(mask):
    hor = np.where(np.sum(mask, axis=0) > 0)
    ver = np.where(np.sum(mask, axis=1) > 0)
    return hor[0][0], hor[0][-1], ver[0][0], ver[0][-1]


# def modify_xml(filename, savefile, xmin, ymin, xmax, ymax):
#     def create_node(tag, property_map, content):
#         element = Element(tag, property_map)
#         element.text = content
#         return element
#
#     copyfile(filename, savefile)
#     tree = ET.parse(savefile)
#     root = tree.getroot()
#     for obj in root.findall('object'):
#         root.remove(obj)
#     new_obj = Element('object', {})
#     new_obj.append(create_node('name', {}, 'tampered'))
#     bndbox = Element('bndbox', {})
#     bndbox.append(create_node('xmin', {}, str(xmin)))
#     bndbox.append(create_node('ymin', {}, str(ymin)))
#     bndbox.append(create_node('xmax', {}, str(xmax)))
#     bndbox.append(create_node('ymax', {}, str(ymax)))
#     new_obj.append(bndbox)
#     root.append(new_obj)
#     tree.write(savefile)

def train_val_test():
    a = randint(1, 100)
    if a <= 10:
        return COCO_TEST  # test
    elif 0 < a <= 20:
        return COCO_VAL  # val
    else:
        return COCO_TRAIN  # train


def mod_or_not():
    a = randint(1, 100)
    if a <= 0:
        return False  # not mod
    else:
        return True  # mod


def json_add_image(json_data, img_info, type, img=None, mask=None, img_name=None, mask_name=None):
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp_add_img = {}
    tmp_add_img['license'] = 4

    # firstpos = img_info['path'].rfind("/")
    # filename = img_info['path'][firstpos + 1:]
    filename = "{:012d}.jpg".format(img_info['id'])
    tmp_add_img['file_name'] = filename
    tmp_add_img['id'] = img_info['id']
    tmp_add_img['coco_url'] = os.sep.join([COCO_URL, filename])
    tmp_add_img['height'] = img_info['height']
    tmp_add_img['width'] = img_info['width']
    tmp_add_img['date_captured'] = create_time
    tmp_add_img['id'] = img_info['id']
    json_data['images'].append(tmp_add_img)

    new_dir = os.path.join(gene_dir_path, "{}2017".format(type), )
    new_mask_dir = os.path.join(gene_dir_path, "{}2017_mask".format(type), )
    # img_dir = os.path.join(new_dir, filename)
    # mask_dir = os.path.join(new_mask_dir, filename.split('.')[0] + ".png")
    img_dir = os.path.join(new_dir, img_name + ".jpg")
    mask_dir = os.path.join(new_mask_dir, mask_name + ".png")
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    if not os.path.exists(new_mask_dir):
        os.makedirs(new_mask_dir)

    if not img:  # img is None
        shutil.copyfile(img_info['path'], img_dir)
    else:
        img.save(img_dir)
        mask.save(mask_dir)


def json_add_ann(json_data, img_info, seg_ann, loc_x, dx, loc_y, dy):
    tmp_add_ann = {}
    tmp_add_ann['segmentation'] = seg_ann['segmentation']
    tmp_add_ann['area'] = seg_ann['area']
    tmp_add_ann['iscrowd'] = seg_ann['iscrowd']
    tmp_add_ann['image_id'] = img_info['id']
    tmp_add_ann['bbox'] = [loc_x, loc_y, dx, dy]
    tmp_add_ann['category_id'] = 1
    tmp_add_ann['id'] = seg_ann['id']
    json_data['annotations'].append(tmp_add_ann)


if __name__ == '__main__':

    # type = 'sp'
    create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # MS COCO Dataset
    from coco import coco as my_coco

    config = my_coco.CocoConfig()  # subclass of Config.py(大多存mask-rcnn的參數) 該子class則存入針對coco資料集的參數

    # Load dataset
    my_coco_dataset = my_coco.CocoDataset(None)
    coco = my_coco_dataset.load_coco(COCO_2017_DIR, LOAD_DATA_TYPE, return_coco=True)

    # Must call before using the dataset
    my_coco_dataset.prepare()

    print("Image Count: {}".format(len(my_coco_dataset.image_ids)))
    print("Class Count: {}".format(my_coco_dataset.num_classes))

    for i, info in enumerate(my_coco_dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    ann_json_file = open(os.path.join(COCO_2017_DIR, 'annotations', 'instances_val2017.json'))
    train_json = json.load(ann_json_file)
    train_json['images'] = []
    train_json['annotations'] = []
    train_json['categories'] = []
    tmp_add_cat = {}
    tmp_add_cat['supercategory'] = "modified"
    tmp_add_cat['id'] = 1
    tmp_add_cat['name'] = "modified"
    train_json['categories'].append(tmp_add_cat)
    val_json = copy.deepcopy(train_json)

    image_index = my_coco_dataset.image_ids  # image_index = image_ids

    seg_index = my_coco_dataset.obj_info
    random.shuffle(image_index)
    pbar = tqdm(total=DATASET_SIZE)
    count = 0
    while count <= DATASET_SIZE:  # need how many data
        # img_idx = count % len(image_index)
        img_idx = np.random.choice(image_index, 1)[0]  # 防止index访问越界
        image_id = image_index[img_idx]
        img = my_coco_dataset.load_image(image_id)  # 需要修改的原图
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_info = my_coco_dataset.image_info[image_id]
        if (mod_or_not()):  # random chance to mod or not
            seg_img = None
            seg_img_id = None
            seg_ann = None
            min_x, max_x, min_y, max_y = None, None, None, None
            loop_counter = 0
            find_obj = False
            new_img, mask_img = None, None
            while (loop_counter <= LOOP_CNT):  # try 1000 times to find a object
                # random choose a random image for segmentation
                ##################################################################################################
                ## for splied
                if FORGE_TYPE == 'sp':
                    seg_img_id = np.random.choice(image_index, 1)[0]  # original coco id
                    if seg_img_id == image_id:
                        continue
                elif FORGE_TYPE == 'cm':
                    # copy-move
                    seg_img_id = image_id
                else:
                    raise Exception('type not included')
                ##################################################################################################
                seg_img = my_coco_dataset.load_image(seg_img_id)
                seg_img_info = my_coco_dataset.image_info[seg_img_id]
                mask_lst_tmp, class_ids_tmp, seg_anns = my_coco_dataset.load_mask(seg_img_id)

                for i in range(len(seg_anns)):
                    loop_counter += 1
                    mask2 = mask_lst_tmp[:, :, i].astype(int)
                    seg_ann = seg_anns[i]

                    # seg = Image.open(imdb.seg_path_at(seg_idx)).convert('P')  # 待抠图的segmentation ground truth，模式“P”为8位彩色图像
                    # mask2 = my_coco_dataset.annToMask(seg_ann, seg_img.shape[0], seg_img.shape[1])  # obj mask
                    if abs(mask2.shape[0] - seg_img.shape[0]) > 2 or abs(mask2.shape[1] - seg_img.shape[1]) > 2:
                        print(
                            "get a wierd object mask with height: {} and width: {} but ann with height: {} and width: {}".format(
                                mask2.shape[0], mask2.shape[1], seg_img.shape[0], seg_img.shape[1]))
                        continue
                    # min_x, max_x, min_y, max_y = find_obj_vertex(mask2)  # 覆盖区域最小外接长方形
                    min_x, max_x, min_y, max_y = int(seg_ann['bbox'][0]), \
                                                 int(seg_ann['bbox'][0] + seg_ann['bbox'][2]), \
                                                 int(seg_ann['bbox'][1]), \
                                                 int(seg_ann['bbox'][1] + seg_ann['bbox'][3])
                    dx, dy = max_x - min_x, max_y - min_y
                    w, h = dx, dy

                    if (((max_x - min_x) * (max_y - min_y) <= img_pil.width * img_pil.height * 0.05) or
                            ((max_x - min_x) * (max_y - min_y) >= img_pil.width * img_pil.height * 0.5) or
                            (max_x - min_x > img_pil.width) or
                            (max_y - min_y > img_pil.height)):
                        continue

                    mask = np.stack((mask2, mask2, mask2), axis=2)
                    mask_pil = Image.fromarray(mask2 * 255).convert('L')
                    # seg_img_rgb = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
                    seg_img_pil = Image.fromarray(seg_img)

                    crop_obj = seg_img_pil.crop((min_x, min_y, max_x, max_y))
                    crop_mask = mask_pil.crop((min_x, min_y, max_x, max_y))
                    orig_crop_mask = crop_mask
                    # ---------------------------
                    # Random scale
                    # ---------------------------
                    scale = random.uniform(0.4, 0.7)
                    w, h = int(w * scale), int(h * scale)
                    crop_obj = crop_obj.resize((w, h), resample=Image.BILINEAR)
                    crop_mask = crop_mask.resize((w, h), resample=Image.BILINEAR)

                    # ---------------------------
                    # Random rotate
                    # ---------------------------
                    angle = random.random() * 360
                    crop_obj = crop_obj.rotate(angle, resample=Image.BILINEAR, expand=1)
                    crop_mask = crop_mask.rotate(angle, resample=Image.BILINEAR, expand=1)

                    # ---------------------------
                    # Crop according to mask
                    # ---------------------------
                    where = np.where(np.array(crop_mask))  # value == 255 location
                    # where = np.vstack((where[0], where[1]))  ## ian added
                    assert len(where[0]) != 0
                    assert len(where[1]) != 0
                    assert len(where[0]) == len(where[1])
                    y1, x1 = np.amin(where, axis=1)
                    y2, x2 = np.amax(where, axis=1)
                    crop_obj = crop_obj.crop((x1, y1, x2, y2))
                    crop_mask = crop_mask.crop((x1, y1, x2, y2))
                    w, h = crop_obj.width, crop_obj.height
                    # crop_obj.show("{}".format(f'{count:06}'))
                    # crop_mask.show("{}".format(f'{count:06}'))
                    # time.sleep(1)
                    ## ============= 太大object剔除 =================
                    if ((w * h <= img_pil.width * img_pil.height * 0.05) or
                            (w * h >= img_pil.width * img_pil.height * 0.5) or
                            (w > img_pil.width) or
                            (h > img_pil.height)):
                        continue
                    new_loc_valid = True if FORGE_TYPE == 'sp' else False
                    loc_y, loc_x = None, None

                    if FORGE_TYPE == 'cm':
                        try_no = 0
                        while not new_loc_valid:
                            loc_y, loc_x = random_obj_loc(img_pil.height, img_pil.width, h, w)
                            boxA = [min_x, min_y, max_x, max_y]  # original
                            boxB = [loc_x, loc_y, loc_x + w, loc_y + h]  # new location
                            overlap = bb_intersection_over_union(boxA, boxB)
                            if overlap < 0.05:
                                new_loc_valid = True
                            else:
                                print('new location got overlap:{}'.format(overlap))
                                try_no += 1
                                if try_no > OVERLAP_CNT:
                                    break
                    elif FORGE_TYPE == 'sp':
                        loc_y, loc_x = random_obj_loc(img_pil.height, img_pil.width, h, w)
                    if not new_loc_valid:
                        print('object can\'t get valid location.')
                        continue

                    mask_tmp = np.zeros_like(img)
                    mask_tmp_pil = Image.fromarray(mask_tmp).convert('L')
                    img_pil.paste(crop_obj, box=(loc_x, loc_y), mask=crop_mask)
                    mask_tmp_pil.paste(crop_mask, box=(loc_x, loc_y), mask=crop_mask)
                    if FORGE_TYPE == 'cm':
                        mask_tmp_pil.paste(orig_crop_mask, box=(min_x, min_y), mask=orig_crop_mask)

                    # tmp = np.array(mask_tmp_pil)
                    # kernel = np.ones((3, 3), np.uint8)
                    # tmp[tmp >= 1] = 255
                    # tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)
                    # mask_tmp_pil = Image.fromarray(tmp).convert('L')
                    # img_pil.show()
                    # mask_tmp_pil.show()
                    # time.sleep(5)
                    new_img = img_pil
                    mask_img = mask_tmp_pil

                    find_obj = True
                    break
                if find_obj:  # spliced
                    break

            if not find_obj:  # spliced
                continue
            # # 未得到合适大小的obj
            # if loop_counter >= LOOP_CNT:
            #     continue
            assert new_img != None
            assert mask_img != None
            count += 1
            pbar.update(1)
            split_type = train_val_test()
            img_name = "{}_{}".format(FORGE_TYPE, f'{count:06}')
            json_add_image(json_data=train_json, img_info=img_info, type=split_type, img=new_img, mask=mask_img,
                           img_name=img_name, mask_name=img_name)
        else:
            ## no modification, so no need to add ann json
            split_type = train_val_test()
            if split_type == COCO_TRAIN:
                json_add_image(json_data=train_json, img_info=img_info, type=type)
            elif split_type == COCO_VAL:
                json_add_image(json_data=val_json, img_info=img_info, type=type)
            elif split_type == COCO_TEST:
                firstpos = img_info['path'].rfind("/")
                filename = img_info['path'][firstpos + 1:]

                new_dir = os.path.join(COCO_DIR, "coco2017_forge", "{}2017".format(type), )
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                shutil.copyfile(img_info['path'], os.path.join(new_dir, filename))

    pbar.close()
