from tqdm import tqdm
from utils import *
import re
from glob import glob
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.metrics import confusion_matrix
from efficientnet import EfficientNet_b0, EfficientNet_b1, EfficientNet_b2


def predict_img(classifier,
                net1,
                net2,
                full_img,
                resize=(300, 300),
                use_gpu=True):
    tf = T.Compose([
        T.Resize([resize[0], resize[1]], InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img = tf(full_img)
    img = img.unsqueeze(dim=0)
    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        pred = classifier(img)

        mask = net1(img)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()
    # return mask > out_threshold
    return mask


def get_output_filenames(file_list, predict_subfolder='predict'):
    """
    input a filename list
    return a output filename list
    """
    tmp = file_list[0]
    parent_folder = os.path.split(os.path.split(tmp)[0])[0]
    OUTPUT_ROOT = os.path.join(parent_folder, predict_subfolder)
    if not os.path.exists(OUTPUT_ROOT):
        os.mkdir(OUTPUT_ROOT)

    def _generate_name(fn):
        filename = os.path.split(fn)[1]
        split = os.path.splitext(filename)
        # return os.path.join(OUTPUT_ROOT, f'{split[0]}_OUT{split[1]}')
        return os.path.join(OUTPUT_ROOT, f'{split[0]}_OUT.png')

    out_list = list(map(_generate_name, file_list))
    return out_list


# def mask_to_image(mask):
#     return Image.fromarray((mask * 255).astype(np.uint8))

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == "__main__":
    import pathlib

    DATASETS_DIR = get_dataset_root()
    scale, mask_threshold, cpu, viz, no_save = 1, 0.5, False, False, True
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'

    current_path = str(pathlib.Path().resolve())
    classifier = 'EFFICIENTNET_B0'
    network1 = 'Ringed_Res_Unet'
    network2 = 'Ringed_Res_Unet'
    resize = (300, 300)
    #######################################################################################
    in_files = os.path.join(DATASETS_DIR, 'CASIA2', 'total', 'images')
    gt_files = os.path.join(DATASETS_DIR, 'CASIA2', 'total', 'masks')
    ##########################################COPY-MOVE MODEL###########################################################
    model_classifier = os.path.join(current_path, 'result', 'logs', 'large_cm_sp', classifier, 'best_model.pth.pth')
    model1 = os.path.join(current_path, 'result', 'logs', 'superlarge_cm', network1, 'best_model.pth.pth')
    model2 = os.path.join(current_path, 'result', 'logs', 'superlarge_sp', network2, 'best_model.pth.pth')
    ##########################################SPLICING MODEL###########################################################
    net_classifier, net1, net2 = None, None, None
    if classifier == "EFFICIENTNET_B0":
        net_classifier = EfficientNet_b0(num_classes=1)
    elif classifier == "EFFICIENTNET_B2":
        net_classifier = EfficientNet_b2(num_classes=1)

    if not cpu:
        net_classifier.cuda()
        net_classifier.load_state_dict(torch.load(model_classifier))
    else:
        net_classifier.cpu()
        net_classifier.load_state_dict(torch.load(model_classifier, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    if network1 == 'Unet':
        net1 = Unet(n_channels=3, n_classes=1)
    elif network1 == 'Res_Unet':
        net1 = Res_Unet(n_channels=3, n_classes=1)
    elif network1 == 'Ringed_Res_Unet':
        net1 = Ringed_Res_Unet(n_channels=3, n_classes=1)
    elif network1 == 'MyTransUnet':
        net1 = MyTransUNet(in_channels=3, classes=1, img_dim=resize[0])
    elif network1 == 'MyTransUnet2':
        net1 = MyTransUNet2(in_channels=3, classes=1, img_dim=resize[0])
    else:
        raise Exception("model not implements.")

    if not cpu:
        net1.cuda()
        net1.load_state_dict(torch.load(model1))
    else:
        net1.cpu()
        net1.load_state_dict(torch.load(model1, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    if network2 == 'Unet':
        net2 = Unet(n_channels=3, n_classes=1)
    elif network2 == 'Res_Unet':
        net2 = Res_Unet(n_channels=3, n_classes=1)
    elif network2 == 'Ringed_Res_Unet':
        net2 = Ringed_Res_Unet(n_channels=3, n_classes=1)
    elif network2 == 'MyTransUnet':
        net2 = MyTransUNet(in_channels=3, classes=1, img_dim=resize[0])
    elif network2 == 'MyTransUnet2':
        net2 = MyTransUNet2(in_channels=3, classes=1, img_dim=resize[0])
    else:
        raise Exception("model not implements.")

    if not cpu:
        net2.cuda()
        net2.load_state_dict(torch.load(model1))
    else:
        net2.cpu()
        net2.load_state_dict(torch.load(model1, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    if isinstance(in_files, str):
        if os.path.isfile(in_files):
            print("path is file")
        elif os.path.isdir(in_files):  # in_files是資料夾
            auc_list = list()
            time_list = list()
            tp_list = list()
            fp_list = list()
            tn_list = list()
            fn_list = list()
            pixel_list = list()
            threshold_list = list()
            input_files = glob(os.path.join(in_files, '*.*'))

            input_files = sorted(input_files)

            # print("input folder got {} files".format(len(input_files)))
            output_files = get_output_filenames(input_files)
            for i, filename in enumerate(tqdm(input_files)):

                gt_mask = cv2.imread(os.path.join(DATASETS_DIR, 'CASIA2', 'total', 'masks'), cv2.IMREAD_GRAYSCALE)
                (_, gt_mask) = cv2.threshold(gt_mask, 125, 1, cv2.THRESH_BINARY)

                img = Image.open(filename).convert('RGB')  # cause some image got 4 channel(RGBA), eg:001221 file
                width, height = img.size
                pixel_list.append(width * height)
                try:
                    start_time = time.time()
                    mask = predict_img(net=net,
                                       full_img=img,
                                       resize=resize,
                                       use_gpu=not cpu)
                    # print("predict time:{}".format(time.time() - start_time))
                    time_list.append(time.time() - start_time)
                    # save_mask = mask > mask_threshold
                    mask = cv2.resize(mask, (gt_mask.shape[1], gt_mask.shape[0]), cv2.INTER_CUBIC)
                    y_test = gt_mask.flatten()
                    y_pred = mask.flatten()
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                    fpr_tpr_diff = fpr - tpr
                    best_thres_idx = np.argmax(fpr_tpr_diff, axis=0)
                    save_mask = mask > thresholds[best_thres_idx]
                    tn, fp, fn, tp = confusion_matrix(y_test, save_mask.flatten(), labels=[0, 1]).ravel()
                    roc_auc = auc(fpr, tpr)
                    auc_list.append(roc_auc)
                    tn_list.append(tn)
                    fp_list.append(fp)
                    fn_list.append(fn)
                    tp_list.append(tp)
                    if not no_save:
                        out_filename = output_files[i]
                        mask_no = mask_regex.match(os.path.split(out_filename)[1]).groups()[0]
                        assert int(file_no) == int(mask_no), "file number and mask number not match"
                        result = mask_to_image(save_mask)
                        result = result.resize((width, height))
                        result.save(out_filename)
                        # print(f'Mask saved to {out_filename}')
                except RuntimeError as re:
                    print('img:{} predict encounter error:{}'.format(filename, str(re)))
            print("average time:{}".format(sum(time_list) / len(time_list)))
            print("average auc: {}".format(sum(auc_list) / len(auc_list)))
            print("average tp: {}".format(sum(tp_list) / len(tp_list)))
            print("average fp: {}".format(sum(fp_list) / len(fp_list)))
            print("average tn: {}".format(sum(tn_list) / len(tn_list)))
            print("average fn: {}".format(sum(fn_list) / len(fn_list)))
            print("average pixel: {}".format(sum(pixel_list) / len(pixel_list)))
