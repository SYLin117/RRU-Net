from tqdm import tqdm
from utils import *
import re
from glob import glob
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def predict_img(net,
                full_img,
                resize=(300, 300),
                use_gpu=True):
    # net.eval()

    # img = resize_and_crop(full_img, scale=scale_factor).astype(np.float32)
    # img = np.transpose(normalize(img), (2, 0, 1))
    # img = torch.from_numpy(img).unsqueeze(dim=0)
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
        mask = net(img)
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
    network = 'Ringed_Res_Unet'
    resize = (300, 300)
    ###############################################PREDICT FILES(FOLDER)################################################
    # in_files = os.path.join(current_path, 'data', 'video_test', 'images')
    # in_files = os.path.join(DATASETS_DIR, 'total_forge', 'train_and_test', 'test', 'images')  # total
    # gt_files = os.path.join(DATASETS_DIR, 'total_forge', 'train_and_test', 'test', 'masks')  # total
    # in_files = os.path.join(DATASETS_DIR, 'total_forge', 'CM', 'test_and_train', 'test', 'images')  # copy-move
    # gt_files = os.path.join(DATASETS_DIR, 'total_forge', 'CM', 'test_and_train', 'test', 'masks')  # copy-move
    in_files = os.path.join(DATASETS_DIR, 'total_forge', 'SP', 'test_and_train', 'test', 'images')  # splicing
    gt_files = os.path.join(DATASETS_DIR, 'total_forge', 'SP', 'test_and_train', 'test', 'masks')  # splicing
    ##########################################COPY-MOVE MODEL###########################################################
    model = os.path.join(current_path, 'result', 'logs', 'Total', network, 'checkpoint_epoch50.pth')
    # model = os.path.join(current_path, 'result', 'logs', 'large_cm', network, 'checkpoint_epoch47.pth')
    # model = os.path.join(current_path, 'result', 'logs', 'SP', network, 'epoch50.pth')
    ##########################################SPLICING MODEL###########################################################
    # model = os.path.join(current_path, 'result', 'logs', 'SP',
    #                      'Ringed_Res_Unet', 'epoch50.pth')
    if network == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif network == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif network == 'Ringed_Res_Unet':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)
    elif network == 'MyTransUnet':
        net = MyTransUNet(in_channels=3, classes=1, img_dim=resize[0])
    elif network == 'MyTransUnet2':
        net = MyTransUNet2(in_channels=3, classes=1, img_dim=resize[0])
    else:
        raise Exception("model not implements.")

    if not cpu:
        net.cuda()
        net.load_state_dict(torch.load(model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    # mask = predict_img(net=net,
    #                    full_img=img,
    #                    scale_factor=scale,
    #                    out_threshold=mask_threshold,
    #                    use_gpu=not cpu)

    if isinstance(in_files, str):
        if os.path.isfile(in_files):
            print("path is file")
        elif os.path.isdir(in_files):  # in_files是資料夾
            auc_list = list()
            time_list = list()
            input_files = glob(os.path.join(in_files, '*.*'))
            gt_masks = glob(os.path.join(gt_files, '*.*'))

            input_files = sorted(input_files)
            gt_masks = sorted(gt_masks)

            # print("input folder got {} files".format(len(input_files)))
            output_files = get_output_filenames(input_files)
            for i, filename in enumerate(tqdm(input_files)):
                img_regex = re.compile("([0-9]+)\.")
                mask_regex = re.compile("([0-9]+)_OUT\.")
                gt_filename = gt_masks[i]
                gt_no = img_regex.match(os.path.split(gt_filename)[1]).groups()[0]
                file_no = img_regex.match(os.path.split(filename)[1]).groups()[0]
                assert int(file_no) == int(gt_no)
                gt_mask = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
                (_, gt_mask) = cv2.threshold(gt_mask, 125, 1, cv2.THRESH_BINARY)
                # gt_mask = np.where(gt_mask >= 1, 1, 0)
                cv2.imshow('mask:'.format(gt_no), gt_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                # print(f'\nPredicting image {filename} ...')
                # img = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                img = Image.open(filename).convert('RGB')  # cause some image got 4 channel(RGBA), eg:001221 file
                width, height = img.size

                try:
                    start_time = time.time()
                    mask = predict_img(net=net,
                                       full_img=img,
                                       resize=resize,
                                       use_gpu=not cpu)
                    # print("predict time:{}".format(time.time() - start_time))
                    time_list.append(time.time() - start_time)
                    save_mask = mask > mask_threshold
                    mask = cv2.resize(mask, (gt_mask.shape[1], gt_mask.shape[0]), cv2.INTER_CUBIC)
                    y_test = gt_mask.flatten()
                    y_pred = mask.flatten()
                    fpr, tpr, _ = roc_curve(y_test, y_pred)
                    roc_auc = auc(fpr, tpr)
                    auc_list.append(roc_auc)
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
    # if viz:
    #     print("Visualizing results for image {}, close to continue ...".format(j))
    #     plot_img_and_mask(img, mask)

    # if not no_save:
    #     result = mask_to_image(mask)
    #
    #     if network == 'Unet':
    #         result.save('predict_u.png')
    #     elif network == 'Res_Unet':
    #         result.save('predict_ru.png')
    #     else:
    #         result.save('predict_rru.png')
