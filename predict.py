from tqdm import tqdm
from utils import *
import re
from glob import glob
from pathlib import Path


def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_gpu=True):
    # net.eval()

    # img = resize_and_crop(full_img, scale=scale_factor).astype(np.float32)
    # img = np.transpose(normalize(img), (2, 0, 1))
    # img = torch.from_numpy(img).unsqueeze(dim=0)
    tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize([300, 300], Image.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
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

    return mask > out_threshold


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
    scale, mask_threshold, cpu, viz, no_save = 1, 0.5, False, False, False
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'

    current_path = str(pathlib.Path().resolve())
    network = 'Unet'
    ###############################################PREDICT FILES(FOLDER)################################################
    # in_files = os.path.join(current_path, 'data', 'video_test', 'images')
    in_files = os.path.join(DATASETS_DIR, 'total_forge', 'CM', 'test_and_train', 'test', 'images')  # copy-move
    # in_files = os.path.join('/media', 'ian', 'WD', 'datasets', 'total_forge', 'SP', 'test_and_train', 'test', 'images') # splicing
    ##########################################COPY-MOVE MODEL###########################################################
    model = os.path.join(current_path, 'result', 'logs', 'large_cm', network, 'checkpoint_epoch4.pth')
    # model = os.path.join(current_path, 'result', 'logs', 'SP', network, 'epoch50.pth')
    ##########################################SPLICING MODEL###########################################################
    # model = os.path.join(current_path, 'result', 'logs', 'SP',
    #                      'Ringed_Res_Unet', 'epoch50.pth')
    if network == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif network == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    elif network == 'RUU_Net':
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)
    elif network == 'TransUnet':
        net = MyTransUNet(in_channels=3, classes=1)

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
        elif os.path.isdir(in_files):
            input_files = glob(os.path.join(in_files, '*.*'))
            print("input folder got {} files".format(len(input_files)))
            output_files = get_output_filenames(input_files)
            for i, filename in enumerate(tqdm(input_files)):
                img_regex = re.compile("([0-9]+)\.")
                mask_regex = re.compile("([0-9]+)_OUT\.")
                file_no = img_regex.match(os.path.split(filename)[1]).groups()[0]
                # print(f'\nPredicting image {filename} ...')
                # img = cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                img = Image.open(filename).convert('RGB')  # cause some image got 4 channel(RGBA)
                width, height = img.size

                # if file_no == '001221': # this file got 4 channel
                #     print('001221.tif shape:{}'.format(img.size))
                try:
                    mask = predict_img(net=net,
                                       full_img=img,
                                       scale_factor=scale,
                                       out_threshold=mask_threshold,
                                       use_gpu=not cpu)
                    if not no_save:
                        out_filename = output_files[i]
                        mask_no = mask_regex.match(os.path.split(out_filename)[1]).groups()[0]
                        assert int(file_no) == int(mask_no), "file number and mask number not match"
                        result = mask_to_image(mask)
                        result = result.resize((width, height))
                        result.save(out_filename)
                        # print(f'Mask saved to {out_filename}')
                except RuntimeError as re:
                    print('img:{} predict encounter error:{}'.format(filename, str(re)))
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
