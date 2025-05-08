import os.path as osp
import pickle
import shutil
import tempfile
from collections import OrderedDict, namedtuple

import cv2
import flow_vis
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.image import tensor2imgs
from mmengine.dist import get_dist_info
from mmengine.logging import print_log
from pytorch_msssim import ssim
from torch.autograd import Variable


def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name


Label = namedtuple("Label", ["color", "id", "name", "minsize"])


def add_new_label(color, id, name, minsize=0):
    return Label(color, id, name, minsize)


def warp_flow_val(img, flow):
    flow = -flow
    # pre_image = torch.from_numpy(img).float()  # 579531041
    pre_image = img.float()  # 579531041
    image = pre_image.unsqueeze(0).permute(0, 3, 1, 2)
    # image = Variable(image, requires_grad=True)
    numpy_inp = False
    if isinstance(flow, (np.ndarray, np.generic)):
        torch_flow = torch.from_numpy(flow).unsqueeze(0).cuda()
        B, H, W, C = torch_flow.size()
        numpy_inp = True
    else:
        torch_flow = flow.unsqueeze(0).cuda()
        B, C, H, W = torch_flow.size()

    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float().cuda()
    # print("vgrid size", vgrid.size())
    if numpy_inp:
        torch_flow = torch_flow.permute(0, 3, 1, 2)
    vgrid += torch_flow

    # scale grid to [-1,1]
    vgrid_x = vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid_y = vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3).cuda()

    output = F.grid_sample(image, vgrid_scaled, mode="nearest", align_corners=True)
    res = output.permute(0, 2, 3, 1).squeeze(0)

    return res


def label_array_to_color_array_cityscape(label):
    index = np.where(label == 255)
    label[index] = 19
    color_map = np.array(
        [
            [128, 64, 128],  # 0 Road
            [232, 35, 244],  # 1 Sidewalk
            [70, 70, 70],  # 2 Building
            [156, 102, 102],  # 3 Wall
            [153, 153, 190],  # 4 Fence
            [153, 153, 153],  # 5 Pole
            [30, 170, 250],  # 6 Traffic Light
            [0, 220, 220],  # 7 Traffic Sign
            [35, 142, 107],  # 8 Vegetation
            [152, 251, 152],  # 9 Terrain*
            [180, 130, 70],  # 10 Sky
            [60, 20, 220],  # 11  Person
            [0, 0, 255],  # 12  Rider
            [142, 0, 0],  # 13 Car
            [70, 0, 0],  # 14  Truck
            [100, 60, 0],  # 15 Bus
            [100, 80, 0],  # 16 Train
            [230, 0, 0],  # 17 Motorcycle
            [32, 11, 119],
        ]
    )  # 18 Bicycle

    return color_map[label.astype(np.int)]


labels_19_IDtoRGB = [
    add_new_label((128, 64, 128), 0, "Road", 1000),
    add_new_label((244, 35, 232), 1, "Sidewalk", 1000),
    add_new_label((70, 70, 70), 2, "Building", 1000),
    add_new_label((102, 102, 156), 3, "Wall", 1000),
    add_new_label((190, 153, 153), 4, "Fence", 500),
    add_new_label((153, 153, 153), 5, "Pole", 50),
    add_new_label((250, 170, 30), 6, "Traffic Light", 30),
    add_new_label((220, 220, 0), 7, "Traffic Sign", 30),
    add_new_label((107, 142, 35), 8, "Vegetation", 1000),
    add_new_label((152, 251, 152), 9, "Terrain", 1000),
    add_new_label((70, 130, 180), 10, "Sky", 1000),
    add_new_label((220, 20, 60), 11, "Person", 100),
    add_new_label((255, 0, 0), 12, "Rider", 100),
    add_new_label((0, 0, 142), 13, "Car", 300),
    add_new_label((0, 0, 70), 14, "Truck", 300),
    add_new_label((0, 60, 100), 15, "Bus", 300),
    add_new_label((0, 80, 100), 16, "Train", 300),
    add_new_label((0, 0, 230), 17, "Motorcycle", 100),
    add_new_label((119, 11, 32), 18, "Bicycle", 50),
    # Label((0, 0, 0), 255, 'Background'),
]

# label for kia
## Labels of 19 CityScapes classes with RGB values and class IDs
labels_33_IDtoRGB = [
    add_new_label((100, 90, 0), 0, "animal", 100),
    add_new_label((220, 20, 200), 1, "BaustellenarbeiterIn", 100),
    add_new_label((220, 20, 60), 2, "FußgängerIn", 100),
    add_new_label((202, 20, 100), 3, "Gehhilfe", 100),
    add_new_label((220, 20, 0), 4, "Kind", 50),
    add_new_label((220, 20, 175), 5, "Kinderwagen", 100),
    add_new_label((220, 20, 225), 6, "Polizistin", 500),
    add_new_label((220, 20, 150), 7, "RollstuhlfahrerIn", 100),
    add_new_label((255, 64, 64), 8, "FahradfahrerIn", 100),
    add_new_label((255, 0, 0), 9, "MotorradfahrerIn", 100),
    add_new_label((0, 0, 142), 10, "Auto", 300),
    add_new_label((0, 0, 110), 11, "Anhänger", 100),
    add_new_label((0, 0, 80), 12, "Baustellenfahrzeug", 100),
    add_new_label((0, 60, 100), 13, "Bus", 300),
    add_new_label((119, 11, 32), 14, "Fahrrad", 100),
    add_new_label((0, 0, 150), 15, "Krankenwagen", 100),
    add_new_label((0, 0, 70), 16, "LKW", 100),
    add_new_label((0, 0, 230), 17, "Motorrad", 200),
    add_new_label((0, 0, 155), 18, "Polizeiauto", 300),
    add_new_label((111, 74, 0), 19, "Barriere", 300),
    add_new_label((250, 170, 160), 20, "Parkfläche", 300),
    add_new_label((153, 153, 153), 21, "Pole", 300),
    add_new_label((250, 170, 30), 22, "Ampel", 300),
    add_new_label((220, 220, 0), 23, "Verkehrsschild", 300),
    add_new_label((128, 64, 128), 24, "road", 100),
    add_new_label((244, 35, 232), 25, "sidewalk", 200),
    add_new_label((70, 70, 70), 26, "building", 300),
    add_new_label((102, 102, 156), 27, "wall", 300),
    add_new_label((190, 153, 153), 28, "fence", 50),
    add_new_label((107, 142, 35), 29, "vegetation", 300),
    add_new_label((152, 251, 152), 30, "terrain", 300),
    add_new_label((70, 130, 180), 31, "sky", 300),
    add_new_label((0, 80, 100), 32, "train", 300),
    # Label((0, 0, 0), 255, 'Background'),
]


def get_detected_pixels(img, class_colors):
    x = Variable(torch.tensor([1.0]), requires_grad=True).cuda()
    y = Variable(torch.tensor([0.0]), requires_grad=True).cuda()

    class_positions = torch.where(
        (
            (img[:, :, 0] == class_colors[2])
            & (img[:, :, 1] == class_colors[1])
            & (img[:, :, 2] == class_colors[0])
        ),
        x,
        y,
    )
    return class_positions


def class_based_iou(warped_img, pred2, label_id, min_size=0):
    labels = labels_19_IDtoRGB
    pred1_binary = get_detected_pixels(
        warped_img, labels[label_id].color
    )  # act as gt image
    pred2_binary = get_detected_pixels(
        pred2, labels[label_id].color
    )  # act as actual image
    intersection = pred1_binary * pred2_binary  # (w*h)
    union = pred1_binary + pred2_binary - (pred1_binary * pred2_binary)
    sum_intersec = intersection.int().sum()
    sum_union = union.int().sum()

    iou = torch.tensor([-1]).cuda()
    if sum_union.item() != 0:
        iou = torch.div(sum_intersec.float(), sum_union)
    return iou, sum_intersec, sum_union


def calculate_stability_city_saaid(flow, pred_1, pred_2):
    pred_1 = torch.from_numpy(pred_1).cuda()
    pred_2 = torch.from_numpy(pred_2).cuda()
    warped_img = warp_flow_val(pred_1, flow)
    error = pred_1 == pred_2

    stabilities = OrderedDict()
    class_count = 0
    sum_iou = 0
    labels = labels_19_IDtoRGB
    for label in labels:
        iou_class, intersec, union = class_based_iou(
            warped_img, pred_2, label.id, label.minsize
        )
        stabilities[label.name] = (iou_class.item(), intersec.item(), union.item())
        if iou_class.item() != -1:
            sum_iou = sum_iou + iou_class.item()
            class_count = class_count + 1
    tc = sum_iou / class_count
    return tc, stabilities, error


def ssim_image(out_file, prev, current, flow):
    prev_1 = torch.from_numpy(prev).cuda()
    curr_2 = torch.from_numpy(np.transpose(current, (2, 0, 1))).cuda()
    warped_img = warp_flow_val(prev_1, flow)
    curr_2 = curr_2.double()
    warped_img = warped_img.double()
    warped_img = warped_img.transpose(2, 0)
    warped_img = warped_img.transpose(1, 2)
    # print("warped", warped_img.shape)
    # print("img shape ", curr_2.shape)

    step = 40
    error_im = torch.zeros((warped_img.shape[1], warped_img.shape[2])).cuda()
    with torch.no_grad():
        for i in range(warped_img.shape[1] // step):
            for j in range(warped_img.shape[2] // step):
                ssim_val = ssim(
                    warped_img[
                        :, step * i : step * (i + 1), step * j : step * (j + 1)
                    ].unsqueeze(0),
                    curr_2[
                        :, step * i : step * (i + 1), step * j : step * (j + 1)
                    ].unsqueeze(0),
                    data_range=255,
                    size_average=False,
                    win_size=11,
                )
                error_im[step * i : step * (i + 1), step * j : step * (j + 1)] = (
                    ssim_val
                )
    thresh = 0.6
    error_im = error_im.cpu().numpy().astype(np.float32)
    mask = error_im < thresh
    mask_f = error_im >= thresh
    error_im[mask_f] = thresh
    error_im = cv2.normalize(
        src=error_im,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    color_seg = cv2.applyColorMap(error_im, cv2.COLORMAP_AUTUMN)
    # img = img * 0.6 + color_seg * 0.4

    ##thresholding
    # current[mask] = current[mask] * 0.6 + color_seg[mask] * 0.4
    # warped_img = warped_img[:1024, :2048, :]
    # warped_img = mmcv.imresize(warped_img, (2048, 1024))
    warped_img = np.transpose(
        warped_img.cpu().squeeze().numpy().astype(np.uint8), (1, 2, 0)
    )
    warped_img[mask] = warped_img[mask] * 0.6 + color_seg[mask] * 0.4
    img = warped_img.astype(np.uint8)
    mmcv.imwrite(img, out_file + "_ssim_error_warped.png")
    mmcv.imwrite(current, out_file + ".png")
    mmcv.imwrite(prev, out_file + "_t-1.png")


def stability_image(out_file, img, flow, pred_1, pred_2):
    pred_1 = torch.from_numpy(pred_1).cuda()
    pred_2 = torch.from_numpy(pred_2).cuda()
    warped_img = warp_flow_val(pred_1, flow)

    stabilities = OrderedDict()
    class_count = 0
    sum_iou = 0
    labels = labels_19_IDtoRGB
    error_im = np.zeros((pred_1.shape[0], pred_1.shape[1]), np.float32)
    step = 20
    with torch.no_grad():
        for i in range(pred_1.shape[0] // step):
            for j in range(pred_1.shape[1] // step):
                for label in labels:
                    # print("shape warped", warped_img[20*i:20*(i+1), 20*j:20*(j+1)].shape)
                    # print("shape pred_2", pred_2[10*i:10*(i+1), 10*j:10*(j+1)].shape)
                    iou_class, intersec, union = class_based_iou(
                        warped_img[
                            step * i : step * (i + 1), step * j : step * (j + 1)
                        ],
                        pred_2[step * i : step * (i + 1), step * j : step * (j + 1)],
                        label.id,
                        label.minsize,
                    )
                    stabilities[label.name] = (
                        iou_class.item(),
                        intersec.item(),
                        union.item(),
                    )
                    if iou_class.item() != -1:
                        sum_iou = sum_iou + iou_class.item()
                        class_count = class_count + 1
                tc = sum_iou / class_count
                sum_iou = 0
                class_count = 0
                # print("tc patch" ,tc)
                error_im[step * i : step * (i + 1), step * j : step * (j + 1)] = tc
    # import matplotlib.pyplot as plt
    # import cv2
    # fig, axs = plt.subplots(nrows=1, ncols=3,
    #                     subplot_kw={'xticks': [], 'yticks': []})
    # axs[0].set_title("tc image")
    # z1_plot = axs[0].imshow(error_im[0:1000,0:2000])
    # axs[1].set_title("t-1 pred")
    # axs[1].imshow(pred_1.cpu().numpy()[0:1000,0:2000])
    # axs[2].set_title("t pred")
    # axs[2].imshow(pred_2.cpu().numpy()[0:1000,0:2000])
    # plt.colorbar(z1_plot,cax=axs[3])
    # color_seg = pred_2[..., ::-1]
    # from IPython import embed; embed(header='debug vis')
    # colormap = plt.get_cmap('inferno')
    # color_seg = (colormap(error_im) * 2**16).astype(np.uint16)[:,:,:3]
    # color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)
    np.save(out_file + "tc_np.npy", error_im)
    thresh = 0.4
    mask = error_im < thresh
    mask_f = error_im >= thresh
    error_im[mask_f] = thresh
    error_im = cv2.normalize(
        src=error_im.astype(np.float32),
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    color_seg = cv2.applyColorMap(error_im, cv2.COLORMAP_AUTUMN)
    # img = img * 0.6 + color_seg * 0.4

    ##thresholding
    img[mask] = img[mask] * 0.6 + color_seg[mask] * 0.4
    img = img.astype(np.uint8)
    mmcv.imwrite(img, out_file + "_tc_error.png")
    # plt.savefig("/data/TC/temporal_consistency/work_dirs/example.png", dpi=1000)
    return error_im


def eval_tc(out_dir, data, seq, seq_flo, mode="val"):
    labels = labels_19_IDtoRGB
    stabilities = OrderedDict()
    stabilities["general"] = OrderedDict()
    stability_values = OrderedDict()

    for label in labels:
        stabilities[label.name] = OrderedDict()
        stability_values[label.name] = OrderedDict()
    img_arr = []
    meantc = 0.0
    i = 0
    t = -5
    # print("length of list ", len(seq))
    # img_tensor = data['seq'][0]
    # img_metas = data['img_metas'][0].data[0]
    # img_meta = img_metas[0]
    #         #print(img_tensor[0].shape)
    # imgs = []
    # previous_im = None
    # for im in img_tensor:
    #     if len(img_tensor.shape) > 4:
    #         im = im[:, :, -1]
    #     imgs.append(tensor2imgs(im, **img_metas[0]['img_norm_cfg'])[0])
    stabilities_o = []
    TC_o = []
    for pred, flow in zip(seq, seq_flo):
        i += 1
        # if out_dir is not None:
        # out_file = osp.join(out_dir, img_meta['ori_filename'].split(".png")[0] + "error_t_" + str(t) + ".png")
        # out_file = osp.join(out_dir, img_meta['ori_filename'].split("converted/")[1].split(".ppm")[0] + "_t_" + str(t) + "_")
        # print(out_file)
        # t+=1
        # h, w, _ = img_meta['img_shape']
        # img_show = im[:h, :w, :]

        # ori_h, ori_w = img_meta['ori_shape'][:-1]
        # img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        if i == 1:
            # print(pred.size()[-1])
            # print("Pred shape", pred[0].shape)
            # if pred.size()[-2] != size[2] or pred.size()[-1] != size[3]:
            #     pred = nn.functional.interpolate(pred, (size[-2], size[-1]),
            #                                      mode='bilinear')
            # pred = pred.cpu()
            # preds = np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)
            # pred_softmax = nn.functional.softmax(pred, dim=1)
            preds = label_array_to_color_array_cityscape(np.asarray(pred[0]))
            previous = preds

            # previous_im = img_show
            continue
        # if isinstance(pred, (list, tuple)):
        #     pred = pred[-1]
        # if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
        #     pred = nn.functional.interpolate(pred, (size[-2], size[-1]),
        #                   mode='bilinear')
        # pred = pred.cpu()
        # preds = np.asarray(np.argmax(pred, axis=1), dtype=np.uint8)
        # pred_softmax = nn.functional.softmax(pred, dim=1)
        preds = label_array_to_color_array_cityscape(np.asarray(pred[0]))
        TC, single_stabilities, error = calculate_stability_city_saaid(
            flow[0], previous, preds
        )
        stabilities_o.append(single_stabilities)
        TC_o.append(TC)
        # stability_image(out_file, img_show, flow[0], previous, preds)
        # ssim_image(out_file,previous_im, img_show, flow[0])
        previous = preds
        # previous_im = img_show

    return TC_o, stabilities_o, error


def collect_tc(TC, stabilites_c):
    rank, world_size = get_dist_info()
    if rank == 0:
        labels = labels_19_IDtoRGB
        stabilities = OrderedDict()
        stabilities["general"] = OrderedDict()
        stability_values = OrderedDict()
        for label in labels:
            stabilities[label.name] = OrderedDict()
            stability_values[label.name] = OrderedDict()

        # t = 0

        for single_stabilities in stabilites_c:
            for label in labels:
                if not stability_values[label.name]:
                    print(label.name)
                    stability_values[label.name] = (
                        single_stabilities[label.name][1],
                        single_stabilities[label.name][2],
                    )
                else:
                    # t +=1
                    # print("adding values to stability values", t)
                    stability_values[label.name] = (
                        stability_values[label.name][0]
                        + single_stabilities[label.name][1],
                        stability_values[label.name][1]
                        + single_stabilities[label.name][2],
                    )
            # previous = preds
        # print("stabilty values dict", stability_values)
        conf_matrix = np.zeros((len(labels), 1), np.float32)
        per_class_mean = OrderedDict()

        for label in labels:
            if label.name == "Background":
                continue
            if stability_values[label.name]:
                if stability_values[label.name][1] > 0:
                    class_mean = (
                        np.float32(stability_values[label.name][0])
                        / stability_values[label.name][1]
                    )
                    conf_matrix[label.id] = class_mean
                    per_class_mean[label.name] = class_mean
                else:
                    conf_matrix[label.id] = np.nan
                    per_class_mean[label.name] = np.nan

        final_mean = np.nanmean(conf_matrix)
        print(conf_matrix)
        per_class_mean["final_mean"] = final_mean

        print(final_mean)
        return final_mean
    else:
        return None


def single_gpu_test(model, data_loader, show=False, out_dir=None, efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    stabils = []
    tcs = []
    show_seq = False
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if "seq" in data:
                result, seq_pred = model(return_loss=False, rescale=True, **data)
                TC, single_stabilites, errorim = eval_tc(
                    out_dir, data, seq_pred, seq_flo=data["seq_flo"][0]
                )
                stabils.extend(single_stabilites)

                tcs.append(TC)

            else:
                result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data["img"][0]
            img_metas = data["img_metas"][0].data[0]
            if len(img_tensor.shape) > 4:
                img_tensor = img_tensor[:, :, -1]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta["img_shape"]
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta["ori_shape"][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    if "ppm" in img_meta["ori_filename"]:
                        out_file = osp.join(
                            out_dir,
                            img_meta["ori_filename"]
                            .split("converted/")[1]
                            .split(".ppm")[0]
                            + ".png",
                        )
                        out_file_err = osp.join(
                            out_dir,
                            img_meta["ori_filename"]
                            .split("converted/")[1]
                            .split(".ppm")[0]
                            + "_error.png",
                        )
                    else:
                        out_file = osp.join(out_dir, img_meta["ori_filename"])
                    print(out_file)
                else:
                    out_file = None

                error_np = errorim.cpu().numpy().astype("uint8")
                error_np = cv2.cvtColor(error_np, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(
                    out_file_err,
                    cv2.normalize(
                        error_np,
                        None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8UC1,
                    ),
                )

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                )

        if show_seq and out_dir:
            img_tensor = data["seq"][0]
            img_metas = data["img_metas"][0].data[0]

            imgs = []
            for im in img_tensor:
                imgs.append(tensor2imgs(im, **img_metas[0]["img_norm_cfg"])[0])

            t = -5
            for img, pred, flow in zip(imgs, seq_pred, data["seq_flo"][0]):
                img_meta = img_metas[0]
                h, w, _ = img_meta["img_shape"]
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta["ori_shape"][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    if "ppm" in img_meta["ori_filename"]:
                        out_file = osp.join(
                            out_dir,
                            img_meta["ori_filename"]
                            .split("converted/")[1]
                            .split(".ppm")[0]
                            + "_t_"
                            + str(t)
                            + ".png",
                        )
                        out_file_f = osp.join(
                            out_dir,
                            img_meta["ori_filename"]
                            .split("converted/")[1]
                            .split(".ppm")[0]
                            + "_flow_t_"
                            + str(t)
                            + ".png",
                        )
                    else:
                        out_file = osp.join(
                            out_dir,
                            img_meta["ori_filename"].split(".png")[0]
                            + "_t_"
                            + str(t)
                            + ".png",
                        )
                        out_file_f = osp.join(
                            out_dir,
                            img_meta["ori_filename"].split(".png")[0]
                            + "_flow_t_"
                            + str(t)
                            + ".png",
                        )

                else:
                    out_file = None

                flow_color = flow_vis.flow_to_color(
                    np.transpose(flow.squeeze().cpu().numpy(), axes=[1, 2, 0]),
                    convert_to_bgr=True,
                )
                mmcv.imwrite(flow_color, out_file_f)

                model.module.show_result(
                    img_show,
                    pred,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                )
                t += 1

        if i < 500:
            if isinstance(result, list):
                if efficient_test:
                    result = [np2tmp(_) for _ in result]
                results.extend(result)
            else:
                if efficient_test:
                    result = np2tmp(result)
                results.append(result)

        batch_size = data["img"][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    mTC = collect_tc(tcs, stabils)
    print_log("FINAL mTC" + str(mTC))
    return results, mTC


def multi_gpu_test(
    model, data_loader, tmpdir=None, gpu_collect=False, efficient_test=False
):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    tcs = []

    stabils = []
    global output
    output = []
    tc_res = []
    mTC = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if "seq" in data:
                # print(data.keys())
                result, seq_pred = model(**data, return_loss=False, rescale=True)
                # print(data.keys())
                # result = model(**data,return_loss=False, rescale=True)
                # for s_im in data["seq"]:
                # print("length of result", len(result))
                # print("lenght of seq result", len(seq_pred))
                # gather_t = [torch.ones(len(seq_pred)) for _ in range(dist.get_world_size())]
                TC, single_stabilites, errorim = eval_tc(
                    tmpdir, data, seq_pred, seq_flo=data["seq_flo"][0]
                )
                # single_stabilites = torch.tensor(single_stabilites)
                gather_t = [{} for _ in range(dist.get_world_size())]
                tmp = []
                # gather_t =
                dist.all_gather_object(gather_t, single_stabilites)
                # print(rank, single_stabilites, gather_t)
                if rank == 0:
                    # print(rank, gather_t)
                    for elem in gather_t:
                        for i in range(len(elem)):
                            output.append(elem[i])
                            tmp.append(elem[i])

                # mTC = collect_tc(tcs, output)
                # tc_res.append(mTC)

                # print("first element", type(gather_t[0][0]))
                # mTC = collect_tc(tcs, output)

                # stabils.append(single_stabilites)
                # dist.barrier()
                # if rank in stabils:
                #     stabils[rank].extend(single_stabilites)
                # else:
                #     stabils[rank] = []
                #     stabils[rank].extend(single_stabilites)
                # mTC = collect_tc(tcs, )
                tcs.extend(TC)

                # collect_tc(TC, stabils)
            else:
                result = model(return_loss=False, rescale=True, **data)
        # print(result[0].shape)
        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            # print("length of output list ", len(output))
            batch_size = data["img"][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    dist.barrier()
    if len(tcs) > 0:
        if rank == 0:
            # for k in stabils:
            #    output.extend(stabils[k])
            mTC = collect_tc(tcs, output)
            # mTC = np.mean(np.array(tcs))
            # mTC_s = np.std(np.array(tcs))
            # print_log("FINAL mTC" + str(mTC))
    dist.barrier()
    return results, mTC


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, "part_{}.pkl".format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, "part_{}.pkl".format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
