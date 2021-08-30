import torch
import numpy as np
import os
import argparse
import time
from medical_data_load.dataset import load_from_pkl
from medical_data_load.dataload import datanih

from data import *
import tools
from data.config import anchor_size_3D_try
from lyi.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')

    parser.add_argument('-d', '--dataset', default='pnens',
                        help='voc, coco-val.')
    parser.add_argument('--trained_model', default='E:\\data\\data4T\\ly\\data\\pnens\\yolo_v3\\',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.3, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.50, type=float,
                        help='NMS threshold')
    return parser


def see_feature_segment():
    args = parse_args().parse_args(args=[])
    device = torch.device("cpu")

    input_size = [128, 128, 64]

    # dataset
    num_classes = 1
    class_names = ('pnens')
    # dataset2 = load_from_pkl(r'/data/liyi219/pnens_3D_data/after_dealing/pre_order0_128_128_64_new.pkl')
    dataset2 = load_from_pkl(r'E:\data\pre_order0_128_128_64_new.pkl')
    dataset = datanih(dataset2)

    # load net
    from models.yolo_v3 import myYOLOv3

    anchor_size = anchor_size_3D_try
    net = myYOLOv3(device, input_size=input_size, num_classes=1, conf_thresh=args.conf_thresh,
                   nms_thresh=args.nms_thresh, anchor_size=anchor_size)

    net.load_state_dict(torch.load(os.path.join(args.trained_model + 'yolo_v3_200.pth'), map_location=device))
    net.to(device).eval()
    num_images = len(dataset)
    result = []
    images = []
    for index in range(num_images):
        if index == 1: break
        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        img, _, height, width, depth = dataset.pull_item(index)
        scale = np.array([[height, width, depth, height, width, depth]])
        images.append(img)
        # to tensor
        x = img
        x = x.unsqueeze(0).to(device)
        t0 = time.time()
        # forward
        bboxes, scores, _ = net(x)
        pred_1, pred_2, pred_3 = net.get_feature_map(x)
        bboxes = bboxes * scale
        if not scores.size:
            print('{} has not bbox'.format(index))
            continue
        best_scores = np.argmax(scores, axis=0)
        bboxes = bboxes[best_scores]
        result.append([bboxes, scores])
        print("detection {} time used ".format(index), time.time() - t0, "s")

    img3D = images[0]
    img3D = img3D.squeeze(dim=0)
    img3D = np.array(img3D)
    pred_1 = torch.sigmoid(pred_1)
    pred_1 = pred_1.data.cpu().numpy()[0, 0]
    show3D(pred_1)
    img_resize = resize3D(pred_1, [128, 128, 64])
    # show3Dslice(img_resize)
    img_and_fearturemap = np.hstack((img_resize, img3D))
    show3Dslice(img_and_fearturemap)

def get_targets(targets):
    """
    制作标签
    @param targets: [y1, y2, x1, x2, z1, z2, 0] 相对
    @return:  [batch_index, featuremap_index， grid_y* grid_x * grid_z * ab_ind, 参数（15）]
             (15): [obj, class, tx, ty, tz, tw, th, td, weight, (xmin, ymin, zmin, xmax, ymax, zmax)(绝对)]
    """
    targets = np.array(targets).reshape(1, -1)
    targets = list(targets)
    targets = tools.multi_gt_creator3D(input_size=[128, 128, 64],
                                       strides=[8, 16, 32],
                                       label_lists=targets,
                                       anchor_size=anchor_size_3D_try
                                       )
    # [batch_index, featuremap_index， grid_x* grid_y * grid_z * ab_ind, 参数（15）]
    # (15): [obj, class, tx, ty, tz, th, tw, td, weight, (xmin, ymin, zmin, ymax, xmax, zmax)(绝对)]
    targets = torch.tensor(targets).float()

    return targets

def obj_map_get(obj):
    obj_1 = obj[:, :6144].reshape(3, 16, 16, 8)
    obj_2 = obj[:, 6144:6912].reshape(3, 8, 8, 4)
    obj_3 = obj[:, 6912:].reshape(3, 4, 4, 2)

    # obj位置
    index_obj = torch.argmax(obj)
    if index_obj < 6144:
        obj_1 = obj_1.cpu().numpy()
        index_obj = np.where(obj_1 == 1)
    elif 6144 <= index_obj and index_obj < 6912:
        obj_2 = obj_2.cpu().numpy()
        index_obj = np.where(obj_2 == 1)
    else:
        obj_3 = obj_3.cpu().numpy()
        index_obj = np.where(obj_3 == 1)

    return index_obj


def see_bbox():
    args = parse_args().parse_args(args=[])
    device = torch.device("cpu")
    input_size = [128, 128, 64]
    # dataset2 = load_from_pkl(r'/data/liyi219/pnens_3D_data/after_dealing/pre_order0_128_128_64_new.pkl')
    dataset2 = load_from_pkl(r'E:\ly\pnens_data\nas_data\v1_data\NIH\pre_order0_128_128_64_new.pkl')
    dataset = datanih(dataset2)

    # load net
    from models.yolo_v3 import myYOLOv3

    anchor_size = anchor_size_3D_try
    net = myYOLOv3(device, input_size=input_size, num_classes=1, conf_thresh=args.conf_thresh,
                   nms_thresh=args.nms_thresh, anchor_size=anchor_size)

    net.load_state_dict(torch.load(os.path.join(args.trained_model + 'yolo_v3_200.pth'), map_location=device))
    net.to(device).eval()
    num_images = len(dataset)
    result = []
    images = []
    masks = []
    bbox_gt = []
    for index in range(num_images):
        if index == 1: break

        # target
        im, gt = dataset.__getitem__(index)
        targets = get_targets(gt)
        # gt [[obj, class, tx, ty, tz, tw, th, td, weight, (xmin, ymin, zmin, xmax, ymax, zmax)(绝对)]]
        gt = targets[np.where(targets==1)[0:2]]
        txtytztwthtd = gt[0, 2:8]
        gt_obj = targets[:, :, 0]
        map_obj = obj_map_get(gt_obj)

        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        img, _, mask, height, width, depth = dataset.pull_item(index)
        bb = dataset.get_bbox_juedui(index)  # [y1, y2, x1, x2, z1, z2] 绝对
        bbox_gt.append(bb)
        scale = np.array([[width, height, depth, width, height, depth]])
        images.append(img)
        masks.append(mask)
        # to tensor
        x = torch.tensor(img)
        x = x.unsqueeze(0).to(device)
        x = x.unsqueeze(0).to(device)
        t0 = time.time()
        # forward
        bboxes, scores, _ = net(x)  # bboxes[x1, y1, z1, x2, y2, z2] 相对
        bboxes = bboxes * scale  # to绝对
        if not scores.size:
            print('{} has not bbox'.format(index))
            continue
        best_scores = np.argmax(scores, axis=0)
        bboxes = bboxes[best_scores]
        # print(scores[best_scores])
        # print([bboxes, best_scores])
        result.append([bboxes, best_scores])
        print("detection {} time used ".format(index), time.time() - t0, "s")

    if False:
        pred_1 = torch.sigmoid(pred_1)
        pred_1 = pred_1.data.cpu().numpy()[0, 0]
        img3D = pred_1
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img3D), name='3-d ultrasound ')
        mlab.colorbar(orientation='vertical')
        mlab.show()
        show3Dslice(img3D)
        img_resize = resize3D(img3D, [128, 64, 64])
        show3Dslice(img_resize)
        img = img.squeeze(0).data.cpu().numpy()

        show3Dslice(img)

    test_index = 0
    # img3D[y, x, z]
    img3D = images[test_index]
    mask3D = masks[test_index]

    # show3D(img3D)
    # show3Dslice(img3D)

    # 画bbox -- [x1, y1, z1, x2, y2, z2] 绝对
    line_thick = 1
    bbox3D = result[test_index][0]
    x1, y1, z1, x2, y2, z2 = bbox3D
    bbox3D = [y1, x1, z1, y2, x2, z2]
    bbox3D = np.floor(bbox3D)
    bbox3D = np.array(bbox3D, dtype=int)
    # show3Dbbox_img(img3D, bbox3D, 2)
    bboxInimg = bbox_in_img_for_slice(img3D, bbox3D, line_thick, -2e3)
    # show3Dslice(bboxInimg)
    bb_gt = bbox_gt[test_index]  # [y1, y2, x1, x2, z1, z2] 绝对
    y1, y2, x1, x2, z1, z2 = bb_gt
    bb_gt = [y1, x1, z1, y2, x2, z2]
    bb_gt = np.floor(bb_gt)
    bb_gt = np.array(bb_gt, dtype=int)
    print(bbox3D)
    print(bb_gt)
    bboxInimg = bbox_in_img_for_slice(bboxInimg, bb_gt, line_thick, 1e3)
    bboxInimg = bboxInimg + mask3D*2e3
    # bbox_in_img = np.hstack((bbox_in_img, img3D))
    show3Dslice(bboxInimg)
    # show3D(bboxInimg)

    img_for_3D = np.zeros([128, 128, 64], dtype=np.int)
    img_for_3D = bbox_in_img_for_3D(img_for_3D, bb_gt, line_thick, 100)
    img_for_3D = bbox_in_img_for_3D(img_for_3D, bbox3D, line_thick, 200)
    show3D(img_for_3D)

def obj_test():
    args = parse_args().parse_args(args=[])
    device = torch.device("cpu")

    input_size = [128, 128, 64]

    # dataset
    class_names = ('pnens')
    # dataset2 = load_from_pkl(r'/data/liyi219/pnens_3D_data/after_dealing/pre_order0_128_128_64_new.pkl')
    dataset2 = load_from_pkl(r'E:\data\pre_order0_128_128_64_new.pkl')
    dataset = datanih(dataset2)

    # load net
    from models.yolo_v3 import myYOLOv3

    anchor_size = anchor_size_3D_try
    net = myYOLOv3(device, input_size=input_size, num_classes=1, conf_thresh=args.conf_thresh,
                   nms_thresh=args.nms_thresh, anchor_size=anchor_size)

    net.load_state_dict(torch.load(os.path.join(args.trained_model + 'yolo_v3_200.pth'), map_location=device))
    net.to(device).eval()
    num_images = len(dataset)
    result = []
    images = []
    bbox_gt = []
    for index in range(num_images):
        if index == 4: break
        print('Testing image {:d}/{:d}....'.format(index + 1, num_images))
        img, _, height, width, depth = dataset.pull_item(index)
        bb = dataset.get_bbox_juedui(index)
        bbox_gt.append(bb)
        scale = np.array([[height, width, depth, height, width, depth]])
        images.append(img)
        # to tensor
        x = img
        x = x.unsqueeze(0).to(device)
        t0 = time.time()
        # forward
        bboxes, scores, _ = net(x)
        bboxes = bboxes * scale
        if not scores.size:
            print('{} has not bbox'.format(index))
            continue
        best_scores = np.argmax(scores, axis=0)
        bboxes = bboxes[best_scores]
        # print(scores[best_scores])
        # print([bboxes, best_scores])
        result.append([bboxes, best_scores])
        print("detection {} time used ".format(index), time.time() - t0, "s")

    if False:
        pred_1 = torch.sigmoid(pred_1)
        pred_1 = pred_1.data.cpu().numpy()[0, 0]
        img3D = pred_1
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(img3D), name='3-d ultrasound ')
        mlab.colorbar(orientation='vertical')
        mlab.show()
        show3Dslice(img3D)
        img_resize = resize3D(img3D, [128, 64, 64])
        show3Dslice(img_resize)
        img = img.squeeze(0).data.cpu().numpy()

        show3Dslice(img)

    test_index = 1
    img3D = images[test_index]
    img3D = img3D.squeeze(dim=0)
    img3D = np.array(img3D)
    # show3D(img3D)
    # show3Dslice(img3D)

    # 画bbox -- [x1, y1, z1, x2, y2, z2]
    line_thick = 1
    bbox3D = result[test_index][0]
    # print(bbox3D)
    # print(result[0][1])
    bbox3D = np.floor(bbox3D)
    bbox3D = np.array(bbox3D, dtype=int)
    # show3Dbbox_img(img3D, bbox3D, 2)
    bbox_in_img = get_bbox_in_img(img3D, bbox3D, line_thick)
    bb_gt = bbox_gt[test_index]
    x1, x2, y1, y2, z1, z2 = bb_gt
    bb_gt = [x1, y1, z1, x2, y2, z2]
    bb_gt = np.floor(bb_gt)
    bb_gt = np.array(bb_gt, dtype=int)
    print(bbox3D)
    print(bb_gt)
    bbox_in_img = get_bbox_in_img(bbox_in_img, bb_gt, line_thick, line_value=2e3)
    # bbox_in_img = np.hstack((bbox_in_img, img3D))
    show3Dslice(bbox_in_img)

if __name__ == '__main__':
    see_bbox()
    # see_feature_segment()
