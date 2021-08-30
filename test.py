import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import numpy as np
import cv2
import tools
import time
from medical_data_load.dataset import get_dataset_NIH_pancreas, load_from_pkl
import SimpleITK as sitk

def parse_args():

    parser = argparse.ArgumentParser(description='YOLO Detection')

    parser.add_argument('-v', '--version', default='yolo_v3',
                        help='yolo_v2, yolo_v3, yolo_v3_spp, slim_yolo_v2, tiny_yolo_v3')
    parser.add_argument('-d', '--dataset', default='pnens',
                        help='voc, coco-val.')
    parser.add_argument('-size', '--input_size', default=416, type=int,
                        help='input_size')
    parser.add_argument('--trained_model', default='weight/voc/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='Confidence threshold')
    parser.add_argument('--nms_thresh', default=0.50, type=float,
                        help='NMS threshold')
    parser.add_argument('--visual_threshold', default=0.3, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--cuda', action='store_true', default=None,
                        help='use cuda.')
    return parser

args = parse_args().parse_args(args=[])

class datanih():
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        img, bbox = self.pull_item(index)
        return img, bbox  # 图像和金标准

    def __len__(self):
        return len(self.data[0]['train_set'])

    def pull_item(self, index):
        dataset = self.data[0]
        dataset = dataset['train_set']
        img = dataset[index]['img']
        img = torch.from_numpy(img)
        img = img[np.newaxis, :]
        bbox = dataset[index]['bbox']
        bbox = bbox.tolist()
        bbox.append(0)

        return img, bbox

    def pull_image(self, index):
        dataset = self.data[0]
        dataset = dataset['train_set']
        img = dataset[index]['img']
        img = torch.from_numpy(img)
        img = img[np.newaxis, :]
        bbox = dataset[index]['bbox']
        bbox = bbox.tolist()
        bbox.append(0)

        return img


def vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    """
    img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset = img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset_data_kind
    """
    if dataset == 'voc':
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                mess = '%s' % (class_names[int(cls_indx)])
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    elif dataset == 'coco-val' and class_indexs is not None:
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            xmin, ymin, xmax, ymax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    elif dataset == 'pnens' and class_indexs is not None:
        img = img.numpy()
        for i, box in enumerate(bboxes):
            #if i == 0:break
            cls_indx = cls_inds[i]
            xmin, ymin, zmin, xmax, ymax, zmax = box
            if scores[i] > thresh:
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[int(cls_indx)], 1)
                cv2.rectangle(img, (int(xmin), int(abs(ymin)-20)), (int(xmax), int(ymin)), class_colors[int(cls_indx)], -1)
                cls_id = class_indexs[int(cls_indx)]
                cls_name = class_names[cls_id]
                # mess = '%s: %.3f' % (cls_name, scores[i])
                mess = '%s' % (cls_name)
                cv2.putText(img, mess, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return img

def test(net, device, testset, transform, thresh, class_colors=None, class_names=None, class_indexs=None, dataset_data_kind='pnens'):
    num_images = len(testset)
    for index in range(num_images):
        # if index == 0: break
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img = testset.pull_image(index)
        _, h, w, d = img.shape

        # to tensor
        x = img
        x = x.unsqueeze(0).to(device)
        t0 = time.time()
        # forward
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")

        # scale each detection back up to the image
        # scale = np.array([[w, h, d, w, h, d]])
        # # map the boxes to origin image scale
        # bboxes *= scale
        #
        # img_processed = vis(img, bboxes, scores, cls_inds, thresh, class_colors, class_names, class_indexs, dataset_data_kind)
        # cv2.imshow('detection', img_processed)
        # cv2.waitKey(0)
        # print('Saving the' + str(index) + '-th image ...')
        # cv2.imwrite('test_images/' + args.dataset+ '3/' + str(index).zfill(6) +'.jpg', img)


if __name__ == '__main__':
      # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda", 2)
    else:
        device = torch.device("cpu")

    input_size = [128, 128, 64]

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(root=VOC_ROOT, image_sets=[('2007', 'test')], transform=None)

    elif args.dataset == 'coco-val':
        print('test on coco-val ...')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
            data_dir=coco_root,
            json_file='instances_val2017.json',
            name='val2017',
            img_size=input_size[0])
    elif args.dataset == 'pnens':
        num_classes = 1
        class_names = ('pnens')
        # dataset2 = load_from_pkl(r'/data/liyi219/pnens_3D_data/after_dealing/pre_order0_128_128_64_new.pkl')
        dataset2 = load_from_pkl('C:\\Users\\hazy\\Desktop\\fsdownload\\\pre_order0_128_128_64_new.pkl')
        # dataset = rechange(dataset2)
        dataset = datanih(dataset2)
    class_colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in
                    range(num_classes)]

    # load net
    from models.yolo_v3 import myYOLOv3

    anchor_size = anchor_size_3D_try
    net = myYOLOv3(device, input_size=input_size, num_classes=1, conf_thresh=args.conf_thresh,
                   nms_thresh=args.nms_thresh, anchor_size=anchor_size)

    # net.load_state_dict(torch.load(args.trained_model, map_location=device))
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=net, 
        device=device, 
        testset=dataset,
        transform=BaseTransform(input_size),
        thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=[0],
        dataset_data_kind=args.dataset
        )
"""
        net=net 
        device=device
        testset=dataset
        transform=BaseTransform(input_size)
        thresh=args.visual_threshold
        class_colors=class_colors
        class_names=class_names
        class_indexs=[0]
        dataset_data_kind=args.dataset
"""