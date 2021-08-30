from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np
from medical_data_load.utils import *

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import *
import tools
# try:
#     from mayavi import mlab
#     from scipy.ndimage import zoom
#     print('mayavi already imported')
# except:
#     pass

from utils.augmentations import SSDAugmentation
# from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.my_Evaluator import myEvaluator
from medical_data_load.dataset import get_dataset_NIH_pancreas, load_from_pkl
from medical_data_load.dataload import datanih
from medical_data_load.config_dataset import config as config_0
from medical_data_load.utils import *


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# dataset2 = get_dataset_NIH_pancreas(preload_cache=True, order=0)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')

    parser.add_argument('-v', '--version', default='yolo_v3',
                        help='yolo_v2, yolo_v3, yolo_v3_spp, slim_yolo_v2, tiny_yolo_v3')
    parser.add_argument('-d', '--dataset', default='pnens',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='/data/data4T/ly/data', type=str,
                        help='Gamma update for SGD')

    return parser


args = parse_args().parse_args(args=[])

path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
os.makedirs(path_to_save, exist_ok=True)

# use hi-res backbone
if args.high_resolution:
    print('use hi-res backbone')
    hr = True
else:
    hr = False

# cuda
if args.cuda:
    print('use cuda')
    cudnn.benchmark = True
    device = torch.device("cuda", 2)
else:
    device = torch.device("cpu")

# multi-scale
if args.multi_scale:
    print('use the multi-scale trick ...')
    train_size = config_0['NIH_pancreas_data_aimshape']
    val_size = config_0['NIH_pancreas_data_aimshape']
else:
    train_size = config_0['NIH_pancreas_data_aimshape']
    val_size = config_0['NIH_pancreas_data_aimshape']

cfg = train_cfg
# dataset and evaluator
print("Setting Arguments.. : ", args)
print("----------------------------------------------------------")
print('Loading the dataset...')
if args.dataset == 'pnens':
    num_classes = 1

    # dataset2 = load_from_pkl(r'/data/liyi219/pnens_3D_data/after_dealing/pre_order0_128_128_64_new.pkl')
    dataset2 = load_from_pkl(r'E:\data\pre_order0_128_128_64_new.pkl')
    # dataset = rechange(dataset2)
    dataset = datanih(dataset2)
    evaluator = myEvaluator(
        dataset=dataset,
        data_root="/data/data4T/ly/data/pnens_3D",
        img_size=val_size,
        device=device,
        transform=BaseTransform(val_size),
        labelmap=('pnens')
    )

else:
    print('unknow dataset !! Only support voc and coco !!')
    exit(0)

print('Training model on: yolov3_3D')
print('The dataset size:', len(dataset))
print("----------------------------------------------------------")

# dataloader
dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=detection_collate,
                num_workers=args.num_workers,
                pin_memory=True
                )

if args.version == 'yolo_v3':
    from models.yolo_v3 import myYOLOv3
    # anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
    anchor_size = anchor_size_3D_try

    yolo_net = myYOLOv3(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
    print('Let us train yolo_v3 on the %s dataset ......' % (args.dataset))

else:
    print('Unknown version !!!')
    exit()

model = yolo_net
model.to(device).train()

# use tfboard
if args.tfboard:
    print('use tensorboard')
    from torch.utils.tensorboard import SummaryWriter
    c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    log_path = os.path.join('log/coco/', args.version, c_time)
    os.makedirs(log_path, exist_ok=True)

    writer = SummaryWriter(log_path)

# keep training
if args.resume is not None:
    print('keep training model: %s' % (args.resume))
    model.load_state_dict(torch.load(args.resume, map_location=device))

# optimizer setup
base_lr = args.lr
tmp_lr = base_lr
optimizer = optim.SGD(model.parameters(),
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay
                        )

max_epoch = cfg['max_epoch']
epoch_size = len(dataset) // args.batch_size

# start training loop
t0 = time.time()

for epoch in range(args.start_epoch, max_epoch):
    break
    # if epoch == 1: break
    # use cos lr
    if args.cos and epoch > 20 and epoch <= max_epoch - 20:
        # use cos lr
        tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
        set_lr(optimizer, tmp_lr)

    elif args.cos and epoch > max_epoch - 20:
        tmp_lr = 0.00001
        set_lr(optimizer, tmp_lr)

    # use step lr
    else:
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)


for iter_i, (images, targets) in enumerate(dataloader):
    break
    # WarmUp strategy for learning rate
    # 训练策略，至今没有弄明白
    # if iter_i == 0:
    #     print(images.shape)
if not args.no_warm_up:
    if epoch < args.wp_epoch:
        tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
        # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
        set_lr(optimizer, tmp_lr)

    elif epoch == args.wp_epoch and iter_i == 0:
        tmp_lr = base_lr
        set_lr(optimizer, tmp_lr)

# to device
images = images.to(device)

# multi-scale trick
if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
    # randomly choose a new size
    size = random.randint(10, 19) * 32
    train_size = [size, size]
    model.set_grid(train_size)
if args.multi_scale:
    # interpolate
    # 上采样
    images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

# make labels
# print(targets)
targets = [label.tolist() for label in targets]
# targets = tools.multi_gt_creator3D(input_size=train_size,
#                                  strides=model.stride,
#                                  label_lists=targets,
#                                  anchor_size=anchor_size
#                                  )

if False:
    img_for_target = tools.multi_gt_creator3D_for_show(input_size=train_size,
                                                       strides=model.stride,
                                                       label_lists=targets,
                                                       anchor_size=anchor_size
                                                       )
    img_for_target_1 = img_for_target[0][0, :, :, :, 0, 0]
    img_for_target_2 = img_for_target[1][0, :, :, :, 0, 0]
    img_for_target_3 = img_for_target[2][0, :, :, :, 0, 0]
    show3D(img_for_target_1)
    show3D(img_for_target_2)
    show3D(img_for_target_3)
    boundry = [0, img_for_target_1.shape[0], 0, img_for_target_1.shape[1], 0, img_for_target_1.shape[2]]
    show3Dbbox(boundry)


"""
        targets = torch.tensor(targets).float().to(device)

        # forward and loss
        conf_loss, cls_loss, txtytwth_loss, total_loss, dice_loss = model(images, target=targets)

        # backprop
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # display
        if iter_i % 10 == 0:
            if args.tfboard:
                # viz loss
                writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)

            t1 = time.time()

            print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                '[Loss: obj %.2f ||dice_loss %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                    % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                        conf_loss.item(), dice_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0], t1-t0),
                    flush=True)

            t0 = time.time()

    # evaluation
    if (epoch + 1) % args.eval_epoch == 0:
        model.trainable = False
        model.set_grid(val_size)
        model.eval()

        # evaluate
        # evaluator.evaluate(model)

        # convert to training mode.
        model.trainable = True
        model.set_grid(train_size)
        model.train()

    # save model
    if (epoch + 1) % 5 == 0:
        print('Saving state, saving in {} epoch:'.format(os.path.join(path_to_save, args.version), epoch + 1, ))
        torch.save(model.state_dict(), os.path.join(path_to_save,
                    args.version + '_' + repr(epoch + 1) + '.pth')
                    )
                    
                    """




