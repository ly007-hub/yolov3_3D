import numpy as np
from data import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



CLASS_COLOR = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(len(VOC_CLASSES))]
# We use ignore thresh to decide which anchor box can be kept.
ignore_thresh = IGNORE_THRESH

class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
    def forward(self, probs, targets):
        # probs = torch.zeros([3,250])
        # targets = torch.zeros([3,250])
        # probs[:,175:] = 1
        # targets[:,125:] = 1
        num = targets.size(0)
        smooth = 1e-12

        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1))  / ((m1.sum(1) + m2.sum(1) )+ smooth)
        score = 1 - score.sum() / num
        return score

class BCELoss(nn.Module):
    def __init__(self,  weight=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, mask):
        # pred_conf, gt_conf, gt_obj
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        # print("pos and neg sum:", pos_id.sum(), neg_id.sum())
        # print("targets:", targets.max(), targets.min())

        # pos_loss = -pos_id * (targets * torch.log(inputs + 1e-14) + (1 - targets) * torch.log(1.0 - inputs + 1e-14))
        pos_loss = -pos_id * (mask * torch.log(inputs + 1e-14) + (1 - mask) * torch.log(1.0 - inputs + 1e-14))
        # pos_loss = -pos_id * (torch.log(inputs + 1e-14) + torch.log(1.0 - inputs + 1e-14))
        neg_loss = -neg_id * (mask * torch.log(inputs + 1e-14) + (1 - mask) * torch.log(1.0 - inputs + 1e-14))
        # neg_loss = -neg_id * (1 - mask) * torch.log(1.0 - inputs + 1e-14)
        # print("pos_loss:", pos_loss.max())
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1)) / int(neg_id.sum())
            # print("pos_loss and neg_loss after:", pos_loss, neg_loss)
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


class MSELoss(nn.Module):
    def __init__(self,  weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets, mask):
        # We ignore those whose tarhets == -1.0. 
        pos_id = (mask==1.0).float()
        neg_id = (mask==0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2
        if self.reduction == 'mean':
            pos_loss = torch.mean(torch.sum(pos_loss, 1))
            neg_loss = torch.mean(torch.sum(neg_loss, 1))
            return pos_loss, neg_loss
        else:
            return pos_loss, neg_loss


def generate_anchor(input_size, stride, anchor_scale, anchor_aspect):
    """
        The function is used to design anchor boxes by ourselves as long as you provide the scale and aspect of anchor boxes.
        Input:
            input_size : list -> the image resolution used in training stage and testing stage.
            stride : int -> the downSample of the CNN, such as 32, 64 and so on.
            anchor_scale : list -> it contains the area ratio of anchor boxes. For example, anchor_scale = [0.1, 0.5]
            anchor_aspect : list -> it contains the aspect ratios of anchor boxes for various anchor area.
                            For example, anchor_aspect = [[1.0, 2.0], [3.0, 1/3]]. And len(anchor_aspect) must 
                            be equal to len(anchor_scale).
        Output:
            total_anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    """
    assert len(anchor_scale) == len(anchor_aspect)
    h, w = input_size
    hs, ws = h // stride, w // stride
    S_fmap = hs * ws
    total_anchor_size = []
    for ab_scale, aspect_ratio in zip(anchor_scale, anchor_aspect):
        for a in aspect_ratio:
            S_ab = S_fmap * ab_scale
            ab_w = np.floor(np.sqrt(S_ab))
            ab_h =ab_w * a
            total_anchor_size.append([ab_w, ab_h])
    return total_anchor_size


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [xmin, ymin, xmax, ymax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]

    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U

    return IoU


def compute_iou3D(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, c_z_s, anchor_w, anchor_h, anchor_d] ->  [xmin, ymin, zmin, xmax, ymax, zmax]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 6])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 3] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 0] + anchor_boxes[:, 3] / 2  # xmax
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 1] - anchor_boxes[:, 4] / 2  # ymin
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 4] / 2  # ymax
    ab_x1y1_x2y2[:, 4] = anchor_boxes[:, 2] - anchor_boxes[:, 5] / 2  # zmin
    ab_x1y1_x2y2[:, 5] = anchor_boxes[:, 2] + anchor_boxes[:, 5] / 2  # zmax
    w_ab, h_ab, d_ab = anchor_boxes[:, 3], anchor_boxes[:, 4], anchor_boxes[:, 5]

    # gt_box :
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily.
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 6])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 3] / 2  # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 0] + gt_box_expand[:, 3] / 2  # xmax
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 1] - gt_box_expand[:, 4] / 2  # ymin
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 4] / 2  # ymax
    gb_x1y1_x2y2[:, 4] = gt_box_expand[:, 2] - gt_box_expand[:, 5] / 2  # zmin
    gb_x1y1_x2y2[:, 5] = gt_box_expand[:, 2] + gt_box_expand[:, 5] / 2  # zmax
    w_gt, h_gt, d_gt = gt_box_expand[:, 3], gt_box_expand[:, 4], gt_box_expand[:, 5]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt * d_gt
    S_ab = w_ab * h_ab * d_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2])
    I_d = np.minimum(gb_x1y1_x2y2[:, 5], ab_x1y1_x2y2[:, 5]) - np.maximum(gb_x1y1_x2y2[:, 4], ab_x1y1_x2y2[:, 4])
    S_I = I_h * I_w * I_d
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U

    return IoU

def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, 0, anchor_w, anchor_h, anchor_d],
                                   [0, 0, 0, anchor_w, anchor_h, anchor_d],
                                   ...
                                   [0, 0, 0, anchor_w, anchor_h, anchor_d]].
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 6])
    for index, size in enumerate(anchor_size):
        anchor_w, anchor_h, anchor_d = size
        anchor_boxes[index] = np.array([0, 0, 0, anchor_w, anchor_h, anchor_d])

    return anchor_boxes


def generate_txtytwth(gt_label, w, h, s, all_anchor_size):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # compute the center, width and height
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if box_w < 1. or box_h < 1.:
        # print('A dirty data !!!')
        return False

    # map the center, width and height to the feature map size
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s

    # the grid cell location
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # generate anchor boxes
    anchor_boxes = set_anchors(all_anchor_size)
    gt_box = np.array([[0, 0, box_ws, box_hs]])
    # compute the IoU
    iou = compute_iou(anchor_boxes, gt_box)
    # We consider those anchor boxes whose IoU is more than ignore thresh,
    iou_mask = (iou > ignore_thresh)

    result = []
    if iou_mask.sum() == 0:
        # We assign the anchor box with highest IoU score.
        index = np.argmax(iou)
        p_w, p_h = all_anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w + 1e-20)
        th = np.log(box_hs / p_h + 1e-20)
        weight = 2.0 - (box_w / w) * (box_h / h)

        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])

        return result

    else:
        # There are more than one anchor boxes whose IoU are higher than ignore thresh.
        # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other 
        # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
        # iou_ = iou * iou_mask

        # We get the index of the best IoU
        best_index = np.argmax(iou)
        for index, iou_m in enumerate(iou_mask):
            if iou_m:
                if index == best_index:
                    p_w, p_h = all_anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w + 1e-20)
                    th = np.log(box_hs / p_h + 1e-20)
                    weight = 2.0 - (box_w / w) * (box_h / h)

                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    # we ignore other anchor boxes even if their iou scores all higher than ignore thresh
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])

        return result


def gt_creator(input_size, stride, label_lists, anchor_size):
    """
    Input:
        input_size : list -> the size of image in the training stage.
        stride : int or list -> the downSample of the CNN, such as 32, 64 and so on.
        label_list : list -> [[[xmin, ymin, xmax, ymax, cls_ind], ... ], [[xmin, ymin, xmax, ymax, cls_ind], ... ]],  
                        and len(label_list) = batch_size;
                            len(label_list[i]) = the number of class instance in a image;
                            (xmin, ymin, xmax, ymax) : the coords of a bbox whose valus is between 0 and 1;
                            cls_ind : the corresponding class label.
    Output:
        gt_tensor : ndarray -> shape = [batch_size, anchor_number, 1+1+4, grid_cell number ]
    """
    assert len(input_size) > 0 and len(label_lists) > 0
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    w = input_size[1]
    h = input_size[0]

    # We  make gt labels by anchor-free method and anchor-based method.
    ws = w // stride
    hs = h // stride
    s = stride

    # We use anchor boxes to build training target.
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size)

    gt_tensor = np.zeros([batch_size, hs, ws, anchor_number, 1+1+4+1+4])

    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            results = generate_txtytwth(gt_label, w, h, s, all_anchor_size)
            if results:
                for result in results:
                    index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0.:
                        if grid_y < gt_tensor.shape[1] and grid_x < gt_tensor.shape[2]:
                            gt_tensor[batch_index, grid_y, grid_x, index, 0] = 1.0
                            gt_tensor[batch_index, grid_y, grid_x, index, 1] = gt_class
                            gt_tensor[batch_index, grid_y, grid_x, index, 2:6] = np.array([tx, ty, tw, th])
                            gt_tensor[batch_index, grid_y, grid_x, index, 6] = weight
                            gt_tensor[batch_index, grid_y, grid_x, index, 7:] = np.array([xmin, ymin, xmax, ymax])
                    else:
                        gt_tensor[batch_index, grid_y, grid_x, index, 0] = -1.0
                        gt_tensor[batch_index, grid_y, grid_x, index, 6] = -1.0

    gt_tensor = gt_tensor.reshape(batch_size, hs * ws * anchor_number, 1+1+4+1+4)

    return gt_tensor


def multi_gt_creator(input_size, strides, label_lists, anchor_size):
    """creator multi scales gt"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h, w = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, anchor_number, 1+1+4+1+4]))
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            xmin, ymin, xmax, ymax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2 * w
            c_y = (ymax + ymin) / 2 * h
            box_w = (xmax - xmin) * w
            box_h = (ymax - ymin) * h

            if box_w < 1. or box_h < 1.:
                # print('A dirty data !!!')
                continue

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])

            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other 
                # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask

                # We get the index of the best IoU
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # get the corresponding stride
                            s = strides[s_indx]
                            # get the corresponding anchor box
                            p_w, p_h = anchor_boxes[index, 2], anchor_boxes[index, 3]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            # compute gt labels
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            # 重点，告诉了为什么需要先验框的数值
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            weight = 2.0 - (box_w / w) * (box_h / h)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([tx, ty, tw, th])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 7:] = np.array([xmin, ymin, xmax, ymax])

                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = -1.0

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)

    return gt_tensor


def multi_gt_creator3D_for_show(input_size, strides, label_lists, anchor_size):
    # 将绝对坐标改为相对坐标 [xmin, xmax, ymin, ymax, zmin, zmax]
    """creator multi scales gt"""
    """
    input_size=train_size
    strides=model.stride
    label_lists=targets
    anchor_size=anchor_size
    """
    # prepare the all empty gt datas
    for i in range(len(label_lists)):
        label_lists[i] = [label_lists[i][0]*128, label_lists[i][1]*128, label_lists[i][2]*128, label_lists[i][3]*128, label_lists[i][4]*64, label_lists[i][5]*64, label_lists[i][6]*64]
    batch_size = len(label_lists)
    h, w, d = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h // s, w // s, d // s, anchor_number, 1 + 1 + 6 + 1 + 6]))
    for batch_index in range(batch_size):
        # batch_index = 0
        for gt_label in label_lists:
            # get a bbox coords
            # gt_label = label_lists[0]
            gt_class = int(gt_label[-1])
            xmin, xmax, ymin, ymax, zmin, zmax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2
            c_y = (ymax + ymin) / 2
            c_z = (zmax + zmin) / 2
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            box_d = (zmax - zmin)

            # if box_w < 1. or box_h < 1. or box_d < 1.:
            #     print('A dirty data !!!')
            #     continue

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, 0, box_w, box_h, box_d]])
            iou = compute_iou3D(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h, p_d = anchor_boxes[index, 3], anchor_boxes[index, 4], anchor_boxes[index, 5]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                c_z_s = c_z / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                grid_z = int(c_z_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tz = c_z_s - grid_z
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                td = np.log(box_d / p_d)
                weight = 2.0 - (box_w / w) * (box_h / h) * (box_d / d)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2] and grid_z < gt_tensor[s_indx].shape[3]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 2:8] = np.array([tx, ty, tz, tw, th, td])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 8] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 9:] = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other
                # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask

                # We get the index of the best IoU
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # We assign the anchor box with highest IoU score.
                            index = np.argmax(iou)
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # get the corresponding stride
                            s = strides[s_indx]
                            # get the corresponding anchor box
                            p_w, p_h, p_d = anchor_boxes[index, 3], anchor_boxes[index, 4], anchor_boxes[index, 5]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            c_z_s = c_z / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            grid_z = int(c_z_s)
                            # compute gt labels
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tz = c_z_s - grid_z
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            td = np.log(box_d / p_d)
                            weight = 2.0 - (box_w / w) * (box_h / h) * (box_d / d)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2] and grid_z < gt_tensor[s_indx].shape[3]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 2:8] = np.array([tx, ty, tz, tw, th, td])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 8] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 9:] = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            c_z_s = c_z / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            grid_z = int(c_z_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 8] = -1.0

    return gt_tensor

def multi_gt_creator3D_for_dataload(input_size, strides, label_lists, anchor_size):
    # 将绝对坐标改为相对坐标 [xmin, xmax, ymin, ymax, zmin, zmax]
    """creator multi scales gt"""
    """
    input_size=train_size
    strides=model.stride
    label_lists=targets
    anchor_size=anchor_size
    """
    # prepare the all empty gt datas
    for i in range(len(label_lists)):
        label_lists[i] = [label_lists[i][0]*128, label_lists[i][1]*128, label_lists[i][2]*128, label_lists[i][3]*128, label_lists[i][4]*64, label_lists[i][5]*64, label_lists[i][6]*64]
    batch_size = len(label_lists)
    h, w, d = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h // s, w // s, d // s, anchor_number, 1 + 1 + 6 + 1 + 6]))
    for batch_index in range(batch_size):
        # batch_index = 0
        for gt_label in label_lists:
            # get a bbox coords
            # gt_label = label_lists[0]
            gt_class = int(gt_label[-1])
            xmin, xmax, ymin, ymax, zmin, zmax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2
            c_y = (ymax + ymin) / 2
            c_z = (zmax + zmin) / 2
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            box_d = (zmax - zmin)

            # if box_w < 1. or box_h < 1. or box_d < 1.:
            #     print('A dirty data !!!')
            #     continue

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)
            gt_box = np.array([[0, 0, 0, box_w, box_h, box_d]])
            iou = compute_iou3D(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h, p_d = anchor_boxes[index, 3], anchor_boxes[index, 4], anchor_boxes[index, 5]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                c_z_s = c_z / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                grid_z = int(c_z_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tz = c_z_s - grid_z
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                td = np.log(box_d / p_d)
                weight = 2.0 - (box_w / w) * (box_h / h) * (box_d / d)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2] and grid_z < gt_tensor[s_indx].shape[3]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 2:8] = np.array([tx, ty, tz, tw, th, td])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 8] = weight
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 9:] = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other
                # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask

                # We get the index of the best IoU
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # We assign the anchor box with highest IoU score.
                            index = np.argmax(iou)
                            # s_indx=第几层featuremap， ab_ind=第s_index层的第几个框
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # get the corresponding stride
                            s = strides[s_indx]
                            # get the corresponding anchor box
                            p_w, p_h, p_d = anchor_boxes[index, 3], anchor_boxes[index, 4], anchor_boxes[index, 5]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            c_z_s = c_z / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            grid_z = int(c_z_s)
                            # compute gt labels
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tz = c_z_s - grid_z
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            td = np.log(box_d / p_d)
                            weight = 2.0 - (box_w / w) * (box_h / h) * (box_d / d)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2] and grid_z < gt_tensor[s_indx].shape[3]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 2:8] = np.array([tx, ty, tz, tw, th, td])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 8] = weight
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 9:] = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            c_z_s = c_z / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            grid_z = int(c_z_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, grid_z, ab_ind, 8] = -1.0

    gt_tensor = np.concatenate(gt_tensor, 1)  # [batch_index, featuremap_index， grid_y* grid_x * grid_z * ab_ind, 参数（15）]

    return gt_tensor  # [batch_index, featuremap_index, grid_y, grid_x, grid_z, ab_ind, 参数（15）]

def multi_gt_creator3D(input_size, strides, label_lists, anchor_size):
    """
    @param label_lists: [y1, y2, x1, x2, z1, z2, 0] 相对
    @param strides: [8, 16, 32]
    @return: [batch_index, featuremap_index， grid_y* grid_x * grid_z * ab_ind, 参数（15）]
             (15): [obj, class, tx, ty, tz, tw, th, td, weight, (xmin, ymin, zmin, xmax, ymax, zmax)(绝对)]
    """
    """
    input_size=train_size
    strides=model.stride
    label_lists=targets
    anchor_size=anchor_size
    """
    # 相对坐标to绝对坐标
    for i in range(len(label_lists)):
        label_lists[i] = [label_lists[i][0]*128, label_lists[i][1]*128, label_lists[i][2]*128, label_lists[i][3]*128, label_lists[i][4]*64, label_lists[i][5]*64, label_lists[i][6]]

    batch_size = len(label_lists)
    h, w, d = input_size
    num_scale = len(strides)
    gt_tensor = []

    # generate gt datas
    all_anchor_size = anchor_size
    anchor_number = len(all_anchor_size) // num_scale
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, w // s, h // s, d // s, anchor_number, 1 + 1 + 6 + 1 + 6]))
    for batch_index in range(batch_size):
        # batch_index = 0
        for gt_label in label_lists:
            # break
            gt_class = int(gt_label[-1])
            ymin, ymax, xmin, xmax, zmin, zmax = gt_label[:-1]
            # compute the center, width and height
            c_x = (xmax + xmin) / 2
            c_y = (ymax + ymin) / 2
            c_z = (zmax + zmin) / 2
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            box_d = (zmax - zmin)

            # if box_w < 1. or box_h < 1. or box_d < 1.:
            #     print('A dirty data !!!')
            #     continue

            # compute the IoU
            anchor_boxes = set_anchors(all_anchor_size)  # [0, 0, 0, anchor_w, anchor_h, anchor_d]
            gt_box = np.array([[0, 0, 0, box_w, box_h, box_d]])
            iou = compute_iou3D(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > ignore_thresh)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // anchor_number
                ab_ind = index - s_indx * anchor_number
                # get the corresponding stride
                s = strides[s_indx]
                # get the corresponding anchor box
                p_w, p_h, p_d = anchor_boxes[index, 3], anchor_boxes[index, 4], anchor_boxes[index, 5]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                c_z_s = c_z / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                grid_z = int(c_z_s)
                # compute gt labels
                tx = c_x_s - grid_x
                ty = c_y_s - grid_y
                tz = c_z_s - grid_z
                tw = np.log(box_w / p_w)
                th = np.log(box_h / p_h)
                td = np.log(box_d / p_d)
                weight = 2.0 - (box_w / w) * (box_h / h) * (box_d / d)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2] and grid_z < gt_tensor[s_indx].shape[3]:
                    gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 1] = gt_class
                    gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 2:8] = np.array([tx, ty, tz, tw, th, td])
                    gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 8] = weight
                    gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 9:] = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other
                # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask

                # We get the index of the best IoU
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # We assign the anchor box with highest IoU score.
                            index = np.argmax(iou)
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            # get the corresponding stride
                            s = strides[s_indx]
                            # get the corresponding anchor box
                            p_w, p_h, p_d = anchor_boxes[index, 3], anchor_boxes[index, 4], anchor_boxes[index, 5]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            c_z_s = c_z / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            grid_z = int(c_z_s)
                            # compute gt labels
                            tx = c_x_s - grid_x
                            ty = c_y_s - grid_y
                            tz = c_z_s - grid_z
                            tw = np.log(box_w / p_w)
                            th = np.log(box_h / p_h)
                            td = np.log(box_d / p_d)
                            weight = 2.0 - (box_w / w) * (box_h / h) * (box_d / d)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2] and grid_z < gt_tensor[s_indx].shape[3]:
                                gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 1] = gt_class
                                gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 2:8] = np.array([tx, ty, tz, tw, th, td])
                                gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 8] = weight
                                gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 9:] = np.array([xmin, ymin, zmin, xmax, ymax, zmax])

                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // anchor_number
                            ab_ind = index - s_indx * anchor_number
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            c_z_s = c_z / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            grid_z = int(c_z_s)
                            gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_x, grid_y, grid_z, ab_ind, 8] = -1.0
    # from [featuremap_index][batch_index, grid_x, grid_y, grid_z, ab_ind, 参数（15）]
    gt_tensor = [gt.reshape(batch_size, -1, 1 + 1 + 6 + 1 + 6) for gt in gt_tensor]  # [batch_index, grid_x * grid_y * grid_z * ab_ind, 参数（15）]
    gt_tensor = np.concatenate(gt_tensor, 1)  # [batch_index, featuremap_index， grid_x* grid_y * grid_z * ab_ind, 参数（15）]

    return gt_tensor

def iou_score(bboxes_a, bboxes_b):
    """
        bbox_1 : [B*N, 6] = [x1, y1, z1, x2, y2, z2]
        bbox_2 : [B*N, 6] = [x1, y1, z1, x2, y2, z2]
        bboxes_a, bboxes_b = x1y1z1x2y2z2_pred, x1y1z1x2y2z2_gt
    """
    if False:
        for true in bboxes_b:
            if not true[0] == 0:
                iou_compute_gt = true
                iou_compute_pred = bboxes_a[bboxes_b==true]
        if False:
            iou_compute_gt = iou_compute_gt * model.scale_torch
            iou_compute_pred = iou_compute_pred * model.scale_torch
            iou_compute_gt = iou_compute_gt.squeeze(0).squeeze(0)
            iou_compute_pred = iou_compute_pred.squeeze(0).squeeze(0)
        tl = torch.max(iou_compute_pred[:3], iou_compute_gt[:3])
        br = torch.min(iou_compute_pred[3:], iou_compute_gt[3:])
        area_a = torch.prod(iou_compute_pred[3:] - iou_compute_pred[:3], 0)
        area_b = torch.prod(iou_compute_gt[3:] - iou_compute_gt[:3], 0)

        en = (tl < br).type(tl.type()).prod(dim=0)
        area_i = torch.prod(br - tl, 0) * en  # * ((tl < br).all())
        iou = area_i / (area_a + area_b - area_i)


    tl = torch.max(bboxes_a[:, :3], bboxes_b[:, :3])
    br = torch.min(bboxes_a[:, 3:], bboxes_b[:, 3:])
    area_a = torch.prod(bboxes_a[:, 3:] - bboxes_a[:, :3], 1)
    area_b = torch.prod(bboxes_b[:, 3:] - bboxes_b[:, :3], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    return area_i / (area_a + area_b - area_i)

# def iou_score(anchor_boxes, gt_box):
#     # anchor_boxes: [x1,y1,z1,x2,y2,z2]
#     ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 6])
#     ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0]  # xmin
#     ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 3]  # xmax
#     ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 1]  # ymin
#     ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 4]  # ymax
#     ab_x1y1_x2y2[:, 4] = anchor_boxes[:, 2]  # zmin
#     ab_x1y1_x2y2[:, 5] = anchor_boxes[:, 5]  # zmax
#     w_ab = anchor_boxes[:, 3] - anchor_boxes[:, 0]
#     h_ab = anchor_boxes[:, 4] - anchor_boxes[:, 1]
#     d_ab = anchor_boxes[:, 5] - anchor_boxes[:, 2]
#
#     # gt_box :
#     # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily.
#     gt_box_expand = np.zeros([len(anchor_boxes), 6])
#     for i in range(len(anchor_boxes)):
#         gt_box_expand[i, :] = gt_box
#
#     gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 6])
#     gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0]  # xmin
#     gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 3]  # xmax
#     gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 1]  # ymin
#     gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 4]  # ymax
#     gb_x1y1_x2y2[:, 4] = gt_box_expand[:, 2]  # zmin
#     gb_x1y1_x2y2[:, 5] = gt_box_expand[:, 5]  # zmax
#     w_gt, h_gt, d_gt = gt_box_expand[:, 3], gt_box_expand[:, 4], gt_box_expand[:, 5]
#
#     # Then we compute IoU between anchor_box and gt_box
#     S_gt = w_gt * h_gt * d_gt
#     # print("w_ab:", w_ab)
#     # print("h_ab:", h_ab)
#     # print("d_ab:", d_ab)
#     S_ab = w_ab * h_ab * d_ab
#     I_w = np.minimum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
#     I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2])
#     I_d = np.minimum(gb_x1y1_x2y2[:, 5], ab_x1y1_x2y2[:, 5]) - np.maximum(gb_x1y1_x2y2[:, 4], ab_x1y1_x2y2[:, 4])
#     S_I = I_h * I_w * I_d
#     U = S_gt + S_ab - S_I + 1e-20
#     IoU = S_I / U
#
#     return IoU

def obj_map_get(obj, conf):
    obj_1 = obj[:, :6144].reshape(3, 16, 16, 8)
    obj_2 = obj[:, 6144:6912].reshape(3, 8, 8, 4)
    obj_3 = obj[:, 6912:].reshape(3, 4, 4, 2)

    conf_1 = conf[:, :6144].reshape(3, 16, 16, 8)
    conf_2 = conf[:, 6144:6912].reshape(3, 8, 8, 4)
    conf_3 = conf[:, 6912:].reshape(3, 4, 4, 2)

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

    # conf位置
    index_conf = torch.argmax(conf)
    conf_max = conf.max().detach().cpu().numpy()
    if index_conf < 6144:
        conf_1 = conf_1.detach().cpu().numpy()
        index_conf = np.where(conf_1 == conf_max)
    elif 6144 <= index_conf and index_conf < 6912:
        conf_2 = conf_2.detach().cpu().numpy()
        index_conf = np.where(conf_2 == conf_max)
    else:
        conf_3 = conf_3.detach().cpu().numpy()
        index_conf = np.where(conf_3 == conf_max)

    return index_obj, index_conf
    

def loss(pred_conf, pred_cls, pred_txtytwth, label, num_classes, obj_loss_f='bce'):
    """
    @param pred_conf: obj
    @param pred_cls:  class
    @param pred_txtytwth:
    @param label: [iou, obj, class, ty, tx, tz, th, tw, td, weight]
    @param num_classes:
    @param obj_loss_f:
    @return:
    """
    """
    pred_conf=conf_pred
    pred_cls=cls_pred
    pred_txtytwth=txtytwth_pred
    label=target
    num_classes=model.num_classes
    obj_loss_f='bce'
    """
    if obj_loss_f == 'bce':
        # In yolov3, we use bce as conf loss_f
        conf_loss_function = BCELoss(reduction='mean')
        conf_diceloss = SoftDiceLoss()
        obj = 1.0
        noobj = 1.0
    elif obj_loss_f == 'mse':
        # In yolov2, we use mse as conf loss_f.
        conf_loss_function = MSELoss(reduction='mean')
        obj = 5.0
        noobj = 1.0

    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    # txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    txty_loss_function = nn.MSELoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')


    pred_conf = torch.sigmoid(pred_conf[:, :, 0])
    pred_cls = torch.sigmoid(pred_cls)
    pred_cls = pred_cls.permute(0, 2, 1)
    txtytz_pred = pred_txtytwth[:, :, :3]
    twthtd_pred = pred_txtytwth[:, :, 3:]

    gt_conf = label[:, :, 0].float()
    # print('gt_conf', gt_conf)
    gt_obj = label[:, :, 1].float()
    index_obj, index_conf = obj_map_get(gt_obj, pred_conf)
    print(index_obj)
    print(index_conf)

    gt_cls = label[:, :, 2].long()
    gt_txtyztwthtd = label[:, :, 3:-1].float()
    gt_box_scale_weight = label[:, :, -1]
    gt_mask = (gt_box_scale_weight > 0.).float()

    # objectness loss
    # inputs, targets, mask

    dice_loss = conf_diceloss(pred_conf, (gt_obj >= 0.5).long())
    # print("pred_conf:", pred_conf.max(), pred_conf.min())
    # print("gt_conf:", gt_conf.max(), gt_conf.min())
    # print(gt_obj.max(), gt_obj.min())
    pos_loss, neg_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)
    print("pos, neg", pos_loss, neg_loss)

    conf_loss = obj * pos_loss + noobj * neg_loss

    # class loss
    # todo pred_cls 和 gt_cls 维度不一致, 但代码能运行并计算出数值
    cls_loss = torch.mean(torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask, 1))

    # box loss
    print(gt_txtyztwthtd[torch.where(gt_txtyztwthtd[:, :, 0]!=0)][:, :3])
    print(txtytz_pred[torch.where(gt_txtyztwthtd[:, :, 0]!=0)])
    print(gt_txtyztwthtd[torch.where(gt_txtyztwthtd[:, :, 0]!=0)][:, 3:])
    print(twthtd_pred[torch.where(gt_txtyztwthtd[:, :, 0]!=0)])

    txty_loss = torch.mean(torch.sum(torch.sum(txty_loss_function(txtytz_pred, gt_txtyztwthtd[:, :, :3]), 2) * gt_box_scale_weight * gt_mask, 1))
    twth_loss = torch.mean(torch.sum(torch.sum(twth_loss_function(twthtd_pred, gt_txtyztwthtd[:, :, 3:]), 2) * gt_box_scale_weight * gt_mask, 1))
    print(txty_loss, twth_loss)
    txtytwth_loss = txty_loss + twth_loss

    # total_loss = conf_loss + cls_loss + txtytwth_loss
    total_loss = conf_loss + txtytwth_loss
    # total_loss = conf_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss, dice_loss


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10],
                             [0.0, 0.0, 4, 4],
                             [0.0, 0.0, 8, 8],
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)