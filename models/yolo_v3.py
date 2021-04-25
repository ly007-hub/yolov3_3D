import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv2d, Conv3d
from backbone import *
import numpy as np
import tools


class myYOLOv3(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.50,
                 anchor_size=None, hr=False):
        super(myYOLOv3, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = [8, 16, 32]
        self.anchor_size = torch.tensor(anchor_size).view(3, len(anchor_size) // 3, 3)
        self.anchor_number = self.anchor_size.size(1)

        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[2], input_size[1], input_size[0], input_size[2]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()

        # backbone darknet-53 (optional: darknet-19)
        self.backbone = darknet53(pretrained=trainable, hr=hr)

        # s = 32
        self.conv_set_3 = nn.Sequential(
            Conv3d(1024, 512, 1, leakyReLU=True),
            Conv3d(512, 1024, 3, padding=1, leakyReLU=True),
            Conv3d(1024, 512, 1, leakyReLU=True),
            Conv3d(512, 1024, 3, padding=1, leakyReLU=True),
            Conv3d(1024, 512, 1, leakyReLU=True),
        )
        self.conv_1x1_3 = Conv3d(512, 256, 1, leakyReLU=True)
        self.extra_conv_3 = Conv3d(512, 1024, 3, padding=1, leakyReLU=True)
        self.pred_3 = nn.Conv3d(1024, self.anchor_number * (1 + 6 + self.num_classes), 1)

        # s = 16
        self.conv_set_2 = nn.Sequential(
            Conv3d(768, 256, 1, leakyReLU=True),
            Conv3d(256, 512, 3, padding=1, leakyReLU=True),
            Conv3d(512, 256, 1, leakyReLU=True),
            Conv3d(256, 512, 3, padding=1, leakyReLU=True),
            Conv3d(512, 256, 1, leakyReLU=True),
        )
        self.conv_1x1_2 = Conv3d(256, 128, 1, leakyReLU=True)
        self.extra_conv_2 = Conv3d(256, 512, 3, padding=1, leakyReLU=True)
        self.pred_2 = nn.Conv3d(512, self.anchor_number * (1 + 6 + self.num_classes), 1)

        # s = 8
        self.conv_set_1 = nn.Sequential(
            Conv3d(384, 128, 1, leakyReLU=True),
            Conv3d(128, 256, 3, padding=1, leakyReLU=True),
            Conv3d(256, 128, 1, leakyReLU=True),
            Conv3d(128, 256, 3, padding=1, leakyReLU=True),
            Conv3d(256, 128, 1, leakyReLU=True),
        )
        self.extra_conv_1 = Conv3d(128, 256, 3, padding=1, leakyReLU=True)
        self.pred_1 = nn.Conv3d(256, self.anchor_number * (1 + 6 + self.num_classes), 1)

    def create_grid(self, input_size):
        total_grid_xy = []
        total_stride = []
        total_anchor_wh = []
        w, h, d = input_size[1], input_size[0], input_size[2]
        for ind, s in enumerate(self.stride):
            # generate grid cells
            ws, hs, ds = w // s, h // s, d // s
            grid_y, grid_x, grid_z = torch.meshgrid([torch.arange(hs), torch.arange(ws), torch.arange(ds)])
            grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()
            grid_xyz = grid_xyz.view(1, hs * ws * ds, 3, 3)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs * ws * ds, self.anchor_number, 3]) * s

            # generate anchor_wh tensor
            anchor_wh = self.anchor_size[ind].repeat(hs * ws * ds, 1, 1)
            # anchor_wh = yolo_net.anchor_size[ind].repeat(hs * ws * ds, 1, 1)

            total_grid_xy.append(grid_xyz)
            total_stride.append(stride_tensor)
            total_anchor_wh.append(anchor_wh)

        """
        for ind, s in enumerate(yolo_net.stride):
            # generate grid cells
            ws, hs, ds = w // s, h // s, d // s
            grid_y, grid_x, grid_z = torch.meshgrid([torch.arange(hs), torch.arange(ws), torch.arange(ds)])
            grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()
            grid_xyz = grid_xyz.view(1, hs * ws * ds, 1, 3)

            # generate stride tensor
            stride_tensor = torch.ones([1, hs * ws * ds, yolo_net.anchor_number, 3]) * s

            # generate anchor_wh tensor
            anchor_wh = yolo_net.anchor_size[ind].repeat(hs * ws * ds, 1, 1)
            # anchor_wh = yolo_net.anchor_size[ind].repeat(hs * ws * ds, 1, 1)

            total_grid_xy.append(grid_xyz)
            total_stride.append(stride_tensor)
            total_anchor_wh.append(anchor_wh)
        """
        total_grid_xy = torch.cat(total_grid_xy, dim=1).to(self.device)
        total_stride = torch.cat(total_stride, dim=1).to(self.device)
        total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(self.device).unsqueeze(0)
        """
        total_grid_xy = torch.cat(total_grid_xy, dim=1).to(yolo_net.device)
        total_stride = torch.cat(total_stride, dim=1).to(yolo_net.device)
        total_anchor_wh = torch.cat(total_anchor_wh, dim=0).to(yolo_net.device).unsqueeze(0)
        """

        return total_grid_xy, total_stride, total_anchor_wh

    def set_grid(self, input_size):
        self.grid_cell, self.stride_tensor, self.all_anchors_wh = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [x, y, w, h]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HWD, ab_n, _ = txtytwth_pred.size()
        c_xyz_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :3]) + self.grid_cell) * self.stride_tensor
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        b_whd_pred = torch.exp(txtytwth_pred[:, :, :, 3:]) * self.all_anchors_wh
        # [B, H*W, anchor_n, 4] -> [B, H*W*anchor_n, 4]
        xyzwhd_pred = torch.cat([c_xyz_pred, b_whd_pred], -1).view(B, HWD * ab_n, 4)
        
        """
        B, HWD, ab_n, _ = txtytwth_pred.size()
        c_xyz_pred = (torch.sigmoid(txtytwth_pred[:, :, :, :3]) + yolo_net.grid_cell) * yolo_net.stride_tensor
        b_whd_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * yolo_net.all_anchors_wh
        xywh_pred = torch.cat([c_xy_pred, b_wh_pred], -1).view(B, HWD * ab_n, 4)
        """

        return xywh_pred

    def decode_boxes(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W, anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [B, H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)  # the size of bbox
        order = scores.argsort()[::-1]  # sort bounding boxes by decreasing order

        keep = []  # store the final bounding boxes
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW*anchor_n, 4), bsize = 1
        prob_pred: (HxW*anchor_n, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds



    def forward(self, x, target=None):
        # backbone
        fmp_1, fmp_2, fmp_3 = self.backbone(x)
        # fmp_1, fmp_2, fmp_3 = model.backbone(x)

        # detection head
        # multi scale feature map fusion
        fmp_3 = self.conv_set_3(fmp_3)
        fmp_3_up = F.interpolate(self.conv_1x1_3(fmp_3), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(self.conv_1x1_2(fmp_2), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        """
        import torch.nn.functional as F
        fmp_3 = model.conv_set_3(fmp_3)
        fmp_3_up = F.interpolate(model.conv_1x1_3(fmp_3), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = model.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(model.conv_1x1_2(fmp_2), scale_factor=2.0, mode='bilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = model.conv_set_1(fmp_1)
        """

        # head
        # s = 32
        fmp_3 = self.extra_conv_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)

        # s = 16
        fmp_2 = self.extra_conv_2(fmp_2)
        pred_2 = self.pred_2(fmp_2)

        # s = 8
        fmp_1 = self.extra_conv_1(fmp_1)
        pred_1 = self.pred_1(fmp_1)

        """
         fmp_3 = model.extra_conv_3(fmp_3)
        pred_3 = model.pred_3(fmp_3)

        # s = 16
        fmp_2 = model.extra_conv_2(fmp_2)
        pred_2 = model.pred_2(fmp_2)

        # s = 8
        fmp_1 = model.extra_conv_1(fmp_1)
        pred_1 = model.pred_1(fmp_1)
        """

        preds = [pred_1, pred_2, pred_3]
        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        B = HW = 0
        for pred in preds:
            B_, abC_, H_, W_ = pred.size()

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B_, H_ * W_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred   
            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * self.anchor_number].contiguous().view(B_, H_ * W_ * self.anchor_number, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :,
                       1 * self.anchor_number: (1 + self.num_classes) * self.anchor_number].contiguous().view(B_,
                                                                                                              H_ * W_ * self.anchor_number,
                                                                                                              self.num_classes)
            # [B, H*W*anchor_n, 4]
            txtytwth_pred = pred[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            B = B_
            HW += H_ * W_

        """
        for pred in preds:
            B_, abC_, H_, W_ = pred.size()

            # [B, anchor_n * C, H, W] -> [B, H, W, anchor_n * C] -> [B, H*W, anchor_n*C]
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B_, H_*W_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred   
            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * model.anchor_number].contiguous().view(B_, H_*W_*model.anchor_number, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :, 1 * model.anchor_number : (1 + model.num_classes) * model.anchor_number].contiguous().view(B_, H_*W_*model.anchor_number, model.num_classes)
            # [B, H*W*anchor_n, 4]
            txtytwth_pred = pred[:, :, (1 + model.num_classes) * model.anchor_number:].contiguous()

            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            B = B_
            HW += H_*W_
        """

        conf_pred = torch.cat(total_conf_pred, 1)
        cls_pred = torch.cat(total_cls_pred, 1)
        txtytwth_pred = torch.cat(total_txtytwth_pred, 1)

        # test
        if not self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1                
                all_obj = torch.sigmoid(conf_pred)[0]  # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], dim=1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                # print(len(all_boxes))
                return bboxes, scores, cls_inds

        else:
            txtytwth_pred = txtytwth_pred.view(B, HW, self.anchor_number, 4)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

            """
            txtytwth_pred = txtytwth_pred.view(B, HW, model.anchor_number, 4)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (model.decode_boxes(txtytwth_pred) / model.scale_torch).view(-1, 4)
            """

            txtytwth_pred = txtytwth_pred.view(B, -1, 4)

            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

            # compute iou
            iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
            # print(iou.min(), iou.max())

            # we set iou between pred bbox and gt bbox as conf label. 
            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            target = torch.cat([iou, target[:, :, :7]], dim=2)

            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=self.num_classes,
                                                                        obj_loss_f='mse')

            """
            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=model.num_classes,
                                                                        obj_loss_f='mse')
            """

            return conf_loss, cls_loss, txtytwth_loss, total_loss

    def test(self, x, target=None):
        fmp_1, fmp_2, fmp_3 = self.backbone(x)

        fmp_3 = self.conv_set_3(fmp_3)
        temp_3 = self.conv_1x1_3(fmp_3)
        # 对于体积volumetric输入，则期待着5D张量的输入，即minibatch x channels x depth x height x width
        fmp_3_up = F.interpolate(temp_3, scale_factor=2.0, mode='trilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = self.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(self.conv_1x1_2(fmp_2), scale_factor=2.0, mode='trilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = self.conv_set_1(fmp_1)

        fmp_3 = self.extra_conv_3(fmp_3)
        pred_3 = self.pred_3(fmp_3)

        # s = 16
        fmp_2 = self.extra_conv_2(fmp_2)
        pred_2 = self.pred_2(fmp_2)

        # s = 8
        fmp_1 = self.extra_conv_1(fmp_1)
        pred_1 = self.pred_1(fmp_1)

        """
        fmp_1, fmp_2, fmp_3 = yolo_net.backbone(input_3D)

        fmp_3 = yolo_net.conv_set_3(fmp_3)
        temp_3 = yolo_net.conv_1x1_3(fmp_3)
        # 对于体积volumetric输入，则期待着5D张量的输入，即minibatch x channels x depth x height x width
        fmp_3_up = F.interpolate(temp_3, scale_factor=2.0, mode='trilinear', align_corners=True)

        fmp_2 = torch.cat([fmp_2, fmp_3_up], 1)
        fmp_2 = yolo_net.conv_set_2(fmp_2)
        fmp_2_up = F.interpolate(yolo_net.conv_1x1_2(fmp_2), scale_factor=2.0, mode='trilinear', align_corners=True)

        fmp_1 = torch.cat([fmp_1, fmp_2_up], 1)
        fmp_1 = yolo_net.conv_set_1(fmp_1)

        fmp_3 = yolo_net.extra_conv_3(fmp_3)
        pred_3 = yolo_net.pred_3(fmp_3)

        # s = 16
        fmp_2 = yolo_net.extra_conv_2(fmp_2)
        pred_2 = yolo_net.pred_2(fmp_2)

        # s = 8
        fmp_1 = yolo_net.extra_conv_1(fmp_1)
        pred_1 = yolo_net.pred_1(fmp_1)
        """

        preds = [pred_1, pred_2, pred_3]
        total_conf_pred = []
        total_cls_pred = []
        total_txtytwth_pred = []
        B = HWD = 0
        for pred in preds:
            B_, abC_, H_, W_, D_ = pred.size()

            # [B, anchor_n * C, H, W, D] -> [B, H, W, D, anchor_n * C] -> [B, H*W*D, anchor_n*C]
            pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(B_, H_ * W_ * D_, abC_)

            # Divide prediction to obj_pred, xywh_pred and cls_pred
            # [B, H*W*anchor_n, 1]
            conf_pred = pred[:, :, :1 * self.anchor_number].contiguous().view(B_, H_ * W_ * D_ * self.anchor_number, 1)
            # [B, H*W*anchor_n, num_cls]
            cls_pred = pred[:, :,
                       1 * self.anchor_number: (1 + self.num_classes) * self.anchor_number].contiguous().view(B_,
                                                                                                              H_ * W_ * D_ * self.anchor_number,
                                                                                                              self.num_classes)
            # [B, H*W*anchor_n, 4]
            txtytwth_pred = pred[:, :, (1 + self.num_classes) * self.anchor_number:].contiguous()
            total_conf_pred.append(conf_pred)
            total_cls_pred.append(cls_pred)
            total_txtytwth_pred.append(txtytwth_pred)
            
            """
            for pred in preds:
                B_, abC_, H_, W_, D_ = pred.size()
    
                # [B, anchor_n * C, H, W, D] -> [B, H, W, D, anchor_n * C] -> [B, H*W*D, anchor_n*C]
                pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(B_, H_ * W_ * D_, abC_)
    
                # Divide prediction to obj_pred, xywh_pred and cls_pred
                # [B, H*W*anchor_n, 1]
                conf_pred = pred[:, :, :1 * yolo_net.anchor_number].contiguous().view(B_, H_ * W_ * D_ * yolo_net.anchor_number, 1)
                # [B, H*W*anchor_n, num_cls]
                cls_pred = pred[:, :,
                           1 * yolo_net.anchor_number: (1 + yolo_net.num_classes) * yolo_net.anchor_number].contiguous().view(B_,
                                                                                                                  H_ * W_ * D_ * yolo_net.anchor_number,
                                                                                                                  yolo_net.num_classes)
                # [B, H*W*anchor_n, 4]
                txtytwth_pred = pred[:, :, (1 + yolo_net.num_classes) * yolo_net.anchor_number:].contiguous()
                total_conf_pred.append(conf_pred)
                total_cls_pred.append(cls_pred)
                total_txtytwth_pred.append(txtytwth_pred)
                B = B_
                HWD += H_ * W_ * D_
            """

            B = B_
            HWD += H_ * W_ * D_

        conf_pred = torch.cat(total_conf_pred, 1)
        cls_pred = torch.cat(total_cls_pred, 1)
        txtytwth_pred = torch.cat(total_txtytwth_pred, 1)

        if not self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, HWD, self.anchor_number, 4)
            with torch.no_grad():
                # batch size = 1
                all_obj = torch.sigmoid(conf_pred)[0]  # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
                all_class = (torch.softmax(cls_pred[0, :, :], dim=1) * all_obj)
                # separate box pred and class conf
                all_obj = all_obj.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()

                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                # print(len(all_boxes))
                return bboxes, scores, cls_inds

        else:
            txtytwth_pred = txtytwth_pred.view(B, HWD, self.anchor_number, 6)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad():
                x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.scale_torch).view(-1, 4)

            """
            txtytwth_pred = txtytwth_pred.view(B, HWD, yolo_net.anchor_number, 6)
            # decode bbox, and remember to cancel its grad since we set iou as the label of objectness.
            with torch.no_grad(): problem
                x1y1x2y2_pred = (yolo_net.decode_boxes(txtytwth_pred) / yolo_net.scale_torch).view(-1, 5)
            """

            txtytwth_pred = txtytwth_pred.view(B, -1, 5)

            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)
            # x1y1x2y2_gt = torch.ones([1, 1752, 5])

            # compute iou
            iou = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)
            # print(iou.min(), iou.max())

            # we set iou between pred bbox and gt bbox as conf label.
            # [obj, cls, txtytwth, scale_weight, x1y1x2y2] -> [conf, obj, cls, txtytwth, scale_weight]
            target = torch.cat([iou, target[:, :, :7]], dim=2)

            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=self.num_classes,
                                                                        obj_loss_f='mse')

            """
            conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                        pred_txtytwth=txtytwth_pred,
                                                                        label=target,
                                                                        num_classes=model.num_classes,
                                                                        obj_loss_f='mse')
            """

            return conf_loss, cls_loss, txtytwth_loss, total_loss


if __name__ == '__main__':
    anchor_size = [[32.64, 47.68, 40.16], [50.24, 108.16, 79.2], [126.72, 96.32, 111.52],
                   [78.4, 201.92, 140.16], [178.24, 178.56, 178.4], [129.6, 294.72, 212.16],
                   [331.84, 194.56, 262.5], [227.84, 325.76, 276], [365.44, 358.72, 361.5]]

    input_3D = torch.ones([1, 3, 64, 64, 64])
    yolo_net = myYOLOv3(torch.device("cpu"), input_size=[416, 416, 416], num_classes=20, trainable=True, anchor_size=anchor_size, hr=False)
    output_3D = yolo_net.test(input_3D)
