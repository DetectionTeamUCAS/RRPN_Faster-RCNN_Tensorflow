# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from libs.configs import cfgs
import numpy as np
import numpy.random as npr
from libs.box_utils import iou_rotate
from libs.box_utils import encode_and_decode
import tensorflow as tf


def anchor_target_layer(
        gt_boxes, anchors, overlaps, is_restrict_bg=False):
    """Same as the anchor target layer in original Fast/er RCNN """

    total_anchors = anchors.shape[0]
    gt_boxes = gt_boxes[:, :-1]  # remove class label

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((total_anchors,), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes

    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(total_anchors), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[
        gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    if not cfgs.TRAIN_RPN_CLOOBER_POSITIVES:
        labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= cfgs.RPN_IOU_POSITIVE_THRESHOLD] = 1

    if cfgs.TRAIN_RPN_CLOOBER_POSITIVES:
        labels[max_overlaps < cfgs.RPN_IOU_NEGATIVE_THRESHOLD] = 0

    num_fg = int(cfgs.RPN_MINIBATCH_SIZE * cfgs.RPN_POSITIVE_RATE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    num_bg = cfgs.RPN_MINIBATCH_SIZE - np.sum(labels == 1)
    if is_restrict_bg:
        num_bg = max(num_bg, num_fg * 1.5)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # labels = labels.reshape((1, height, width, A))
    rpn_labels = labels.reshape((-1, 1))

    # bbox_targets
    bbox_targets = bbox_targets.reshape((-1, 5))
    rpn_bbox_targets = bbox_targets

    return rpn_labels, rpn_bbox_targets


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # targets = bbox_transform(ex_rois, gt_rois[:, :5]).astype(
    #     np.float32, copy=False)
    targets = encode_and_decode.encode_boxes_rotate(unencode_boxes=gt_rois,
                                                    reference_boxes=ex_rois,
                                                    scale_factors=cfgs.ANCHOR_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=None)
    return targets
