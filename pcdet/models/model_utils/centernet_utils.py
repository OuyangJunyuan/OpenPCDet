# This file is modified from https://github.com/tianweiy/CenterPoint

import torch
import torch.nn.functional as F
import numpy as np
import numba


def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:
    Returns:
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1, valid_mask=None):
    # 直径
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]
    # center在heatmap上能扩充的四边，防止超出边界。
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    # 得到center周边的heatmap和gaussian
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        if valid_mask is not None:
            cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
            masked_gaussian = masked_gaussian * cur_valid_mask.float()
        
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


@numba.jit(nopython=True)
def circle_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2
            
            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1
    return keep


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]
    
    keep = torch.from_numpy(keep).long().to(boxes.device)
    
    return keep


# 从feat(batch,num_class*K,feat_channel])中用ind(batch,K[0~num_class*K-1]))
# 得到对应位置上的值
# 其中feat是(batch,每个类别的K个最大,特征), 而ind是(batch，在num_clss*K所有类别的前K个)
def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)  #
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # feat = gather(a,dim,ind)
    # feat[d0,...,d_dim=i,...,d_n] = a[d0,...,d_dim=ind[d0,...,d_dim=i,...,d_n],...,d_n]
    # 也就是说，其余维度索引相同然后，feat的第dim维度上的第i个=a的第dim维度上的第(ind的第dim维度上的第i个)个
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    batch, num_class, height, width = scores.size()
    # 展开2,3维度->(batch,num_class,height*width)
    # topk((batch,num_class,height*width)) -> val/ind(batch,num_class,K)
    topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)
    
    # (batch,num_class,K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()
    
    # topk((batch,num_class*K)) -> score/topk(batch,K:0~num_class*K-1)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # topk_classes(batch,K:0~num_class)
    topk_classes = (topk_ind // K).int()
    # _gather_feature((batch,num_class*K,1[0~height*width]), (batch,K[0~num_class*K-1]))
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    
    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


# heatmap/../(batch,feat_channel,w,h)
def decode_bbox_from_heatmap(heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100,
                             circle_nms=False, score_thresh=None, post_center_limit_range=None):
    batch_size, num_class, _, _ = heatmap.size()
    
    if circle_nms:
        # TODO: not checked yet
        assert False, 'not checked yet'
        heatmap = _nms(heatmap)
    # scores/inds/class_ids/ys/xs(batch, K)
    scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)
    # center(batch,feat_channel,w,h) -> center(batch,K,feat_channel)
    center = _transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
    rot_sin = _transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
    rot_cos = _transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
    center_z = _transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)
    # (batch, K, 1)
    angle = torch.atan2(rot_sin, rot_cos)
    xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]  # x_fine = x_coarse + cemter_offset
    ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]
    
    xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
    ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]
    
    box_part_list = [xs, ys, center_z, dim, angle]
    if vel is not None:
        vel = _transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
        box_part_list.append(vel)
    # (batch, K, 7or8)
    final_box_preds = torch.cat((box_part_list), dim=-1)
    # (batch, K)
    final_scores = scores.view(batch_size, K)
    final_class_ids = class_ids.view(batch_size, K)
    
    assert post_center_limit_range is not None
    mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
    mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)
    
    if score_thresh is not None:
        mask &= (final_scores > score_thresh)
    
    ret_pred_dicts = []
    for k in range(batch_size):
        cur_mask = mask[k]
        cur_boxes = final_box_preds[k, cur_mask]
        cur_scores = final_scores[k, cur_mask]
        cur_labels = final_class_ids[k, cur_mask]
        
        if circle_nms:
            assert False, 'not checked yet'
            centers = cur_boxes[:, [0, 1]]
            boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
            keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)
            
            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_labels = cur_labels[keep]
        
        ret_pred_dicts.append({
            'pred_boxes': cur_boxes,
            'pred_scores': cur_scores,
            'pred_labels': cur_labels
        })
    # ret_pred_dicts(list[len=batch,type=dict])
    return ret_pred_dicts
