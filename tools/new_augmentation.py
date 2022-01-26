import argparse
import glob
from pathlib import Path

try:
    import open3d
    from tools.visual_utils import open3d_vis_utils as V
    
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from tools.visual_utils import visualize_utils as V
    
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets import __all__ as dataset_dict
import pcdet.datasets.augmentor.data_augmentor as da
from pcdet.datasets import build_dataloader
from pcdet.utils.box_utils import boxes_to_corners_3d


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)
    
    return args, cfg


def main():
    args, cfg = parse_config()
    aug_dataset = dataset_dict["KittiDataset"](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=True
    )
    cfg_from_yaml_file('cfgs/kitti_models/pointpillar_my_noaugs.yaml', cfg)
    non_aug_dataset = dataset_dict["KittiDataset"](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=True
    )
    # 19 车多
    # 21 4个人
    # 25 2人多车
    # 393 多人多车
    scene_id = 393
    data_dict = aug_dataset[scene_id]
    non_aug_data_dict = non_aug_dataset[scene_id]
    print(data_dict['gt_boxes'][..., -1].astype(int))
    
    V.draw_augment_test_scenes(non_aug_points=non_aug_data_dict['points'],
                               non_aug_gt_boxes=non_aug_data_dict['gt_boxes'],
                               aug_points=data_dict['points'],
                               aug_gt_boxes=data_dict['gt_boxes'],
                               cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    
    if not OPEN3D_FLAG:
        mlab.show(stop=True)


if __name__ == '__main__':
    main()
