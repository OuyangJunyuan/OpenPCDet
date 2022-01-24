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
from scipy.spatial.transform import Rotation
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset


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


def point_cloud_random_make_slope(points, rotate_point=None, rotate_angle=None):
    def random(n=1):
        return (np.random.random(n) - 0.5) * 2
    
    if rotate_point is None:
        mean, var = np.array([25, 0]), np.array([10, np.pi / 10000])
        polar_pos = mean + random(2) * var
        rotate_point = np.array([polar_pos[0] * np.cos(polar_pos[1]), polar_pos[0] * np.sin(polar_pos[1]), 0])
    
    x0, y0 = rotate_point[0], rotate_point[1]
    if rotate_angle is None:
        mean, var = np.pi / 8, np.pi / 12
        k0 = y0 / x0
        k1 = -1 / (k0 + 1e-6)
        v = np.array([x0 - 0, y0 - (-x0 * k1 + y0), 0])
        v /= np.linalg.norm(v)
        v *= mean + random() * var
        direction = np.sign(np.cross(rotate_point, v)[2])
        v *= -1 if direction > 0 else 1
        rotate_angle = v
    
    print(rotate_point, rotate_angle)
    k = rotate_angle[1] / (rotate_angle[0] + 1e-6)
    sign = np.sign(k * (0 - x0) + y0 - 0)
    in_plane_mask = np.sign(k * (points[:, 0] - x0) + y0 - points[:, 1]) == sign
    slope_points = points[np.logical_not(in_plane_mask)]
    slope_points[:, 0:3] -= rotate_point
    rot = Rotation.from_rotvec(rotate_angle).as_matrix()
    slope_points[:, 0:3] = (slope_points[:, 0:3].dot(rot.T))
    slope_points[:, 0:3] += rotate_point
    points[np.logical_not(in_plane_mask)] = slope_points
    return rotate_point, rotate_angle, in_plane_mask, points


def main():
    args, cfg = parse_config()
    aug_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=None,
        training=True
    )
    scene = 1000
    file_name = '%06d' % scene
    print(file_name)
    points = aug_dataset.get_lidar(file_name)  # 11
    r_p, r_a, mask, sloped_points = point_cloud_random_make_slope(points)
    # point_faraway_indices = points[:, 0] > 15
    # points_faraway = points[point_faraway_indices]
    # points_faraway -= np.array([15, 0, 0, 0])
    # rot = Rotation.from_rotvec(np.pi / 22 * np.array([0, 1, 0])).as_matrix()
    # points_faraway[:, 0:3] = (points_faraway[:, 0:3].dot(rot.T))
    # points_faraway += np.array([15, 0, 0, 0])
    # points = np.concatenate([points[np.logical_not(point_faraway_indices)], points_faraway])
    #
    with open(file_name + ".bin", 'w') as f:
        sloped_points.tofile(f)
    
    V.draw_scenes(points=sloped_points)
    
    # V.draw_augment_test_scenes(non_aug_points=non_aug_data_dict['points'],
    # 						   non_aug_gt_boxes=non_aug_data_dict['gt_boxes'],
    # 						   aug_points=data_dict['points'],
    # 						   aug_gt_boxes=data_dict['gt_boxes'])
    
    if not OPEN3D_FLAG:
        mlab.show(stop=True)


if __name__ == '__main__':
    main()
