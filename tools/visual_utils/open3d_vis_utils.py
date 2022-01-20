"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import os

import numpy
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()
    
    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]
    
    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))
    
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    
    vis.run()
    vis.destroy_window()


def draw_slope_test_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                           draw_origin=True, cloud_range=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))
    
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    
    if cloud_range is not None:
        cloud_range = np.array(cloud_range)
        center = (cloud_range[3:] + cloud_range[:3]) / 2
        dim = cloud_range[3:] - cloud_range[:3]
        range_box = np.concatenate((center, dim, [0]))[None, :]
        
        vis = draw_box(vis, range_box, (0.5, 0.5, 0.5))
    
    vis.run()
    vis.destroy_window()


def draw_augment_test_scenes(non_aug_points, non_aug_gt_boxes, aug_points, aug_gt_boxes=None, cloud_range=None):
    if isinstance(non_aug_points, torch.Tensor):
        non_aug_points = non_aug_points.cpu().numpy()
    if isinstance(non_aug_gt_boxes, torch.Tensor):
        non_aug_gt_boxes = non_aug_gt_boxes.cpu().numpy()
    if isinstance(aug_points, torch.Tensor):
        aug_points = aug_points.cpu().numpy()
    if isinstance(aug_gt_boxes, torch.Tensor):
        aug_gt_boxes = aug_gt_boxes.cpu().numpy()
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.ones(3) * 0
    # draw origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    
    non_aug_pts = open3d.geometry.PointCloud()
    non_aug_pts.points = open3d.utility.Vector3dVector(non_aug_points[:, :3])
    non_aug_pts.colors = open3d.utility.Vector3dVector(np.ones((non_aug_points.shape[0], 3)) * np.array([1, 0, 0]))
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(aug_points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(np.ones((aug_points.shape[0], 3)))
    
    vis.get_render_option().point_size = 1.0
    vis.add_geometry(non_aug_pts)
    for _ in range(int(1e7)):
        pass
    vis.get_render_option().point_size = 2
    vis.add_geometry(pts)
    
    if aug_gt_boxes is not None:
        vis = draw_box(vis, aug_gt_boxes, (0, 1, 0), aug_gt_boxes[..., -1].astype(np.int))
    # if non_aug_gt_boxes is not None:
    #     if np.linalg.norm(aug_gt_boxes - non_aug_gt_boxes) > 0.1:
    #         vis = draw_box(vis, non_aug_gt_boxes, (0, 0, 1), non_aug_gt_boxes[..., -1].astype(np.int))
    if cloud_range is not None:
        cloud_range = np.array(cloud_range)
        center = (cloud_range[3:] + cloud_range[:3]) / 2
        dim = cloud_range[3:] - cloud_range[:3]
        range_box = np.concatenate((center, dim, [0]))[None, :]
        
        vis = draw_box(vis, range_box, (0.5, 0.5, 0.5))
    vis.run()
    vis.destroy_window()


def draw_segmentation(points, raw_points=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(raw_points, torch.Tensor):
        raw_points = raw_points.cpu().numpy()
    if isinstance(point_colors, torch.Tensor):
        point_colors = point_colors.cpu().numpy()
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().background_color = np.ones(3) * 0.3
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    draw_point = np.concatenate([points, raw_points[:, 0:3], ], axis=0)
    draw_color = np.concatenate([np.power(point_colors, 0.8), np.ones((points.shape[0], 3))], axis=0)
    
    vis.get_render_option().point_size = 2.0
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(draw_point[:, :3])
    vis.add_geometry(pts)
    pts.colors = open3d.utility.Vector3dVector(draw_color)
    
    vis.run()
    vis.destroy_window()


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)
    
    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    
    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))
    
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
    
    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    
    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    
    line_set.lines = open3d.utility.Vector2iVector(lines)
    
    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])
        
        vis.add_geometry(line_set)
    
    # if score is not None:
    #     corners = box3d.get_box_points()
    #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
