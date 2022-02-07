import numpy as np
import torch
import torch as t
from scipy.spatial.transform.rotation import Rotation

# from scipy.spatial import Delaunay
# import matplotlib.pyplot as plt
#
# points = np.array([
# 	[0, 1, 0],
# 	[1, 0, 0],
# 	[0, 0, 0],
# 	[0.5, 0.5, 1],
# 	[1, 1, 0]
# ])
# hull = Delaunay(points)
# print(hull.simplices.shape)
# print(points[hull.simplices[0, :]])
# print(hull.find_simplex(np.array([0.4, 0.4, 0.5])))
#
# def _gather_feat(feat, ind, mask=None):
#     dim = feat.size(2)  #
#     ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
#     feat = feat.gather(1, ind)
#     return feat
#
#
# #
# # p1 = t.tensor([1, 1, 1, 2, 2, 2, 3, 4])
# # l = []
# # l.append(p1[None,:])
# # l.append(p1[None,:])
# # l.append(p1[None,:])
# # p = t.cat((l), dim=0)
# # print(p)
#
# a = [1, 2, 3]
# c = [0]
# print(c + a)


from pcdet.utils.common_utils import rotate_points, rotate_points_along_z

n = 10
point = np.random.random(3 * n).reshape(-1, 3) * 10
angle = np.random.random(n).reshape(-1)
print(point)
print(angle)
r1 = rotate_points_along_z(point.reshape(-1, 1, 3), angle)
full_angle = np.hstack((angle[:, None], np.zeros_like(angle)[:, None], np.zeros_like(angle)[:, None]))
print(full_angle)
r2 = rotate_points(point.reshape(-1, 1, 3), full_angle)
print(r1)
print(r2)
