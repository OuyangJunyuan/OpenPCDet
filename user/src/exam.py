import numpy as np
import torch
import torch as t


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
import torch

def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])

conv = torch.nn.Conv1d(8,32,1)
print(count_parameters(conv))
# 288

linear = torch.nn.Linear(8,32)
print(count_parameters(linear))
# 288

print(conv.weight.shape)
# torch.Size([32, 8, 1])
print(linear.weight.shape)
# torch.Size([32, 8])

# use same initialization
linear.weight = torch.nn.Parameter(conv.weight.squeeze(2))
linear.bias = torch.nn.Parameter(conv.bias)

tensor = torch.randn(128,256,8)
permuted_tensor = tensor.permute(0,2,1).clone().contiguous()	# 注意此处进行了维度重新排列

out_linear = linear(tensor)
print(out_linear.mean())
# tensor(0.0067, grad_fn=<MeanBackward0>)

out_conv = conv(permuted_tensor)
print(out_conv.mean())
# tensor(0.0067, grad_fn=<MeanBackward0>)