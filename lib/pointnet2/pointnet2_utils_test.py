import torch
from pointnet2_utils import *

if __name__ == '__main__':
    B = 1
    M = 2
    N = 4
    C = 6
    features = torch.rand((B, C, M))

    known_xyz = torch.rand((B, M, 3))
    unkonwn_xyz = torch.rand((B, N, 3))
    dist, idx = three_nn(unkonwn_xyz, known_xyz)

    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm

    interpolated = three_interpolate(features, idx, weight)
    print(interpolated.shape, interpolated)