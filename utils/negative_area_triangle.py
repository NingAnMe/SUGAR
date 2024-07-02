import torch


def negative_area(faces, xyz):
    n = faces[:, 0]
    n0 = faces[:, 2]
    n1 = faces[:, 1]

    v0 = xyz[n] - xyz[n0]
    v1 = xyz[n1] - xyz[n]

    d1 = -v1[:, 1] * v0[:, 2] + v0[:, 1] * v1[:, 2]
    d2 = v1[:, 0] * v0[:, 2] - v0[:, 0] * v1[:, 2]
    d3 = -v1[:, 0] * v0[:, 1] + v0[:, 0] * v1[:, 1]

    dot = xyz[n][:, 0] * d1 + xyz[n][:, 1] * d2 + xyz[n][:, 2] * d3

    area = torch.sqrt(d1 * d1 + d2 * d2 + d3 * d3) / 2

    area = torch.where(dot < 0, area * -1, area)
    # area[dot < 0] *= -1

    return area


def count_negative_area(faces, xyz):
    area = negative_area(faces, xyz)
    index = area < 0
    count = index.sum()  # 面积为负的面
    print(f'negative area count : {count}')

    return count