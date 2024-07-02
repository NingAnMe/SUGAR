#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2021 12 07

@author: Ning An

Contact: ninganme0317@gmail.com

"""
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_mean

from utils.negative_area_triangle import negative_area
from utils.auxi_data import get_distance_by_points_num


def get_face_coords(faces, vertex_xyz):
    """
    获取每个face三个顶点的坐标
    """
    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]
    xyz_a = vertex_xyz[a]
    xyz_b = vertex_xyz[b]
    xyz_c = vertex_xyz[c]
    return xyz_a, xyz_b, xyz_c


def get_edge_index(faces, device='cuda'):
    x = np.expand_dims(faces[:, 0], 1)
    y = np.expand_dims(faces[:, 1], 1)
    z = np.expand_dims(faces[:, 2], 1)

    a = np.concatenate([x, y], axis=1)
    b = np.concatenate([y, x], axis=1)
    c = np.concatenate([x, z], axis=1)
    d = np.concatenate([z, x], axis=1)
    e = np.concatenate([y, z], axis=1)
    f = np.concatenate([z, y], axis=1)

    edge_index = np.concatenate([a, b, c, d, e, f]).astype(int)
    edge_index = np.unique(edge_index, axis=0).astype(int)
    edge_index = edge_index[np.argsort(edge_index[:, 0])]
    edge_index = torch.from_numpy(edge_index).to(device)
    edge_index = edge_index.t().contiguous()
    return edge_index


def compute_sim_loss(predict, target, weight=None):
    if weight is None:
        corr_top = ((predict - predict.mean(dim=0, keepdim=True)) * (target - target.mean(dim=0, keepdim=True))).mean(
            dim=0, keepdim=True)
        corr_bottom = (predict.std(dim=0, keepdim=True) * target.std(dim=0, keepdim=True))
        corr = corr_top / corr_bottom
        loss_corr = (1 - corr).mean()
        loss_l2 = torch.mean((predict - target) ** 2)
        loss_l1 = torch.mean(torch.abs(predict - target))
    else:
        corr_top = ((predict - predict.mean(dim=0, keepdim=True)) * (target - target.mean(dim=0, keepdim=True)))
        corr_bottom = (predict.std(dim=0, keepdim=True) * target.std(dim=0, keepdim=True))
        corr = corr_top / corr_bottom

        loss_corr = 1 - corr.mean()

        # L2
        loss_l2 = (predict - target) ** 2
        w = torch.sum(loss_l2) / torch.sum(loss_l2 / weight)
        loss_l2 = (loss_l2 * w / weight).mean()

        # L1
        loss_l1 = torch.abs(predict - target)
        loss_l1 = (loss_l1 * weight).mean()

    return loss_corr, loss_l2, loss_l1


class LaplacianSmoothingLoss(nn.Module):
    def __init__(self, faces, xyz=None, rate=False, device='cuda'):
        super(LaplacianSmoothingLoss, self).__init__()
        self.nf = faces.shape[0]
        self.rate = rate

        edge_index = get_edge_index(faces.cpu().numpy(), device=device)
        self.row, self.col = edge_index

        if xyz is not None:
            self.nv = xyz.shape[0]
            self.distance = get_distance_by_points_num(self.nv)
            xyz = xyz * 100
            xyz_mean = scatter_mean(xyz[self.col], self.row, dim=0, dim_size=xyz.size(0))
            xyz_mean = xyz_mean / torch.norm(xyz_mean, dim=1, keepdim=True) * 100
            if self.rate:
                self.xyz_dif = torch.norm(xyz - xyz_mean, dim=1, keepdim=True)
            else:
                self.xyz_dif = (xyz - xyz_mean).abs()
        else:
            self.xyz_dif = None

    def forward(self, x, mean=True, norm=True):
        x = x * 100

        x_mean = scatter_mean(x[self.col], self.row, dim=0, dim_size=x.size(0))

        if norm:
            x_mean = x_mean / torch.norm(x_mean, dim=1, keepdim=True) * 100
            x_dif = (x - x_mean).abs()
            if self.rate:
                x_dif = torch.norm(x - x_mean, dim=1, keepdim=True)
                dif = (x_dif - self.xyz_dif).abs() / (self.distance / 2)
            else:
                dif = (x_dif - self.xyz_dif).abs()
        else:
            dif = (x - x_mean).abs()
        if mean:
            return dif.mean()
        else:
            return dif.sum()


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, true, pred, weight):
        loss_corr, loss_l2, loss_l1 = compute_sim_loss(true, pred, weight=weight)

        return loss_corr, loss_l2, loss_l1


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, true, pred):
        top = 2 * (true * pred).sum(dim=0)
        bottom = torch.clamp((true + pred).sum(dim=0), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1 - dice


class NegativeAreaLoss(nn.Module):
    def __init__(self, faces, xyz, device='cuda'):
        super(NegativeAreaLoss, self).__init__()
        self.nf = faces.shape[0]
        self.faces = faces.to(device)

        self.init_area = negative_area(self.faces, xyz * 100)

    def forward(self, x, mean=True):

        area = negative_area(self.faces, x * 100)
        zero = torch.zeros(1).to(x.device)
        dif = torch.max(-area, zero)

        if mean:
            me = dif.mean()
            return me
        else:
            return dif.sum()


class AreaLoss(nn.Module):
    def __init__(self, faces, xyz, device='cuda'):
        super(AreaLoss, self).__init__()
        self.nf = faces.shape[0]
        self.faces = faces.to(device)

        self.init_area = negative_area(self.faces, xyz).abs()

    def forward(self, x, mean=True):

        area = negative_area(self.faces, x).abs()
        dif = 1 - (area / self.init_area)
        dif = dif.abs()

        if mean:
            return dif.mean()
        else:
            return dif.sum()


class AngleLoss(nn.Module):
    def __init__(self, faces, xyz, device='cuda'):
        super(AngleLoss, self).__init__()
        self.nf = faces.shape[0]
        self.faces = faces.to(device)

        self.init_cos_theta, self.init_angle = self.angle(xyz)

    @staticmethod
    def vector_angle(vector_a, vector_b):
        norm_a = torch.norm(vector_a, dim=1)
        norm_b = torch.norm(vector_b, dim=1)
        axb = torch.sum(vector_a * vector_b, dim=1)
        cos_theta = axb / (norm_a * norm_b)
        angle = torch.acos(cos_theta)
        angle_d = angle * 180 / np.pi
        return cos_theta, angle_d

    def angle(self, xyz):
        """
        获取两个向量的夹角
        """
        xyz_a, xyz_b, xyz_c = get_face_coords(self.faces, xyz)
        cos_theta_a, angle_a = self.vector_angle(xyz_b - xyz_a, xyz_c - xyz_a)
        cos_theta_b, angle_b = self.vector_angle(xyz_a - xyz_b, xyz_c - xyz_b)
        cos_theta_c, angle_c = self.vector_angle(xyz_a - xyz_c, xyz_b - xyz_c)
        angle = torch.cat([angle_a.unsqueeze(1), angle_b.unsqueeze(1), angle_c.unsqueeze(1)], dim=1)
        cos_theta = torch.cat([cos_theta_a.unsqueeze(1), cos_theta_b.unsqueeze(1), cos_theta_c.unsqueeze(1)], dim=1)

        return cos_theta, angle

    def forward(self, x, mean=True):
        # fsaverage6 angle range = (0.3013, 0.5935)  (53.59, 72.46)

        cos_theta, angle = self.angle(x)

        dif = 1 - (cos_theta / self.init_cos_theta)
        dif = dif.abs()

        if mean:
            return dif.mean()
        else:
            return dif.sum()
