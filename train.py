"""
input :
    1. rigid.sphere_moving  # 需要已经跑完刚性配准，在dir_result/目录有 sub_id/surf/*h.rigid.sphere文件
    2. sulc_moving
    3. curv_moving(可选)
    3. sphere_fixed
    4. sulc_fixed
output ：
    1. sphere_moved
    2. sulc_moved
    3. curv_moved(可选)
    4. sphere_moved_interp
    5. sulc_moved_interp
    6. curv_moved_interp(可选)
"""

import os
import torch

from sphere_register import train_val
from utils.auxi_data import get_geometry_all_level_torch
from weight import get_weight


freesurfer_home = '/opt/freesurfer'
# ========================== Predict Config ============================= #
hemisphere = 'lh'
dataset_name = '904'

config = dict()
config["device"] = 'cuda'

config['validation'] = False
config['is_rigid'] = False
config['rd_sample'] = True
config['is_da'] = True
config["hemisphere"] = hemisphere
config["subject_name"] = dataset_name

for split in range(1, 6):
    # input dir
    subject_train_list_file = '904_train_list.txt'
    subject_val_list_file = '904_val_list.txt'

    fsaverage_dir = f'{freesurfer_home}/subjects'  # FreeSurfer fsaverage6 目录
    data_recon_all_dir = f'/mnt/ngshare/SurfReg/Data_Processing'  # FreeSurfer recon-all 结果目录
    subject_list_dir = f'/mnt/ngshare/SurfReg/Data_Extra/list'  # subject_list 目录

    # output dir
    data_rigid_dir = f'/mnt/ngshare/SurfReg/Data_TrainResult/DataRigid'  # 结果文件目录
    data_result_dir = f'/mnt/ngshare/SurfReg/Data_TrainResult/TrainResult'  # 结果文件目录

    # input data
    config["dir_fixed"] = fsaverage_dir

    config["dir_train_recon"] = os.path.join(data_recon_all_dir, dataset_name)
    config["dir_train_rigid"] = os.path.join(data_rigid_dir, dataset_name)
    config["dir_train_result"] = os.path.join(data_result_dir, dataset_name)
    config["subject_train_list"] = os.path.join(subject_list_dir, subject_train_list_file)

    config["dir_val_recon"] = os.path.join(data_recon_all_dir, dataset_name)
    config["dir_val_rigid"] = os.path.join(data_rigid_dir, dataset_name)
    config["dir_val_result"] = os.path.join(data_result_dir, dataset_name)
    config["subject_val_list"] = os.path.join(subject_list_dir, subject_val_list_file)

    # train
    config["save_checkpoint"] = True  # 是否保存模型

    xyzs, faces = get_geometry_all_level_torch()
    config['xyz'] = xyzs
    config['face'] = faces

    # for ico_level in ['fsaverage3']:
    for ico_level in ['fsaverage4']:
    # for ico_level in ['fsaverage5']:
    # for ico_level in ['fsaverage6']:
        ico_levels = ['fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6']
        config["ico_level"] = ico_level
        ico_index = ico_levels.index(ico_level)
        config['ico_levels'] = ico_levels
        config['ico_index'] = ico_index
        level = [4, 5, 6, 7][ico_index]
        n_res = [3, 4, 5, 6][ico_index]
        n_vertex = [642, 2562, 10242, 40962][ico_index]
        config["hidden_channels"] = [64, 64, 64, 64][ico_index]

        config["epoch"] = [200, 200, 200, 200][ico_index]  # 训练多少个epoch
        config["output_channel"] = [3, 3, 3, 3][ico_index]  # 训练多少个epoch

        model_result_dir = [
            f'/mnt/ngshare/SurfReg/SUGAR/models/fsaverage3_split_{split}',
            f'/mnt/ngshare/SurfReg/SUGAR/models/fsaverage4_split_{split}',
            f'/mnt/ngshare/SurfReg/SUGAR/models/fsaverage5_split_{split}',
            f'/mnt/ngshare/SurfReg/SUGAR/models/fsaverage6_split_{split}',
        ]

        config["weight_corr"] = [0, 0, 0, 0][ico_index]  # lambda_cc 1 权重
        config["weight_l2"] = [1, 1, 1, 1.5][ico_index]  # lambda_sim 权重
        config["weight_l1"] = [0, 0, 0, 0][ico_index]  # lambda_sim 权重
        config["weight_dice"] = [4, 4, 4, 4][ico_index]  # 计算dice

        config["weight_smooth"] = [0, 0, 0, 0][ico_index]  # lambda_s  权重
        config['weight_laplacian'] = [2, 2, 2, 2][ico_index]
        config['weight_negative_area'] = [30, 30, 35, 35][ico_index]
        config['weight_angle'] = [1, 1, 1, 1][ico_index]
        config['weight_area'] = [1.5, 1.5, 1.5, 1.5][ico_index]

        config['sim_weight'] = torch.from_numpy(get_weight('sulc', hemisphere).astype(float)).float()

        config["feature"] = ['sulc', 'sulc', 'sulc', 'sulc']

        config["epoch_save"] = 2
        config["lrs_patience"] = 20
        config["lrs_T"] = (20, 5)
        config["lrs_max_T"] = 10
        config["warmup_period"] = 5
        config["input_dropout"] = 0
        config["hidden_dropout"] = 0

        model_files = []
        for i in range(len(ico_levels)):
            model_files.append(os.path.join(model_result_dir[i],
                                            f'{config["hemisphere"]}_NoRigid_904_{ico_levels[i]}.model'))
        config["model_files"] = model_files

        # model
        config["model_name"] = "GatUNet"
        if config["feature"] != 'sucu':
            config['sulc_curv_weight'] = None
        else:
            sulc_curv_weight = [[1, 0], [1, 0], [1, 0], [1, 0]]
            config["sulc_curv_weight"] = torch.tensor(sulc_curv_weight[ico_index], dtype=torch.float32,
                                                      device=config["device"])  # 设置sulc和curv的权重比例
        config["level"] = level  # 数据的细化等级
        config["n_res"] = n_res  # 模型提取特征时使用几个细化等级
        config["n_vertex"] = n_vertex  # 当前细化等级的顶点数量163842 40962
        config["optimizer"] = 'AdamW'
        config["lr"] = 0.0003
        config["lr_scheduler"] = 'CosineAnnealingLR'
        config["normalize_type"] = 'zscore'  # 计算与相邻顶点的push距离

        train_val(config=config)

    break
