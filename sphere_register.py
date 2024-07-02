import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_multi_level_multi_fixed import SphericalDataset
from loss import LaplacianSmoothingLoss, NegativeAreaLoss, AngleLoss, AreaLoss, DiceLoss, SimLoss
from utils.interp_fine import interp_sulc_curv_barycentric, interp_annot_knn
from utils.rotate_matrix import apply_rotate_matrix
from utils.interp_fine import resample_sphere_surface_barycentric, upsample_std_sphere_torch
from utils.auxi_data import get_points_num_by_ico_level


def interp_dir(dir_recon: str, dir_rigid: str, dir_fixed: str, subject_list_file: str, ico_level: str, is_rigid=False):
    """
    预处理：将native空间插值到fsaverage空间
    """
    # 2. get file_list
    if os.path.isfile(subject_list_file):
        with open(subject_list_file, 'r') as f_subject_list:
            sub_list = f_subject_list.readlines()
            sub_list = [i.strip() for i in sub_list]
    else:
        raise FileExistsError(f'{subject_list_file} is not exists')
    count_finish = 0
    count_all = len(sub_list)

    for sub_id in sub_list:
        surf_dir_recon = os.path.join(dir_recon, sub_id, 'surf')
        surf_dir_rigid = os.path.join(dir_rigid, sub_id, 'surf')
        if not os.path.exists(surf_dir_recon):
            continue
        if not os.path.exists(surf_dir_rigid):
            os.makedirs(surf_dir_rigid)
        for hemisphere in ['lh', 'rh']:
            sphere_fixed_file = os.path.join(dir_fixed, ico_level, 'surf', f'{hemisphere}.sphere')

            # ## 将刚性配准的结果插值到fsaverageN
            sulc_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.sulc')
            curv_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.curv')
            if is_rigid:
                data_type = 'orig'
                sphere_moving_file = os.path.join(surf_dir_recon, f'{hemisphere}.sphere')
            else:
                data_type = 'rigid'
                sphere_moving_file = os.path.join(surf_dir_rigid, f'{hemisphere}.rigid.sphere')  # 跑完刚性配准以后有这个文件

            if not os.path.exists(sphere_moving_file):
                continue

            sulc_moving_interp_file = os.path.join(surf_dir_rigid, f'{hemisphere}.{data_type}.interp_{ico_level}.sulc')
            curv_moving_interp_file = os.path.join(surf_dir_rigid, f'{hemisphere}.{data_type}.interp_{ico_level}.curv')
            sphere_moving_interp_file = os.path.join(surf_dir_rigid,
                                                     f'{hemisphere}.{data_type}.interp_{ico_level}.sphere')
            if not os.path.exists(sulc_moving_interp_file):
                interp_sulc_curv_barycentric(sulc_moving_file, curv_moving_file, sphere_moving_file, sphere_fixed_file,
                                             sulc_moving_interp_file, curv_moving_interp_file,
                                             sphere_moving_interp_file)
                print(f'interp: >>> {sulc_moving_interp_file}')
                print(f'interp: >>> {curv_moving_interp_file}')
                print(f'interp: >>> {sphere_moving_interp_file}')

            # ## 如果是NAMIC数据集，需要把人工标注的分区结果投影到fsaverageN
            annot_moving_file = os.path.join(dir_recon, sub_id, 'label', f'{hemisphere}.aparc.annot')
            annot_moving_interp_file = os.path.join(dir_rigid, sub_id, 'label',
                                                    f'{hemisphere}.gt.{data_type}.interp_{ico_level}.aparc.annot')
            if os.path.exists(annot_moving_file) and not os.path.exists(annot_moving_interp_file):
                label_dir_interp = os.path.join(dir_rigid, sub_id, 'label')
                if not os.path.exists(label_dir_interp):
                    os.makedirs(label_dir_interp)
                interp_annot_knn(sphere_moving_file, sphere_fixed_file, annot_moving_file, annot_moving_interp_file,
                                 device='cuda')
        count_finish += 1
    print(f'finish fsaverage interp : {count_finish} / {count_all}')


def infer(moving_datas, fixed_datas, models, faces, ico_levels, features, device='cuda'):
    assert len(moving_datas) > 0

    sulc_moving_fs6, curv_moving_fs6, xyz_moving_fs6, faces_moving_fs6, seg_moving_fs6 = moving_datas
    sulc_fixed_fs6, curv_fixed_fs6, xyz_fixed_fs6, faces_fixed_fs6, seg_fixed_fs6 = fixed_datas

    sulc_moving_fs6 = sulc_moving_fs6.T.to(device)
    sulc_fixed_fs6 = sulc_fixed_fs6.T.to(device)

    curv_moving_fs6 = curv_moving_fs6.T.to(device)
    curv_fixed_fs6 = curv_fixed_fs6.T.to(device)

    seg_moving_fs6 = seg_moving_fs6.squeeze().to(device)
    seg_fixed_fs6 = seg_fixed_fs6.squeeze().to(device)

    xyz_moving_fs6 = xyz_moving_fs6.squeeze().to(device)
    xyz_fixed_fs6 = xyz_fixed_fs6.squeeze().to(device)

    # NAMIC don't have 35
    if not torch.any(seg_moving_fs6 == 0):
        seg_fixed_fs6[seg_fixed_fs6 == 0] = 35

    # 904.et don't have 4
    if not torch.any(seg_moving_fs6 == 4):
        seg_fixed_fs6[seg_fixed_fs6 == 4] = 0

    xyz_moved = None
    seg_moving_lap = None
    for idx, model in enumerate(models):
        feature = features[idx]
        if feature == 'sulc':
            data_moving_fs6 = sulc_moving_fs6.to(device)
            data_fixed_fs6 = sulc_fixed_fs6.to(device)
        elif feature == 'curv':
            data_moving_fs6 = curv_moving_fs6.to(device)
            data_fixed_fs6 = curv_fixed_fs6.to(device)
        else:
            data_moving_fs6 = torch.cat((sulc_moving_fs6, curv_moving_fs6), 1).to(device)
            data_fixed_fs6 = torch.cat((sulc_fixed_fs6, curv_fixed_fs6), 1).to(device)

        ico_level = ico_levels[idx]
        points_num = get_points_num_by_ico_level(ico_level)
        faces_sphere = faces[ico_level].to(device)
        data_moving = data_moving_fs6[:points_num]
        data_fixed = data_fixed_fs6[:points_num]
        seg_moving = seg_moving_fs6[:points_num]
        seg_fixed = seg_fixed_fs6[:points_num]
        xyz_moving = xyz_moving_fs6[:points_num]
        xyz_fixed = xyz_fixed_fs6[:points_num]

        if xyz_moved is None:
            data_x = torch.cat((data_moving, data_fixed), 1).to(device)
            data_x = data_x.detach()

            xyz_moved_lap, euler_angle = model(data_x, xyz_moving, face=faces_sphere)

            xyz_moved = apply_rotate_matrix(euler_angle, xyz_moving, norm=True,
                                            en=model.en, face=faces_sphere)

            data_moving_lap = data_moving
            if seg_moving.sum() > 0:
                seg_moving_lap = seg_moving = F.one_hot(seg_moving).float().to(device)
            else:
                seg_moving_lap = None
        else:
            # upsample xyz_moved
            xyz_moved_upsample = upsample_std_sphere_torch(xyz_moved, norm=True)
            xyz_moved_upsample = xyz_moved_upsample.detach()

            # moved数据重采样
            moving_data_resample = resample_sphere_surface_barycentric(xyz_moved_upsample, xyz_fixed, data_moving)

            data_x = torch.cat((moving_data_resample, data_fixed), 1).to(device)

            xyz_moved_lap, euler_angle = model(data_x, xyz_moving, face=faces_sphere)


            euler_angle_interp_moved_upsample = resample_sphere_surface_barycentric(xyz_fixed, xyz_moved_upsample,
                                                                                    euler_angle)
            xyz_moved = apply_rotate_matrix(euler_angle_interp_moved_upsample, xyz_moved_upsample, norm=True,
                                            face=faces_sphere)

            if seg_moving.sum() > 0:
                seg_moving = F.one_hot(seg_moving).float().to(device)
                seg_moving_resample = resample_sphere_surface_barycentric(xyz_moved_upsample, xyz_fixed, seg_moving)
            else:
                seg_moving_resample = None

            data_moving_lap = moving_data_resample
            seg_moving_lap = seg_moving_resample

    if seg_moving_lap is not None:
        seg_fixed = F.one_hot(seg_fixed).float().to(device)
    else:
        seg_fixed = False

    return xyz_fixed, xyz_moved, xyz_moved_lap, data_fixed, data_moving, data_moving_lap, euler_angle, \
        seg_moving, seg_moving_lap, seg_fixed


def rd_sample_data(data_level6):
    sulc_moving_fs6, curv_moving_fs6, xyz_moving_fs6, faces_moving_fs6, seg_moving_fs6 = data_level6
    points_num = np.random.choice([642, 2562, 10242, 40962])
    if points_num != 40962:
        sulc_moving_fs6 = sulc_moving_fs6.T.squeeze()[:points_num]
        while len(sulc_moving_fs6) < 40962:
            sulc_moving_fs6 = upsample_std_sphere_torch(sulc_moving_fs6)
        data_level6[0] = sulc_moving_fs6.T.unsqueeze(0)
    return data_level6


def run_epoch(epoch, models, faces, optimizer, config, dataloader,
              sim_loss=None, dice_loss=None, smoothing_loss=None, laplacian_smoothing_loss=None,
              negative_area_loss=None, angle_loss=None, area_loss=None,
              is_train=False):
    device = config['device']
    features = config['feature']
    ico_levels = config['ico_levels']
    subs_loss = []
    for datas_moving, datas_fixed, sub_ids in dataloader:
        datas_moving = datas_moving[0]

        rd_sample = config['rd_sample']
        if rd_sample and is_train:
            datas_moving = rd_sample_data(datas_moving)

        datas_fixed = datas_fixed[0]
        sub_id = sub_ids[0]
        time_start = time.time()

        xyz_fixed, xyz_moved, xyz_moved_lap, fixed_data, data_moving, data_moving_lap, euler_angle, \
            seg_moving, seg_moving_lap, seg_fixed \
            = infer(datas_moving, datas_fixed, models, faces, ico_levels, features, device)
        # ################################### Curr-level Loss ##########################
        moved_data = resample_sphere_surface_barycentric(xyz_moved_lap, xyz_fixed, data_moving_lap, device=device)
        seg_moved = resample_sphere_surface_barycentric(xyz_moved_lap, xyz_fixed, seg_moving_lap, device=device)

        # cal sim loss
        loss_train = torch.zeros(1).to(device)

        weight = torch.sqrt(config['sim_weight'].to(device)) if config['sim_weight'] is not None else None
        if weight is not None:
            weight = weight[:len(moved_data)].unsqueeze(1)
        loss_corr, loss_l2, loss_l1 = sim_loss(moved_data, fixed_data, weight=weight)
        if config['weight_corr'] != 0:
            loss_train += loss_corr * config['weight_corr']
        if config['weight_l2'] != 0:
            loss_train += loss_l2 * config['weight_l2']
        if config['weight_l1'] != 0:
            loss_train += loss_l1 * config['weight_l1']

        if dice_loss is not None:
            loss_dice = dice_loss(seg_fixed, seg_moved)
        else:
            loss_dice = torch.zeros(1, dtype=torch.float32).to(device)
        if config['weight_dice'] != 0:
            loss_train += loss_dice * config['weight_dice']

        # cal reg loss
        if config['weight_smooth'] != 0:
            loss_smooth = smoothing_loss(euler_angle, mean=True, norm=False)
            loss_train += loss_smooth * config['weight_smooth']
        else:
            loss_smooth = torch.zeros(1, dtype=torch.float32)

        loss_laplacian = laplacian_smoothing_loss(xyz_moved_lap, mean=True, norm=True)
        if config['weight_laplacian'] != 0:
            loss_train += loss_laplacian * config['weight_laplacian']

        if negative_area_loss is not None:
            loss_negative_area = negative_area_loss(xyz_moved_lap, mean=True)
        else:
            loss_negative_area = torch.ones(1, dtype=torch.float32)

        if config['weight_negative_area'] != 0:
            loss_train += loss_negative_area * config['weight_negative_area']

        if angle_loss is not None:
            loss_angle = angle_loss(xyz_moved_lap, mean=True)
        else:
            loss_angle = torch.ones(1, dtype=torch.float32)
        if config['weight_angle'] != 0:
            loss_train += loss_angle * config['weight_angle']

        if area_loss is not None:
            loss_area = area_loss(xyz_moved_lap, mean=True)
        else:
            loss_area = torch.ones(1, dtype=torch.float32)
        if config['weight_area'] != 0:
            loss_train += loss_area * config['weight_area']

        if optimizer is not None:
            # backward
            optimizer.zero_grad()  # reset gradient
            loss_train.backward()
            optimizer.step()  # update parameters of net
        print_str = f'epoch: {epoch:05d}  sub: {sub_id} ' \
                    f'time/iter: {time.time() - time_start:0.4f}s, loss: {loss_train.item():0.4f}, ' \
                    f'corr : {loss_corr.item():0.4f}, l2 : {loss_l2.item():0.4f}, l1 : {loss_l1.item():0.4f}, ' \
                    f' s : {loss_smooth.item():0.4f}, '
        print_str += f'lap : {loss_laplacian.item():0.4f}, '
        print_str += f'neg : {loss_negative_area.item():0.4f}, '
        print_str += f'angle : {loss_angle.item():0.4f}, '
        print_str += f'area : {loss_area.item():0.4f}, '
        print_str += f'dice : {loss_dice.item():0.4f}, '
        print(print_str)
        subs_loss.append(
            [epoch, loss_train.item(), 1 - loss_corr.item(), loss_l2.item(), loss_l1.item(), loss_smooth.item(),
             loss_laplacian.item(),
             time.time() - time_start, loss_negative_area.item(),
             1 - loss_dice.item(), loss_angle.item(), loss_area.item()])

    return subs_loss


def train(epoch, models, optimizer, config, **losses):
    for model in models[:-1]:
        model.eval()
    models[-1].train()

    # 获取config_train的配置
    feature = config['feature']  # 加载的数据类型
    faces = config['face']
    hemisphere = config["hemisphere"]

    # 数据目录
    dir_fixed = config["dir_fixed"]  # fixed数据目录
    dir_rigid = config["dir_train_rigid"]  # 结果目录

    # 2. get file_list
    if os.path.isfile(config["subject_train_list"]):
        with open(config["subject_train_list"], 'r') as f_subject_list:
            sub_list = f_subject_list.readlines()
            sub_list = [i.strip() for i in sub_list]
    else:
        raise FileExistsError(f'{config["subject_train_list"]} is not exists')

    dataset_train = SphericalDataset(sub_list, dir_fixed, dir_rigid,
                                     hemisphere, feature=feature, norm_type=config["normalize_type"],
                                     ico_levels=['fsaverage6'],
                                     seg=True, is_train=True, is_da=config['is_da'], is_rigid=config['is_rigid'])
    dataloader_train = DataLoader(dataset=dataset_train, shuffle=True, batch_size=1, num_workers=0)
    subs_loss = run_epoch(epoch, models, faces, optimizer, config, dataloader_train, is_train=True, **losses)

    return np.array(subs_loss)


@torch.no_grad()
def val(epoch, models, config, **losses):
    for model in models:
        model.eval()
    # 获取config_train的配置
    feature = config['feature']  # 加载的数据类型
    hemisphere = config["hemisphere"]
    faces = config['face']

    # 数据目录
    dir_fixed = config["dir_fixed"]  # fixed数据目录
    dir_rigid = config["dir_val_rigid"]  # 结果目录

    # 2. get file_list
    if os.path.isfile(config["subject_val_list"]):
        with open(config["subject_val_list"], 'r') as f_subject_list:
            sub_list = f_subject_list.readlines()
            sub_list = [i.strip() for i in sub_list]
    else:
        raise FileExistsError(f'{config["subject_val_list"]} is not exists')

    dataset_train = SphericalDataset(sub_list, dir_fixed, dir_rigid,
                                     hemisphere, feature=feature, norm_type=config["normalize_type"],
                                     ico_levels=['fsaverage6'],
                                     seg=True, is_train=False, is_da=False, is_rigid=config['is_rigid'])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, num_workers=0)

    optimizer = None

    subs_loss = run_epoch(epoch, models, faces, optimizer, config, dataloader_train, **losses)

    return np.array(subs_loss)


@torch.no_grad()
def hemisphere_predict(models, config, hemisphere,
                       dir_recon, dir_rigid=None, dir_result=None, subject_list_file=None,
                       seg=False, **losses):
    for model in models:
        model.eval()
    # 获取config_train的配置
    feature = config['feature']  # 加载的数据类型
    faces = config['face']

    # 数据目录
    dir_fixed = config["dir_fixed"]  # fixed数据目录

    # 2. get file_list
    if os.path.isfile(subject_list_file):
        with open(subject_list_file, 'r') as f_subject_list:
            sub_list = f_subject_list.readlines()
            sub_list = [i.strip() for i in sub_list]
    else:
        raise FileExistsError(f'{subject_list_file} is not exists')

    dataset_train = SphericalDataset(sub_list, dir_fixed, dir_rigid,
                                     hemisphere, feature=feature, norm_type=config["normalize_type"],
                                     ico_levels=['fsaverage6'],
                                     seg=seg, is_train=False, is_da=False, is_rigid=config['is_rigid'])
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, num_workers=0)

    epoch = 0
    optimizer = None

    subs_loss = run_epoch(epoch, models, faces, optimizer, config, dataloader_train, **losses)

    return subs_loss


def train_val(config):
    # 获取config_train的配置
    device = config['device']  # 使用的硬件
    epoch_total = config['epoch']

    xyz_fixed = config['xyz'][config["ico_level"]].to(device)
    faces_fixed = config['face'][config["ico_level"]].to(device)

    sim_loss = SimLoss()
    dice_loss = DiceLoss()
    smoothing_loss = LaplacianSmoothingLoss(faces_fixed, device=device)
    laplacian_smoothing_loss = LaplacianSmoothingLoss(faces_fixed, xyz_fixed, rate=True, device=device)
    negative_area_loss = NegativeAreaLoss(faces_fixed, xyz_fixed, device=device)
    angle_loss = AngleLoss(faces_fixed, xyz_fixed, device=device)
    area_loss = AreaLoss(faces_fixed, xyz_fixed, device=device)

    losses = {
        'sim_loss': sim_loss,
        'dice_loss': dice_loss,
        'smoothing_loss': smoothing_loss,
        'laplacian_smoothing_loss': laplacian_smoothing_loss,
        'negative_area_loss': negative_area_loss,
        'angle_loss': angle_loss,
        'area_loss': area_loss,
    }

    # 1. interp train and val file
    interp_dir(config["dir_train_recon"], config["dir_train_rigid"], config["dir_fixed"], config['subject_train_list'],
               'fsaverage6', is_rigid=config['is_rigid'])
    interp_dir(config["dir_val_recon"], config["dir_val_rigid"], config["dir_fixed"], config['subject_val_list'],
               'fsaverage6', is_rigid=config['is_rigid'])

    models = []
    for model_file in config['model_files'][:config["ico_index"] + 1][:-1]:
        print(f'<<< model : {model_file}')
        model = torch.load(model_file)['model']
        model.to(device)
        model.eval()
        models.append(model)

    # 2. model set
    feature = config['feature'][config['ico_index']]  # 加载的数据类型
    if feature == 'sulc':
        in_ch = 2
    elif feature == 'sucu':
        in_ch = 4
    elif feature == 'curv':
        in_ch = 2
    else:
        raise KeyError("feature is error")
    if config["model_name"] == 'GatUNet':
        from gatunet_model import GatUNet

        model_params = dict(
            in_channels=in_ch + 18,
            out_channels=config["output_channel"],
            num_heads=8,
            dropout=config["hidden_dropout"],
            use_position_decoding=True,
            use_residual=False,
            ico_level=config['ico_level'],
            input_dropout=config['input_dropout'],
            euler_scale=None,
            rigid=config['is_rigid']
        )
        model = GatUNet(**model_params)
    else:
        raise KeyError(f"model name error : {config.get('model_name')}")

    model.to(device)
    models.append(model)

    if config["optimizer"] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=3e-7)
    elif config["optimizer"] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    else:
        raise KeyError(f"error {config['optimizer']}")
    # set lr_scheduler
    if config["lr_scheduler"] == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.001, step_size_up=50)
    elif config["lr_scheduler"] == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["lrs_max_T"], eta_min=0)
    else:
        lr_scheduler = None

    import pytorch_warmup as warmup
    warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=config["warmup_period"])

    # 3. 非刚性配准到fixed：得到旋转矩阵和interp_rigid
    val_losses = None
    best_vali_loss = np.inf
    epoches = []
    epoch_save = config["epoch_save"]
    for epoch_now in range(epoch_total + 1):
        print(epoch_now)

        # 3. train model
        print(f"lr: {optimizer.param_groups[-1]['lr']}")
        train(epoch_now, models, optimizer, config, **losses)  # 更新模型权重
        with warmup_scheduler.dampening():
            lr_scheduler.step()
        if epoch_now % epoch_save != 0:
            continue

        epoches.append(epoch_now)

        # 模型输出目录
        best_model = config['model_files'][:config["ico_index"] + 1][-1]
        model_dir = os.path.dirname(best_model)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        # 4. val model
        val_loss = val(epoch_now, models, config, **losses)  # 验证和保存模型
        val_m = val_loss.mean(axis=0).reshape(1, -1)
        val_losses = val_m if val_losses is None else np.concatenate((val_losses, val_m), axis=0)
        vali_loss_now = val_losses[:, 1][-1]

        if vali_loss_now <= best_vali_loss:
            checkpoint = {
                'model': model
            }
            torch.save(checkpoint, best_model)
            print(f'save checkpoint: {best_model}')

            best_vali_loss = vali_loss_now
