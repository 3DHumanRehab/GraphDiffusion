"""
----------------------------------------------------------------------------------------------
PointHMR Official Code
Copyright (c) Deep Computer Vision Lab. (DCVL.) All Rights Reserved
Licensed under the MIT license.
----------------------------------------------------------------------------------------------
Modified from MeshGraphormer (https://github.com/microsoft/MeshGraphormer)
Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshGraphormer/blob/main/LICENSE for details]
----------------------------------------------------------------------------------------------
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import json
import time
import datetime
import torch
from torchvision.utils import make_grid
import gc
import numpy as np
import cv2
import sys
import os
# a = os.path.abspath(__file__)
# a = os.path.dirname(a)
# a = os.path.dirname(a)
# a = os.path.dirname(a)
# sys.path.append(a)
sys.path.append("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc")
# from src.modeling.model.network_EMA_4 import PointHMR
from src.modeling.model.modeling_fastmetro_EMA_4_S_adapt import FastMETRO_Body_Network as FastMETRO_Network
from src.modeling._smpl import SMPL, Mesh
from src.modeling.hrnet.hrnet_cls_net_featmaps_adapt import get_cls_net
from src.modeling._smpl import SMPL, Mesh
import src.modeling.data.config as cfg
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
from src.datasets.build_byvideo import make_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection

from src.tools.loss import *

from azureml.core.run import Run
aml_run = Run.get_context()
import torch
from torch import Tensor, nn
def calculate_mpjpe(predicted, gt):
    mpjpe = torch.sqrt(((predicted - gt)**2).sum(dim=-1))
    mpjpe = mpjpe.mean(dim=-1)
    return mpjpe[~mpjpe.isnan()]


def calculate_pampjpe(predicted, gt):
    S1_hat = batch_compute_similarity_transform_torch(predicted, gt)
    # per-frame accuracy after procrustes alignment
    mpjpe_pa = torch.sqrt(((S1_hat - gt)**2).sum(dim=-1))
    mpjpe_pa = mpjpe_pa.mean(dim=-1)
    return mpjpe_pa[~mpjpe_pa.isnan()]
def calculate_accel_error(predicted, gt):
    accel_err = compute_error_accel(joints_pred=predicted, joints_gt=gt)

    accel_err=torch.concat((torch.tensor([0]).to(accel_err.device),accel_err,torch.tensor([0]).to(accel_err.device)))
    return accel_err
def slide_window_to_sequence(slide_window,window_step,window_size):
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

    sequence = torch.stack(sequence)

    return sequence


class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.
    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNet(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.
    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .
    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)
    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Build decoder layers
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        x=x.to(torch.float32)

        assert T == self.window_size, (
            'Input sequence length must be equal to the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, output_size]

        return x

pred_j3ds = []
target_j3ds = []
pred_verts = []
target_theta = []

def save_checkpoint(model, args, optimzier, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            torch.save(optimzier.state_dict(), op.join(checkpoint_dir, 'op_state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def save_scores(args, split, mpjpe, pampjpe, mpve):
    eval_log = []
    res = {}
    res['mPJPE'] = mpjpe
    res['PAmPJPE'] = pampjpe
    res['mPVE'] = mpve
    eval_log.append(res)
    with open(op.join(args.output_dir, split+'_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)))
    # for param_group in optimizer.param_groups:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def run(args, train_dataloader, val_dataloader, Network, mesh_sampler, smpl, renderer):
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    if iters_per_epoch<1000:
        args.logging_steps = 500

    optimizer = torch.optim.Adam(params=list(Network.parameters()),
                                 lr=args.lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=0)
    device_ids = [0, 1]
    optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
    # downsample = torch.nn.DataParallel(mesh_sampler.downsample, device_ids=device_ids)

    if args.resume_op_checkpoint is not None:
        op_states = torch.load(args.resume_op_checkpoint, map_location=args.device)
        for k, v in op_states.items():
            op_states[k] = v.cpu()
        optimizer.load_state_dict(op_states)
        del op_states

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    criterion_heatmap = torch.nn.MSELoss().cuda(args.device)

    MAP = torch.eye(112 * 112).cuda()

    if args.distributed:
        Network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Network)
        print("share batch")
        Network = torch.nn.parallel.DistributedDataParallel(
            Network, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    start_training_time = time.time()
    end = time.time()
    Network.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints = AverageMeter()
    log_loss_3djoints = AverageMeter()
    log_loss_vertices = AverageMeter()
    log_eval_metrics = EvalMetricsLogger()
    print("train_data_num:",train_dataloader.__len__())
    
    for iteration, (img_keys, images, annotations) in enumerate(train_dataloader):
        # gc.collect()
        # torch.cuda.empty_cache()
        Network.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        images = images.cuda(args.device)
        B,S,C,H,W = images.shape
        batch_size = B*S
        images = images.view(B*S,C,H,W)
        gt_2d_joints = annotations['joints_2d'].cuda(args.device)
        B,S,point_num,point_xyz = gt_2d_joints.shape
        gt_2d_joints = gt_2d_joints.view(B*S,point_num,point_xyz)
        gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]
        has_2d_joints = annotations['has_2d_joints'].cuda(args.device)

        gt_3d_joints = annotations['joints_3d'].cuda(args.device)
        B,S,point_num,point_xyz = gt_3d_joints.shape
        gt_3d_joints = gt_3d_joints.view(B*S,point_num,point_xyz)
        gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
        gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:]
        gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]
        has_3d_joints = annotations['has_3d_joints'].cuda(args.device)

        gt_pose = annotations['pose'].cuda(args.device)
        gt_betas = annotations['betas'].cuda(args.device)
        has_smpl = annotations['has_smpl'].cuda(args.device)



   
        gt_pose = gt_pose.view(B*S,-1)
        gt_betas = gt_betas.view(B*S,-1) 
        has_smpl = has_smpl.view(B*S) 
        has_3d_joints = has_3d_joints.view(B*S)
        # B,S,point_num,point_xyz = gt_joints.shape
        # gt_joints = gt_joints.view(B*S,point_num,point_xyz)

        # generate simplified mesh
        gt_vertices = smpl(gt_pose, gt_betas)
        gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices, n1=0, n2=2)
        gt_vertices_sub = mesh_sampler.downsample(gt_vertices)

        # normalize gt based on smpl's pelvis
        gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :]
        gt_vertices_sub = gt_vertices_sub - gt_smpl_3d_pelvis[:, None, :]
        gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]

        # forward-pass
        outputs = Network(images)
        need_hloss = True
        pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, heatmap = outputs

        pred_2d_joints_from_smpl, loss_2d_joints, loss_3d_joints, loss_vertices, loss = calc_losses(args, pred_camera, pred_3d_joints, pred_vertices_sub2,
                                                                      pred_vertices_sub, pred_vertices, gt_vertices_sub2, gt_vertices_sub,
                                                                      gt_vertices, gt_3d_joints, gt_2d_joints, has_3d_joints, has_2d_joints,
                                                                      has_smpl, criterion_keypoints, criterion_2d_keypoints, criterion_vertices,smpl,heatmap, criterion_heatmap, MAP, need_hloss)



        # update logs
        log_loss_2djoints.update(loss_2d_joints.item(), batch_size)
        log_loss_3djoints.update(loss_3d_joints.item(), batch_size)
        log_loss_vertices.update(loss_vertices.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back prop
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        # optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % args.logging_steps == 0 or iteration == max_iter:
        # if True:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2djoints.avg, log_loss_3djoints.avg, log_loss_vertices.avg, batch_time.avg, data_time.avg, 
                    optimizer.param_groups[0]['lr'])
            )

            aml_run.log(name='Loss', value=float(log_losses.avg))
            aml_run.log(name='3d joint Loss', value=float(log_loss_3djoints.avg))
            aml_run.log(name='2d joint Loss', value=float(log_loss_2djoints.avg))
            aml_run.log(name='vertex Loss', value=float(log_loss_vertices.avg))

            visual_imgs = visualize_mesh(   renderer,
                                            annotations['ori_img'].detach(),
                                            annotations['joints_2d'].detach(),
                                            pred_vertices.detach(), 
                                            pred_camera.detach(),
                                            pred_2d_joints_from_smpl.detach())
            visual_imgs = visual_imgs.transpose(0,1)
            visual_imgs = visual_imgs.transpose(1,2)
            visual_imgs = np.asarray(visual_imgs)

            if is_main_process()==True:
                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]*255))
                aml_run.log_image(name='visual results', path=temp_fname)

        if iteration % iters_per_epoch == 0:
            checkpoint_dir = save_checkpoint(Network, args, optimizer, 0, 0)
        # if True:

            val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                                Network,
                                                criterion_keypoints, 
                                                criterion_vertices, 
                                                epoch, 
                                                smpl,
                                                mesh_sampler)
            aml_run.log(name='mPVE', value=float(1000*val_mPVE))
            aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
            aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))
            logger.info(
                ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
                + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, Data Count: {:6.2f}'.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE, val_count)
            )

            if val_PAmPJPE<log_eval_metrics.PAmPJPE:
                checkpoint_dir = save_checkpoint(Network, args, optimizer, epoch, iteration)
                log_eval_metrics.update(val_mPVE, val_mPJPE, val_PAmPJPE, epoch)

        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    try:
        checkpoint_dir = save_checkpoint(Network, args, epoch, iteration)
    except:
        print(1)
    logger.info(
        ' Best Results:'
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, at epoch {:6.2f}'.format(1000*log_eval_metrics.mPVE, 1000*log_eval_metrics.mPJPE, 1000*log_eval_metrics.PAmPJPE, log_eval_metrics.epoch)
    )
    



    

def compute_error_verts(pred_verts, target_verts=None, target_theta=None):
    """
    Computes MPJPE over 6890 surface vertices.
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    if target_verts is None:
        from lib.models.smpl import SMPL_MODEL_DIR
        from lib.models.smpl import SMPL
        device = 'cpu'
        smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=1, # target_theta.shape[0],
        ).to(device)

        betas = torch.from_numpy(target_theta[:,75:]).to(device)
        pose = torch.from_numpy(target_theta[:,3:75]).to(device)

        target_verts = []
        b_ = torch.split(betas, 5000)
        p_ = torch.split(pose, 5000)

        for b,p in zip(b_,p_):
            output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
            target_verts.append(output.vertices.detach().cpu().numpy())

        target_verts = np.concatenate(target_verts, axis=0)

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)

def compute_accel(joints):
       """
       Computes acceleration of 3D joints.
       Args:
           joints (Nx25x3).
       Returns:
           Accelerations (N-2).
       """
       joints = joints.detach().cpu()
       velocities = joints[1:] - joints[:-1] # 计算相邻帧之间的位移，得到关节的速度 velocities。
       acceleration = velocities[1:] - velocities[:-1] # 相邻帧速度差分, 得到加速度
       acceleration_normed = np.linalg.norm(acceleration, axis=2) # 二范数
       return np.mean(acceleration_normed, axis=1)
#VIBE    
# def compute_error_accel(joints_gt, joints_pred, vis=None):
#     """
#     Computes acceleration error:
#         1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
#     Note that for each frame that is not visible, three entries in the
#     acceleration error should be zero'd out.
#     Args:
#         joints_gt (Nx14x3).
#         joints_pred (Nx14x3).
#         vis (N).
#     Returns:
#         error_accel (N-2).
#     """
#     # (N-2)x14x3
#     joints_gt = joints_gt.detach().cpu()
#     joints_pred = joints_pred.detach().cpu()
#     accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
#     accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

#     normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

#     if vis is None:
#         new_vis = np.ones(len(normed), dtype=bool)
#     else:
#         invis = np.logical_not(vis)
#         invis1 = np.roll(invis, -1)
#         invis2 = np.roll(invis, -2)
#         new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
#         new_vis = np.logical_not(new_invis)

#     return np.mean(normed[new_vis], axis=1)  

#smooth
def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = torch.norm(accel_pred - accel_gt, dim=2)

    if vis is None:
        new_vis = torch.ones(len(normed), dtype=bool)
    else:
        invis = torch.logical_not(vis)
        invis1 = torch.roll(invis, -1)
        invis2 = torch.roll(invis, -2)
        new_invis = torch.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = torch.logical_not(new_invis)

    acc=torch.mean(normed[new_vis], axis=1)

    return acc[~acc.isnan()]

def calcul(evaluation_accumulators):
    # global pred_j3ds 
    # global target_j3ds 
    # global pred_verts
    # global target_theta
    

    pred_j3ds = evaluation_accumulators['pred_j3d']
    target_j3ds = evaluation_accumulators['target_j3d']
    pred_verts = evaluation_accumulators['pred_verts']
    # pred_j3ds = torch.from_numpy(pred_j3ds).float()
    # target_j3ds = torch.from_numpy(target_j3ds).float()


    # pred_j3ds = np.array(pred_j3ds)
    # target_j3ds = np.array(target_j3ds)
    # pred_verts = np.array(pred_verts)
    # target_theta = np.array(target_theta)

    # pred_j3ds = torch.from_numpy(pred_j3ds).float()
    # target_j3ds = torch.from_numpy(target_j3ds).float()

    # print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
    pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
    target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0


    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    # Absolute error (MPJPE)
    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    # errors_pa = torch.sqrt(((S1_hat - target_j3ds.reshape(target_j3ds.shape[0] * target_j3ds.shape[1], 14, 3)) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    m2mm = 1000

    # pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
    accel = np.mean(compute_accel(pred_j3ds)) * m2mm
    accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
    mpjpe = np.mean(errors) * m2mm
    pa_mpjpe = np.mean(errors_pa) * m2mm

    eval_dict = {
        'mpjpe': mpjpe,
        'pa-mpjpe': pa_mpjpe,
        # 'pve': pve,
        'accel': accel,
        'accel_err': accel_err
    }
    # print(eval_dict)
    return eval_dict

    aa = 0

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat








def run_eval_general(args, val_dataloader, Network, smpl, mesh_sampler):
    smpl.eval()
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    epoch = 0
    if args.distributed:
        Network = torch.nn.parallel.DistributedDataParallel(
            Network, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    
    Network.eval()
    run_validate(args, val_dataloader, 
                                    Network,
                                    criterion_keypoints, 
                                    criterion_vertices, 
                                    epoch,
                                    smpl,
                                    mesh_sampler)
    # val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
    #                                 Network,
    #                                 criterion_keypoints, 
    #                                 criterion_vertices, 
    #                                 epoch, 
    #                                 smpl,
    #                                 mesh_sampler)

    # aml_run.log(name='mPVE', value=float(1000*val_mPVE))
    # aml_run.log(name='mPJPE', value=float(1000*val_mPJPE))
    # aml_run.log(name='PAmPJPE', value=float(1000*val_PAmPJPE))

    # logger.info(
    #     ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
    #     + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE)
    # )
    # # checkpoint_dir = save_checkpoint(Network, args, 0, 0)
    # return

def run_validate(args, val_dataloader, Network, criterion_keypoints, criterion_vertices, epoch, smpl, mesh_sampler):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    # switch to evaluate mode
    Network.eval()
    smpl.eval()

    with torch.no_grad():
        global pred_j3ds
        global target_j3ds
        batch_accel_result = 0
        batch_accel = []
        gt_accel = []
        # denoise_accel = torch.empty((0))
        # input_accel = torch.empty((0))
        input_mpve = torch.empty((0))
        input_mpjpe = torch.empty((0))
        input_pampjpe = torch.empty((0))
        input_accel = torch.empty((0))

        denoise_mpve = torch.empty((0))
        denoise_mpjpe = torch.empty((0))
        denoise_pampjpe = torch.empty((0))
        denoise_accel = torch.empty((0))

        # end = time.time()
        import json

        # 创建一个示例字典
        person_dict = {
        }

        smooth_model = SmoothNet(window_size=32,
        output_size=32,
        hidden_size=512,
        res_hidden_size=128,
        num_blocks=3,
        dropout=0.5).cuda()
        # checkpoint = torch.load("/HOME/HOME/Zhongzhangnan/SmoothNet/data/checkpoints/pw3d_spin_3D/checkpoint_32.pth.tar")
        checkpoint = torch.load("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/models/checkpoint_32.pth.tar")
        
        # checkpoint = torch.load("/HOME/HOME/Zhongzhangnan/SmoothNet/data/checkpoints/pw3d_eft_3D/checkpoint_32.pth.tar")
        performance = checkpoint['performance']
        smooth_model.load_state_dict(checkpoint['state_dict'],strict=False)
        # evaluation_accumulators['target_theta'] = []
        acc_all = {}
        print("val_data_num:",val_dataloader.__len__())
        for i, (img_keys, images, annotations) in enumerate(val_dataloader):
            # if i <26:
            #     continue
            # if os.path.exists("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test"):
            #     import shutil
            #     shutil.rmtree("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test")
            # os.mkdir("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test")
            evaluation_accumulators = {}
            evaluation_accumulators['pred_j3d'] = []
            evaluation_accumulators['smooth_j3d'] = []
            evaluation_accumulators['target_j3d'] = []
            # evaluation_accumulators['pred_verts'] = []
            evaluation_accumulators['pred_camera'] = []
            batch_size = images.size(0)
            # compute output
            images = images
            B,S,C,H,W = images.shape
        
            new_S = 0
            images = annotations['transfromed_img'][:,:S-new_S,:,:]
            gt_3d_joints = annotations['joints_3d'][:,:S-new_S,:,:].cuda(args.device)
            B,S,point_num,point_xyz = gt_3d_joints.shape

            gt_3d_joints = gt_3d_joints.view(B*S,point_num,point_xyz)
            gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3]
            gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:]

            gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :]

            # evaluation_accumulators['target_j3d'].append(gt_3d_joints[:,:,:3].cpu())

            has_3d_joints = annotations['has_3d_joints'][:,:S-new_S].cuda(args.device)

            gt_pose = annotations['pose'][:,:S-new_S,:].cuda(args.device)
            gt_betas = annotations['betas'][:,:S-new_S,:].cuda(args.device)
            has_smpl = annotations['has_smpl'][:,:S-new_S,:].cuda(args.device)

            B,S,C,H,W = images.shape
            batch_size = B*S
            # print(B)
            images = images.view(B*S,C,H,W)
    
            gt_pose = gt_pose.view(B*S,-1)
            gt_betas = gt_betas.view(B*S,-1)
            has_smpl = has_smpl.view(B*S)
            has_3d_joints = has_3d_joints.view(B*S)
            # B,S,point_num,point_xyz = gt_joints.shape
            # gt_joints = gt_joints.view(B*S,point_num,point_xyz)
            # generate simplified mesh
            gt_vertices = smpl(gt_pose, gt_betas)
            gt_vertices_sub = mesh_sampler.downsample(gt_vertices)
            gt_vertices_sub2 = mesh_sampler.downsample(gt_vertices_sub, n1=1, n2=2)

            # normalize gt based on smpl pelvis
            gt_smpl_3d_joints = smpl.get_h36m_joints(gt_vertices)
            gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            gt_vertices_sub2 = gt_vertices_sub2 - gt_smpl_3d_pelvis[:, None, :]
            gt_vertices = gt_vertices - gt_smpl_3d_pelvis[:, None, :]

            # forward-pass
            length,_,_,_ = images.shape
            step = 32
            for b in range(0,length-step):
                # print(b,b,b+32*2)
                input = images[b:b+step]
                input = input.cuda(args.device)
                
                # pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, _ = Network(input)
                out = Network(input)
                pred_camera, pred_3d_joints,pred_3d_vertices_fine = out['pred_cam'], out['pred_3d_joints'], out['pred_3d_vertices_fine']
                # pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_3d_vertices_fine) # batch_size X 17 X 3
                # pred_3d_joints_from_smpl_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
                # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:] # batch_size X 14 X 3
                # # normalize predicted vertices 
                # pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 6890 X 3

                pred_save = pred_3d_joints.clone()

                # pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
                # pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
                # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
                # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
                # evaluation_accumulators['pred_j3d'].append(pred_3d_joints_from_smpl.cpu())
                S = 32
                # pred_3d_joints = new_dict['pred_j3d'][:1152]
                B,C,N = pred_3d_joints.shape
                smooth_3d_joints = pred_3d_joints.flatten(1)
                smooth_3d_joints = smooth_3d_joints.reshape(B//S,S,-1)
                # smooth_3d_joints = smooth_3d_joints.transpose(2,1)
                # smooth_3d_joints = smooth_model(smooth_3d_joints.cuda())
                # smooth_3d_joints = smooth_3d_joints.transpose(2,1)
                # pred_3d_joints = pred_3d_joints.reshape(B,C,N)
                # new_dict['pred_j3d'][:1152] = pred_3d_joints.cpu()
                    
                gt_save = gt_3d_joints[b:b+step,:,:3]
                gt_save = gt_save.flatten(1)
                gt_save = gt_save.reshape(B//S,S,-1)

                pred_save = pred_save.flatten(1)
                pred_save = pred_save.reshape(B//S,S,-1)
                evaluation_accumulators['pred_j3d'].append(pred_save.cpu())
                evaluation_accumulators['smooth_j3d'] .append(smooth_3d_joints.cpu())
                evaluation_accumulators['target_j3d'].append(gt_save.cpu())
                # evaluation_accumulators['pred_verts'].append(pred_3d_vertices_fine.cpu())
                # evaluation_accumulators['gt_verts'].append(gt_vertices.cpu())
                evaluation_accumulators['pred_camera'].append(pred_camera.cpu())


        
            # if b+step*2!=length:
            #     input = images[-step*2:]
            #     input = input.cuda(args.device)
            #     pred_camera, pred_3d_joints, pred_vertices_sub2, pred_vertices_sub, pred_vertices, _ = Network(input)
            #     ll = length - (b+step*2)
            #     # pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
            #     # pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            #     # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            #     # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            #     # evaluation_accumulators['pred_j3d'].append(pred_3d_joints_from_smpl[-ll:].cpu())
            #     S = 8
            #     # pred_3d_joints = new_dict['pred_j3d'][:1152]
            #     B,C,N = pred_3d_joints.shape
            #     pred_3d_joints = pred_3d_joints.flatten(1)
            #     pred_3d_joints = pred_3d_joints.reshape(B//S,S,-1)
            #     pred_3d_joints = pred_3d_joints.transpose(2,1)
            #     pred_3d_joints = smooth_model(pred_3d_joints.cuda())
            #     pred_3d_joints = pred_3d_joints.transpose(2,1)
            #     pred_3d_joints = pred_3d_joints.reshape(B,C,N)
            #     evaluation_accumulators['pred_j3d'].append(pred_3d_joints[-ll:].cpu())
            #     evaluation_accumulators['pred_verts'].append(pred_vertices[-ll:].cpu())
            #     evaluation_accumulators['pred_camera'].append(pred_camera[-ll:].cpu())
            # for k, v in evaluation_accumulators.items():
            #     evaluation_accumulators[k] = torch.vstack(v)
            #     print(evaluation_accumulators[k].shape)
            # evaluation_accumulators['target_j3d'].append(target_j3d)
            # pred_camera, pred_3d_joints, pred_vertices_sub2 = token_[:, 0], token_[:, 1:18], token_[:, 18:]

            # obtain 3d joints from full mesh
            # pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_vertices)
            # gt_3d_joint = gt_smpl_3d_joints.reshape(8, 8, 14, 3)
            # pred_3d_joint = pred_3d_joints_from_smpl.reshape(8, 8, 14, 3)


            # gt_ver = gt_vertices.reshape(8, 8, 6890, 3)
            # pred_ver = pred_vertices.reshape(8, 8, 6890, 3)
            # batch_accel = []
            # for b in range(8):
            #     batch_accel_item = compute_error_accel(gt_3d_joint[b], pred_3d_joint[b])
            #     gt_accel_item = compute_error_accel(pred_3d_joint[b])
            #     batch_accel.append(batch_accel_item)
            #     gt_accel.append(gt_accel_item)
            # batch_accel = np.array(batch_accel)
            # batch_accel_result = batch_accel_result + batch_accel
            # for jj in range(64):
            #     pred_verts.append(pred_vertices[jj].detach().cpu().numpy())
            #     target_theta.append(gt_vertices[jj].detach().cpu().numpy())
            # for ii in range(8):
            #     pred_j3ds.append(pred_3d_joint[ii].detach().cpu().numpy())
            #     target_j3ds.append(gt_3d_joint[ii].detach().cpu().numpy())
            # pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            # pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]

            # # measure errors
            # try:
            #     error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices, has_smpl)
            # except:
            #     print(11)
            # error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints,  has_3d_joints)
            # error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)
            
            # if len(error_vertices)>0:
            #     mPVE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            # if len(error_joints)>0:
            #     mPJPE.update(np.mean(error_joints), int(torch.sum(has_3d_joints)) )
            # if len(error_joints_pa)>0:
            #     PAmPJPE.update(np.mean(error_joints_pa), int(torch.sum(has_3d_joints)))
            # print(len(evaluation_accumulators['pred_j3d']))
            
            # if i > 5:
            #     break
    # aa = np.array(batch_accel)
    # np.mean(aa) * 1000


            name = annotations["name"][0]

            if name.split("/")[0]=="images":
                name = name.split("/")[1]
                name = name.split(".")[0]
            else:
                name = name.split("/")[0]
            print(name)
            for k, v in evaluation_accumulators.items():
                # if k in ["pred_verts"]:
                #     continue
                evaluation_accumulators[k] = torch.vstack(v)
            pred_j3ds = evaluation_accumulators['pred_j3d']
            target_j3ds = evaluation_accumulators['target_j3d']
            denoised_pos = evaluation_accumulators['smooth_j3d']

            # pred_verts = evaluation_accumulators['pred_verts']
            # gt_verts = evaluation_accumulators['gt_verts']

            smooth_3d_joints = denoised_pos.transpose(2,1)
            smooth_3d_joints = smooth_model(smooth_3d_joints.cuda())
            smooth_3d_joints = smooth_3d_joints.transpose(2,1)

            # pred_verts = pred_verts.transpose(2,1)
            # gt_verts = gt_verts.transpose(2,1)

            # pred_verts = evaluation_accumulators['pred_verts']
            denoised_pos = slide_window_to_sequence(smooth_3d_joints.cpu(),1,32)
            # denoised_pos = smooth_3d_joints.cpu()[:,0,:]
            data_pred = slide_window_to_sequence(pred_j3ds,1,32)
            data_gt = slide_window_to_sequence(target_j3ds,1,32)


            frame_num=denoised_pos.shape[0]
            denoised_pos=denoised_pos.reshape(frame_num, -1, 3)
            data_pred=data_pred.reshape(frame_num, -1, 3)
            data_gt=data_gt.reshape(frame_num, -1, 3)

            # print(calculate_accel_error(data_pred, data_gt).mean(),calculate_accel_error(denoised_pos, data_gt).mean(),end="")
         
            keypoint_root = [2, 3]
            denoised_pos = denoised_pos - denoised_pos[:,
                                                          keypoint_root, :].mean(
                                                              axis=1).reshape(
                                                                  -1, 1, 3)
            data_pred = data_pred - data_pred[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)
            data_gt = data_gt - data_gt[:, keypoint_root, :].mean(
                axis=1).reshape(-1, 1, 3)
            # print(calculate_accel_error(data_pred, data_gt).mean(),calculate_accel_error(denoised_pos, data_gt).mean())
         
            # filter_pos,filter_inference_time=filter(data_pred)
            # input_mpve = torch.cat(
            #     (input_mpjpe, calculate_mpjpe(data_pred, data_gt)), dim=0)
            input_mpjpe = torch.cat(
                (input_mpjpe, calculate_mpjpe(data_pred, data_gt)), dim=0)
            input_pampjpe = torch.cat(
                (input_pampjpe, calculate_pampjpe(data_pred, data_gt)), dim=0)
            input_accel = torch.cat(
                (input_accel, calculate_accel_error(data_pred, data_gt)), dim=0)
                
            denoise_mpjpe = torch.cat(
                (denoise_mpjpe, calculate_mpjpe(denoised_pos, data_gt)), dim=0)
            denoise_pampjpe = torch.cat(
                (denoise_pampjpe, calculate_pampjpe(denoised_pos, data_gt)), dim=0)
            denoise_accel = torch.cat(
            (   denoise_accel, calculate_accel_error(denoised_pos, data_gt)), dim=0)

        m2mm = 1000
        print( "input_mpjpe", input_mpjpe.mean() * m2mm,
                "smoothnet_mpjpe", denoise_mpjpe.mean() * m2mm,
   
                "input_pampjpe", input_pampjpe.mean() * m2mm,
                "smoothnet_pampjpe", denoise_pampjpe.mean() * m2mm,
 
                "input_accel", input_accel.mean() * m2mm,
                "smoothnet_accel", denoise_accel.mean() * m2mm
 )
            # inference_time["filter"]+=filter_inference_time


            
            # person_num = 1
            # while True:
            #     for pn in range(person_num):
            #         new_dict = {}
            #         new_dict['pred_j3d'] = pred_j3ds[pn::person_num,:,:]
            #         new_dict['target_j3d'] = target_j3ds[pn::person_num,:,:]
            #         new_dict['pred_verts']= pred_verts[pn::person_num,:,:]
            #         # a = new_dict['pred_j3d']
            #         # b = a.cuda()
            #         # b = b.permute(1, 2, 0)

            #         # new_dict['pred_j3d'] = pred_j3ds[pn::person_num,:,:][:1152]
            #         # new_dict['target_j3d'] = target_j3ds[pn::person_num,:,:][:1152]
            #         # a = smooth_model(b)
            #         acc =calcul(new_dict)
            #         if acc["accel_err"]>100 or acc["accel"] >100 :
            #             person_num += 1
            #             break
            #         else:
            #             print(acc)
            #     else:
            #         break
                        

            # person_dict[name] = person_num
                        

            # pred_3d_joints_from_smpl = smpl.get_h36m_joints(evaluation_accumulators["pred_verts"].cuda())
            # pred_3d_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:]
            # pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_pelvis[:, None, :]
            # pred_2d_joints = orthographic_projection(pred_3d_joints_from_smpl, evaluation_accumulators["pred_camera"].cuda())

            # gt_2d_joints = annotations['joints_2d'].cuda(args.device)
            # B,S,point_num,point_xyz = gt_2d_joints.shape
            # gt_2d_joints = gt_2d_joints.view(B*S,point_num,point_xyz)
            # gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:]

            # import skimage.io as io
            # for i in range(gt_2d_joints.shape[0]):
            #     im = images[i,:,:,:].detach().cpu().numpy()
            #     im = np.transpose(im,[1,2,0])
            #     im = (im-im.min())/(im.max()-im.min())
            #     im = im*255
            #     im = im.astype(np.uint8)
            #     points = gt_2d_joints[i,:,0:2].detach().cpu().numpy()
            #     this_point = points[:,:]
            #     # this_point =  (this_point-this_point.min())/(this_point.max()-this_point.min())
            #     this_point[this_point>1]=1
            #     this_point[this_point<-1]=-1

            #     this_point =  (this_point+1)/(2)
            #     this_point = this_point*224
            #     image1 = im.copy()
            #     for ii in range(0,this_point.shape[0]):
            #     # for i in range(0,100):
            #         # ii +=1
            #         x = round(this_point[ii,0])
            #         y = round(this_point[ii,1])
            #         cv2.circle(image1, (x, y), 3, (255, 0, 0), cv2.FILLED)#绘制关键点
            #     points = pred_2d_joints[i,:,0:2].detach().cpu().numpy()
            #     this_point = points[:,:]
            #     # this_point =  (this_point-this_point.min())/(this_point.max()-this_point.min())
            #     this_point[this_point>1]=1
            #     this_point[this_point<-1]=-1

            #     this_point =  (this_point+1)/(2)
            #     this_point = this_point*224
            #     image2 = im.copy()
            #     for ii in range(0,this_point.shape[0]):
            #     # for i in range(0,100):
            #         # ii +=1
            #         x = round(this_point[ii,0])
            #         y = round(this_point[ii,1])
            #         cv2.circle(image2, (x, y), 3, (255, 0, 0), cv2.FILLED)#绘制关键点
            #     all_image = np.concatenate([im,image1,image2],axis=1).astype(np.uint8)
            #     io.imsave("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test/test_image_{}.png".format("%06d"%i),all_image)
            
            # import ffmpeg
     
            # os.system(f"ffmpeg -r 24 -i /HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test/test_image_%06d.png -vcodec libx264 -vf 'fps=24,format=yuv420p' /HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test.mp4")
            
            # name = annotations["name"][0]

            # if name.split("/")[0]=="images":
            #     name = name.split("/")[1]
            #     name = name.split(".")[0]
            # else:
            #     name = name.split("/")[0]
            # os.rename("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test.mp4","/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/{}.mp4".format(name))
            # os.rename("/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/test","/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/debug_3dpw/{}".format(name))
            
            # acc_all[i] = acc
        # print(person_dict)
    # print(acc_all)
    # # print(batch_accel_result / len(val_dataloader))
    # val_mPVE = all_gather(float(mPVE.avg))
    # val_mPVE = sum(val_mPVE)/len(val_mPVE)
    # val_mPJPE = all_gather(float(mPJPE.avg))
    # val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)

    # val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    # val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)

    # val_count = all_gather(float(mPVE.count))
    # val_count = sum(val_count)

    # return val_mPVE, val_mPJPE, val_PAmPJPE, val_count


def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    B,S,point_num,point_xyz = gt_keypoints_2d.shape
    gt_keypoints_2d = gt_keypoints_2d.view(B*S,point_num,point_xyz)
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    
    B,S,C,H,W = images.shape
    batch_size = B*S
    images = images.view(B*S,C,H,W)



    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_test( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d,
                    PAmPJPE_h36m_j14):
    """Tensorboard logging."""
    B,S,point_num,point_xyz = gt_keypoints_2d.shape
    gt_keypoints_2d = gt_keypoints_2d.view(B*S,point_num,point_xyz)
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]

    B,S,C,H,W = images.shape
    batch_size = B*S
    images = images.view(B*S,C,H,W)


    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        score = PAmPJPE_h36m_j14[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, score)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    # parser.add_argument("--val_yaml", default='/HOME/HOME/data/PointHMR/datasets/human3.6m/valid.protocol2.yaml', type=str, required=False,
    #                     help="Yaml file with all data for validation.")
    parser.add_argument("--val_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/test_has_gender.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    # parser.add_argument("--val_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/train.yaml', type=str, required=False,
    #                     help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=2, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    # parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
    #                     help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default="/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/output_3dpw_step8_ema_back_S_adapt_loss/checkpoint-142-24992/state_dict.bin", type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--resume_op_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output_step2/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default='sine', type=str)    
    parser.add_argument("--use_smpl_param_regressor", default=False, action='store_true',) 
    # CNN backbone
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                        help='CNN backbone architecture: hrnet-w64, resnet50')
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-5, type=float, 
                        help="The initial lr.")
    parser.add_argument("--model_name", default='FastMETRO-L', type=str,
                        help='Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L')
    parser.add_argument("--num_train_epochs", default=500, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=100.0, type=float)          
    parser.add_argument("--joints_loss_weight", default=1000.0, type=float)
    parser.add_argument("--heatmap_loss_weight", default=10.0, type=float)


    parser.add_argument("--vloss_w_full", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub", default=0.33, type=float) 
    parser.add_argument("--vloss_w_sub2", default=0.33, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    # parser.add_argument("--transformer_nhead", default=4, type=int, required=False,
    #                     help="Update model config if given. Note that the division of "
    #                          "hidden_size / num_attention_heads should be in integer.")
    # parser.add_argument("--model_dim", default=512, type=int,
    #                     help="The Image Feature Dimension.")
    # parser.add_argument("--feedforward_dim_1", default=1024, type=int,
    #                     help="The Image Feature Dimension.")
    # parser.add_argument("--feedforward_dim_2", default=512, type=int,
    #                     help="The Image Feature Dimension.")
    # parser.add_argument("--position_dim", default=128, type=int,
    #                     help="position dim.")
    parser.add_argument("--activation", default="relu", type=str,
                        help="The Image Feature Dimension.")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="The Image Feature Dimension.")
    parser.add_argument("--mesh_type", default='body', type=str, help="body or hand") 
    parser.add_argument("--interm_size_scale", default=2, type=int)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=True, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")


    args = parser.parse_args()
    return args


def main(args):
    # USE_MULTI_GPU = True
    # import os 
    # # 检测机器是否有多张显卡
    # if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    #     MULTI_GPU = True
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    #     device_ids = [0, 1]
    # else:
    #     MULTI_GPU = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    os.environ['OMP_NUM_THREADS'] = str(args.num_workers)
    print('set os.environ[OMP_NUM_THREADS] to {}'.format(os.environ['OMP_NUM_THREADS']))

    # args.distributed = args.num_gpus > 1
    args.distributed = False
    args.device = torch.device(args.device)
    if args.distributed:
        # print("Init distributed training on local rank {} ({}), rank {}, world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), int(os.environ["NODE_RANK"]), args.num_gpus))
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("Graphormer", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL()
    # smpl = torch.nn.DataParallel(smpl,device_ids=device_ids)
    smpl.to(device)
    mesh_sampler = Mesh()
    # mesh_sampler = torch.nn.DataParallel(mesh_sampler,device_ids=device_ids)
    # mesh_sampler.to(device)

    # Renderer for visualization
    renderer = Renderer(faces=smpl.faces.cpu().numpy())



    hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
    logger.info('=> loading hrnet-v2-w64 model')
    _FastMETRO_Network = FastMETRO_Network(args, backbone, mesh_sampler)
    if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
        # for fine-tuning or resume training or inference, load weights from checkpoint
        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        # workaround approach to load sparse tensor in graph conv.
        states = torch.load(args.resume_checkpoint, map_location=args.device)


        for k, v in states.items():
            states[k] = v.cpu()
        _FastMETRO_Network.load_state_dict(states, strict=False)

        del states
        gc.collect()
        torch.cuda.empty_cache()
    

    # for name,para in _FastMETRO_Network.named_parameters():
    #     # with torch.no_grad():
    #     para = para + (torch.rand(para.size())-0.5) * 0.15 * torch.std(para)
    # _FastMETRO_Network = torch.nn.DataParallel(_FastMETRO_Network,device_ids=device_ids)
    _FastMETRO_Network.to(args.device)
    logger.info("Training parameters %s", args)

    if args.run_eval_only==True:
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run_eval_general(args, val_dataloader, _FastMETRO_Network, smpl, mesh_sampler)

    else:
        train_dataloader = make_data_loader(args, args.train_yaml, 
                                            args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        val_dataloader = make_data_loader(args, args.val_yaml, 
                                        args.distributed, is_train=False, scale_factor=args.img_scale_factor)
        run(args, train_dataloader, val_dataloader, _FastMETRO_Network, mesh_sampler, smpl, renderer)



if __name__ == "__main__":
    args = parse_args()
    main(args)
