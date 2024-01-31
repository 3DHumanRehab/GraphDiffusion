# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------
# Modified from METRO (https://github.com/microsoft/MeshTransformer)
# Copyright (c) Microsoft Corporation. All Rights Reserved [see https://github.com/microsoft/MeshTransformer/blob/main/LICENSE for details]
# ----------------------------------------------------------------------------------------------
"""
Training and evaluation codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import gc
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
import sys 
from torch.nn import functional as F
# sys.path.remove('/HOME/HOME/fzh')
sys.path.append("/root/fengzehui/")
#sys.path.append("/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt")
import matplotlib.pyplot as plt
from src.tools.loss_change import calc_loss_SingleMesh_2D3DVelocity_changeWeight_no_heatmap,mean_velocity_error_train_2
from src.modeling.model.modeling_fastmetro_EMA_4_S_adapt_copy import FastMETRO_Body_Network as FastMETRO_Network
from src.modeling._smpl import SMPL, Mesh
from src.modeling.model.hrnet_cls_net_featmaps_adapt_copy import get_cls_net
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build_more_data import make_data_loader
from src.utils.logger import setup_logger
from src.utils.comm import is_main_process, get_rank, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.metric_pampjpe import reconstruction_error
from src.utils.geometric_layers import orthographic_projection, rodrigues
from src.utils.renderer_opendr import OpenDR_Renderer, visualize_reconstruction_opendr, visualize_reconstruction_multi_view_opendr, visualize_reconstruction_smpl_opendr
try:
    from src.utils.renderer_pyrender import PyRender_Renderer, visualize_reconstruction_pyrender, visualize_reconstruction_multi_view_pyrender, visualize_reconstruction_smpl_pyrender
except:
    print("Failed to import renderer_pyrender. Please see docs/Installation.md")
#from azureml.core.run import Run
#aml_run = Run.get_context()

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam
def make_all_heatmap(all_image,backbone_output,SD_output):
    now=datetime.datetime.now()
    now = now.strftime("%Y-%m-%H:%M:%S")
    #first_featmap_list,ema_out_list,ema_S_out_list,final_list = [],[],[],[]
    SD_featmap_list , backbone_output_list,sample_1_list,sample_2_list = [], [], [], []
    for ff in range(all_image.shape[0]):
        
        # b = all_image[ff,0,:,:]

        # b = b.detach().cpu().numpy()
        # #b = np.sum(b, axis=0)
        # b = np.reshape(b, (1, b.shape[0], b.shape[1]))       
        # feature = np.transpose(b,[1,2,0])
        b = all_image[ff,:,:,:]
        b = b.detach().cpu().numpy()
        feature = np.transpose(b,[1,2,0])
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        # cv2.cvtColor(feature*255 , cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/heatmap/{}_{}_{}.png'.format(now,'%03d'%ff,"image"),cv2.cvtColor(feature*255 , cv2.COLOR_RGB2BGR))

        # 假设我们只关注第一个特征图
        original_image = feature
        #first_featmap_list.append(make_this_heatmap(ff,now,all_image,first_featmap,"first_featmap"))
        #ema_out_list.append(make_this_heatmap(ff,now,all_image,ema_out,"ema_out"))
        #ema_S_out_list.append(make_this_heatmap(ff,now,all_image,ema_S_out,"ema_S_out"))
        #final_list.append(make_this_heatmap(ff,now,all_image,x,"final"))
        SD_featmap_list.append(make_this_heatmap(ff,now,all_image,SD_output,"SD_featmap"))
        backbone_output_list.append(make_this_heatmap(ff,now,all_image,backbone_output,"backbone_output"))
        #sample_1_list.append(make_this_heatmap(ff,now,all_image,sample_1,"sample_1"))
        #sample_2_list.append(make_this_heatmap(ff,now,all_image,sample_2,"sample_2"))
                

    #return first_featmap_list,ema_out_list,ema_S_out_list,final_list
    return SD_featmap_list, backbone_output_list



def make_this_heatmap(ff,now,all_image,featmap,name):
        input_image = all_image

        feature = featmap[ff, :, :, :].detach().cpu().numpy()
        feature = np.mean(feature,axis=0)
        feature = feature

        # 规范化特征图
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)

        # 应用热力图颜色映射
        feature_heatmap = plt.get_cmap('jet')(feature)[:, :, :3]

        # 3. 调整热力图尺寸
        feature_heatmap = cv2.resize(feature_heatmap, (input_image.shape[2], input_image.shape[3]))

        # 4. 叠加热力图和原图
        # 假设 original_image 是原始图像，已经被规范化到 [0, 1]
        b = all_image[ff,:,:,:]
        b = b.detach().cpu().numpy()
        feature = np.transpose(b,[1,2,0])
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        feature = (feature * 255).astype(np.uint8)
        feature_heatmap = (feature_heatmap * 255).astype(np.uint8)

        overlay_image = cv2.addWeighted(feature, 1,feature_heatmap , 0.5, 0)
        return overlay_image
        # return cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/heatmap/{}_{}_{}.png'.format(now,'%03d'%ff,name), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

def make_this_heatmap2(ff,now,all_image,featmap,featmap2,name):
        input_image = all_image

        feature = featmap[ff, :, :, :].detach().cpu().numpy()
        feature = np.mean(feature,axis=0)
        feature = feature

        # 规范化特征图
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)

        # 应用热力图颜色映射
        feature_heatmap = plt.get_cmap('jet')(feature)[:, :, :3]

        # 3. 调整热力图尺寸
        feature_heatmap = cv2.resize(feature_heatmap, (input_image.shape[2], input_image.shape[3]))

        # 4. 叠加热力图和原图
        # 假设 original_image 是原始图像，已经被规范化到 [0, 1]
        b = all_image[ff,:,:,:]
        b = b.detach().cpu().numpy()
        feature = np.transpose(b,[1,2,0])
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        feature = (feature * 255).astype(np.uint8)
        feature_heatmap = (feature_heatmap * 255).astype(np.uint8)

        overlay_image = cv2.addWeighted(feature, 1,feature_heatmap , 0.5, 0)
        
        input_image = all_image

        feature = featmap2[ff, :, :, :].detach().cpu().numpy()
        feature = np.mean(feature,axis=0)
        feature = feature

        # 规范化特征图
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)

        # 应用热力图颜色映射
        feature_heatmap = plt.get_cmap('jet')(feature)[:, :, :3]

        # 3. 调整热力图尺寸
        feature_heatmap = cv2.resize(feature_heatmap, (input_image.shape[2], input_image.shape[3]))

        # 4. 叠加热力图和原图
        # 假设 original_image 是原始图像，已经被规范化到 [0, 1]
        b = all_image[ff,:,:,:]
        b = b.detach().cpu().numpy()
        feature = np.transpose(b,[1,2,0])
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        feature = (feature * 255).astype(np.uint8)
        feature_heatmap = (feature_heatmap * 255).astype(np.uint8)

        overlay_image = cv2.subtract(overlay_image,feature_heatmap)


        return overlay_image
        # return cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/HOME/HOME/Zhongzhangnan/PointHMR_PIPS_test_acc/heatmap/{}_{}_{}.png'.format(now,'%03d'%ff,name), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for _ in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def save_scores(args, split, mPJPE, PAmPJPE, mPVPE):
    eval_log = []
    res = {}
    res['MPJPE'] = mPJPE
    res['PA-MPJPE'] = PAmPJPE
    res['MPVPE'] = mPVPE
    eval_log.append(res)
    with open(op.join(args.output_dir, split+'_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return

def mean_per_joint_position_error(pred, gt, has_3d_joints):
    """ 
    Compute MPJPE
    """
    gt = gt[has_3d_joints == 1]
    gt = gt[:, :, :-1]
    pred = pred[has_3d_joints == 1]
    with torch.no_grad():
        gt_pelvis = (gt[:, 2,:] + gt[:, 3,:]) / 2
        gt = gt - gt_pelvis[:, None, :]
        pred_pelvis = (pred[:, 2,:] + pred[:, 3,:]) / 2
        pred = pred - pred_pelvis[:, None, :]
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_position_error(pred, gt, has_smpl):
    """
    Compute MPVPE
    """
    pred = pred[has_smpl == 1]
    gt = gt[has_smpl == 1]
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    # shape of gt_keypoints_2d: batch_size X 14 X 3 (last for visibility)
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    # shape of gt_keypoints_3d: Batch_size X 14 X 4 (last for confidence)
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices_with_shape = pred_vertices[has_smpl == 1]
    gt_vertices_with_shape = gt_vertices[has_smpl == 1]
    if len(gt_vertices_with_shape) > 0:
        return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) 

def smpl_param_loss(criterion_smpl, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl, device):
    """
    Compute smpl parameter loss if smpl annotations are available.
    """
    pred_rotmat_with_shape = pred_rotmat[has_smpl == 1].view(-1, 3, 3)
    pred_betas_with_shape = pred_betas[has_smpl == 1]
    gt_rotmat_with_shape = rodrigues(gt_pose[has_smpl == 1].view(-1, 3))
    gt_betas_with_shape = gt_betas[has_smpl == 1]
    if len(gt_rotmat_with_shape) > 0:
        loss = criterion_smpl(pred_rotmat_with_shape, gt_rotmat_with_shape) + 0.1 * criterion_smpl(pred_betas_with_shape, gt_betas_with_shape)
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(device) 
    return loss


class EdgeLengthGTLoss(torch.nn.Module):
    """
    Modified from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/nets/loss.py)
    """
    def __init__(self, face, edge, uniform=True):
        super().__init__()
        self.face = face # num_faces X 3
        self.edge = edge # num_edges X 2
        self.uniform = uniform

    def forward(self, pred_vertices, gt_vertices, has_smpl, device):
        face = self.face
        edge = self.edge
        coord_out = pred_vertices[has_smpl == 1]
        coord_gt = gt_vertices[has_smpl == 1]
        if len(coord_gt) > 0:
            if self.uniform:
                d1_out = torch.sqrt(torch.sum((coord_out[:,edge[:,0],:] - coord_out[:,edge[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_edges X 1
                d1_gt = torch.sqrt(torch.sum((coord_gt[:,edge[:,0],:] - coord_gt[:,edge[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_edges X 1
                edge_diff = torch.abs(d1_out - d1_gt)
            else:
                d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

                d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1

                diff1 = torch.abs(d1_out - d1_gt)
                diff2 = torch.abs(d2_out - d2_gt) 
                diff3 = torch.abs(d3_out - d3_gt) 
                edge_diff = torch.cat((diff1, diff2, diff3),1)
            loss = edge_diff.mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device) 

        return loss


class EdgeLengthSelfLoss(torch.nn.Module):
    def __init__(self, face, edge, uniform=True):
        super().__init__()
        self.face = face # num_faces X 3
        self.edge = edge # num_edges X 2
        self.uniform = uniform

    def forward(self, pred_vertices, has_smpl, device):
        face = self.face
        edge = self.edge
        coord_out = pred_vertices[has_smpl == 1]
        if len(coord_out) > 0:
            if self.uniform:
                edge_self_diff = torch.sqrt(torch.sum((coord_out[:,edge[:,0],:] - coord_out[:,edge[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_edges X 1
            else:
                d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True)+1e-8) # batch_size X num_faces X 1
                edge_self_diff = torch.cat((d1_out, d2_out, d3_out),1)
            loss = torch.mean(edge_self_diff)
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device) 

        return loss


class NormalVectorLoss(torch.nn.Module):
    """
    Modified from [I2L-MeshNet](https://github.com/mks0601/I2L-MeshNet_RELEASE/blob/master/common/nets/loss.py)
    """
    def __init__(self, face):
        super().__init__()
        self.face = face # num_faces X 3

    def forward(self, pred_vertices, gt_vertices, has_smpl, device):
        face = self.face
        coord_out = pred_vertices[has_smpl == 1]
        coord_gt = gt_vertices[has_smpl == 1]
        if len(coord_gt) > 0:
            v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
            v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:] # batch_size X num_faces X 3
            v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
            v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:] # batch_size X num_faces X 3
            v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

            v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
            v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:] # batch_size X num_faces X 3
            v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
            normal_gt = torch.cross(v1_gt, v2_gt, dim=2) # batch_size X num_faces X 3
            normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

            cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) # batch_size X num_faces X 1
            loss = torch.cat((cos1, cos2, cos3),1).mean()
        else:
            loss = torch.FloatTensor(1).fill_(0.).to(device) 

        return loss

def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def run_train(args, train_dataloader, val_dataloader, FastMETRO_model, smpl, mesh_sampler, renderer, smpl_intermediate_faces, smpl_intermediate_edges):    
    smpl.eval()
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    args.logging_steps = iters_per_epoch // 2
    iteration = args.resume_epoch * iters_per_epoch

    FastMETRO_model_without_ddp = FastMETRO_model
    if args.distributed:
        FastMETRO_model = torch.nn.parallel.DistributedDataParallel(
            FastMETRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
        FastMETRO_model_without_ddp = FastMETRO_model.module
        if is_main_process():
            logger.info(
                    ' '.join(
                    ['Local-Rank: {o}', 'Max-Iteration: {a}', 'Iterations-per-Epoch: {b}','Number-of-Training-Epochs: {c}',]
                    ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
                )

    param_dicts = [
        {"params": [p for p in FastMETRO_model_without_ddp.parameters() if p.requires_grad]}
    ]

    # optimizer & learning rate scheduler
    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(params=list(FastMETRO_model.parameters()),
                                 lr=args.lr,
                            )
    device_ids = [0, 1]
    optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer.module, args.lr_drop)
    lr_scheduler = torch.nn.DataParallel(lr_scheduler, device_ids=device_ids)
    # define loss functions for joints & vertices
    criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)
    criterion_3d_keypoints = torch.nn.L1Loss(reduction='none').cuda(args.device)
    criterion_3d_vertices = torch.nn.L1Loss().cuda(args.device)
    if args.use_smpl_param_regressor:
        criterion_smpl_param = torch.nn.MSELoss().cuda(args.device)
    
    # define loss functions for edge length & normal vector
    edge_gt_loss = EdgeLengthGTLoss(smpl_intermediate_faces, smpl_intermediate_edges, uniform=args.uniform)
    edge_self_loss = EdgeLengthSelfLoss(smpl_intermediate_faces, smpl_intermediate_edges, uniform=args.uniform)
    normal_loss = NormalVectorLoss(smpl_intermediate_faces)

    start_training_time = time.time()
    end = time.time()
    FastMETRO_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_3d_joints = AverageMeter()
    log_loss_3d_vertices = AverageMeter()
    log_loss_edge_normal = AverageMeter()
    log_loss_2d_joints = AverageMeter()
    log_eval_metrics_mpjpe = EvalMetricsLogger()
    log_eval_metrics_pampjpe = EvalMetricsLogger()
    if args.resume_epoch > 0:
        log_eval_metrics_mpjpe.set(epoch=args.resume_mpjpe_best_epoch, mPVPE=args.resume_mpjpe_best_mpvpe, mPJPE=args.resume_mpjpe_best_mpjpe, PAmPJPE=args.resume_mpjpe_best_pampjpe)
        log_eval_metrics_pampjpe.set(epoch=args.resume_pampjpe_best_epoch, mPVPE=args.resume_pampjpe_best_mpvpe, mPJPE=args.resume_pampjpe_best_mpjpe, PAmPJPE=args.resume_pampjpe_best_pampjpe)
    print("train_data_num:",train_dataloader.__len__())
    for _, (img_keys, images, annotations) in enumerate(train_dataloader):
        FastMETRO_model.train()
        iteration = iteration + 1
        epoch = iteration // iters_per_epoch
        batch_size = images.size(0)

        data_time.update(time.time() - end)

        images = images.cuda(args.device) # batch_size X 3 X 224 X 224 
        B,S,C,H,W = images.shape
        batch_size = B*S
        images = images.view(B*S,C,H,W)


        # gt 2d joints
        gt_2d_joints = annotations['joints_2d'].cuda(args.device) # batch_size X 24 X 3 (last for visibility)
        B,S,point_num,point_xyz = gt_2d_joints.shape
        gt_2d_joints = gt_2d_joints.view(B*S,point_num,point_xyz)
        gt_2d_joints = gt_2d_joints[:,cfg.J24_TO_J14,:] # batch_size X 14 X 3

        has_2d_joints = annotations['has_2d_joints'].cuda(args.device) # batch_size

        # gt 3d joints
        gt_3d_joints = annotations['joints_3d'].cuda(args.device) # batch_size X 24 X 4 (last for confidence)
        B,S,point_num,point_xyz = gt_3d_joints.shape
        gt_3d_joints = gt_3d_joints.view(B*S,point_num,point_xyz)
        gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3] # batch_size X 3
        gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] # batch_size X 14 X 4
        gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :] # batch_size X 14 X 4
        has_3d_joints = annotations['has_3d_joints'].cuda(args.device) # batch_size

        # gt params for smpl
        gt_pose = annotations['pose'].cuda(args.device) # batch_size X 72
        gt_betas = annotations['betas'].cuda(args.device) # batch_size X 10
        has_smpl = annotations['has_smpl'].cuda(args.device) # batch_size 


        gt_pose = gt_pose.view(B*S,-1)
        gt_betas = gt_betas.view(B*S,-1) 
        has_smpl = has_smpl.view(B*S) 
        has_3d_joints = has_3d_joints.view(B*S)
        # generate simplified mesh
        gt_3d_vertices_fine = smpl(gt_pose, gt_betas) # batch_size X 6890 X 3
        gt_3d_vertices_intermediate = mesh_sampler.module.downsample(gt_3d_vertices_fine, n1=0, n2=1) # batch_size X 1723 X 3
        gt_3d_vertices_coarse = mesh_sampler.module.downsample(gt_3d_vertices_intermediate, n1=1, n2=2) # batch_size X 431 X 3

        # normalize ground-truth vertices & joints (based on smpl's pelvis)
        # smpl.get_h36m_joints: from vertex to joint (using smpl)
        gt_smpl_3d_joints = smpl.module.get_h36m_joints(gt_3d_vertices_fine) # batch_size X 17 X 3
        gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:] # batch_size X 3
        gt_3d_vertices_fine = gt_3d_vertices_fine - gt_smpl_3d_pelvis[:, None, :] # batch_size X 6890 X 3
        gt_3d_vertices_intermediate = gt_3d_vertices_intermediate - gt_smpl_3d_pelvis[:, None, :] # batch_size X 1723 X 3
        gt_3d_vertices_coarse = gt_3d_vertices_coarse - gt_smpl_3d_pelvis[:, None, :] # batch_size X 431 X 3

        # forward-pass
        out = FastMETRO_model(images)
        pred_cam, pred_3d_joints_from_token = out['pred_cam'], out['pred_3d_joints']
        pred_3d_vertices_coarse, pred_3d_vertices_intermediate, pred_3d_vertices_fine = out['pred_3d_vertices_coarse'], out['pred_3d_vertices_intermediate'], out['pred_3d_vertices_fine']

        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_3d_vertices_fine) # batch_size X 17 X 3
        pred_3d_joints_from_smpl_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:] # batch_size X 14 X 3
        # normalize predicted vertices 
        pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 6890 X 3
        pred_3d_vertices_intermediate = pred_3d_vertices_intermediate - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 1723 X 3
        pred_3d_vertices_coarse = pred_3d_vertices_coarse - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 431 X 3
        # normalize predicted joints 
        pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 14 X 3
        pred_3d_joints_from_token_pelvis = (pred_3d_joints_from_token[:,2,:] + pred_3d_joints_from_token[:,3,:]) / 2
        pred_3d_joints_from_token = pred_3d_joints_from_token - pred_3d_joints_from_token_pelvis[:, None, :] # batch_size X 14 X 3
        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_cam) # batch_size X 14 X 2
        pred_2d_joints_from_token = orthographic_projection(pred_3d_joints_from_token, pred_cam) # batch_size X 14 X 2

        # compute 3d joint loss
        loss_3d_mean_velocity = mean_velocity_error_train_2(pred_3d_joints_from_token,gt_3d_joints[:, :, :-1].clone())+mean_velocity_error_train_2(pred_3d_joints_from_smpl,gt_3d_joints[:, :, :-1].clone())


        loss_3d_joints = (keypoint_3d_loss(criterion_3d_keypoints, pred_3d_joints_from_token, gt_3d_joints, has_3d_joints, args.device) + \
                         keypoint_3d_loss(criterion_3d_keypoints, pred_3d_joints_from_smpl, gt_3d_joints, has_3d_joints, args.device))
        loss_3d_joints = loss_3d_joints+loss_3d_mean_velocity
        # compute 3d vertex loss
        loss_3d_vertices_coarse_mean_velocity = mean_velocity_error_train_2(pred_3d_vertices_coarse,gt_3d_vertices_coarse.clone())
        loss_3d_vertices_intermediate_mean_velocity = mean_velocity_error_train_2(pred_3d_vertices_intermediate,gt_3d_vertices_intermediate.clone())
        loss_3d_vertices_fine_mean_velocity = mean_velocity_error_train_2(pred_3d_vertices_fine,gt_3d_vertices_fine.clone())

        loss_3d_vertices = (args.vertices_coarse_loss_weight * (vertices_loss(criterion_3d_vertices, pred_3d_vertices_coarse, gt_3d_vertices_coarse, has_smpl, args.device) +loss_3d_vertices_coarse_mean_velocity)+ \
                           args.vertices_intermediate_loss_weight * (vertices_loss(criterion_3d_vertices, pred_3d_vertices_intermediate, gt_3d_vertices_intermediate, has_smpl, args.device) +loss_3d_vertices_intermediate_mean_velocity)+ \
                           args.vertices_fine_loss_weight * (vertices_loss(criterion_3d_vertices, pred_3d_vertices_fine, gt_3d_vertices_fine, has_smpl, args.device)+loss_3d_vertices_fine_mean_velocity))
        # compute edge length loss (GT supervision & self regularization) & normal vector loss
        loss_edge_normal = (args.edge_gt_loss_weight * edge_gt_loss(pred_3d_vertices_intermediate, gt_3d_vertices_intermediate, has_smpl, args.device) + \
                           args.edge_self_loss_weight * edge_self_loss(pred_3d_vertices_intermediate, has_smpl, args.device) + \
                           args.normal_loss_weight * normal_loss(pred_3d_vertices_intermediate, gt_3d_vertices_intermediate, has_smpl, args.device))
        # compute 2d joint loss

        loss_2d_mean_velocity = mean_velocity_error_train_2(pred_2d_joints_from_token,gt_2d_joints[:, :, :-1].clone())+mean_velocity_error_train_2(pred_2d_joints_from_smpl,gt_2d_joints[:, :, :-1].clone())
        loss_2d_joints = (keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_token, gt_2d_joints, has_2d_joints) + \
                         keypoint_2d_loss(criterion_2d_keypoints, pred_2d_joints_from_smpl, gt_2d_joints, has_2d_joints))
        loss_2d_joints = loss_2d_joints+loss_2d_mean_velocity
        # empirically set hyperparameters to balance different losses
        loss = (args.joints_3d_loss_weight * loss_3d_joints + \
               args.vertices_3d_loss_weight * loss_3d_vertices + \
               args.edge_normal_loss_weight * loss_edge_normal + \
               args.joints_2d_loss_weight * loss_2d_joints)
        '''new loss'''

        # loss = calc_loss_SingleMesh_2D3DVelocity_changeWeight_no_heatmap(args,
        #         pred_cam,
        #         pred_3d_joints_from_token,
        #         gt_3d_joints,
        #         gt_2d_joints,
        #         gt_3d_joints,
        #         gt_2d_joints,
        #         smpl,
        #         weight_3d = 1000,
        #         weight_2d = 100,
        #         weight_vertices = 100)
        # loss = calc_loss_SingleMesh_2D3DVelocity_changeWeight_no_heatmap(args,
        #         pred_cam,
        #         pred_3d_joints_from_token,
        #         pred_3d_vertices_fine,
        #         gt_3d_vertices_fine,
        #         gt_3d_joints,
        #         gt_2d_joints,
        #         smpl,
        #         weight_3d = 1000,
        #         weight_2d = 100,
        #         weight_vertices = 100)
        '''end new'''
        if args.use_smpl_param_regressor:
            pred_rotmat, pred_betas = out['pred_rotmat'], out['pred_betas']
            pred_smpl_3d_vertices = smpl(pred_rotmat, pred_betas) # batch_size X 6890 X 3
            pred_smpl_3d_joints = smpl.module.get_h36m_joints(pred_smpl_3d_vertices) # batch_size X 17 X 3
            pred_smpl_3d_joints_pelvis = pred_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_smpl_3d_joints = pred_smpl_3d_joints[:,cfg.H36M_J17_TO_J14,:] # batch_size X 14 X 3
            pred_smpl_3d_vertices = pred_smpl_3d_vertices - pred_smpl_3d_joints_pelvis[:, None, :] # batch_size X 6890 X 3
            pred_smpl_3d_joints = pred_smpl_3d_joints - pred_smpl_3d_joints_pelvis[:, None, :] # batch_size X 14 X 3
            pred_smpl_2d_joints = orthographic_projection(pred_smpl_3d_joints, pred_cam.clone().detach()) # batch_size X 14 X 2
            loss_smpl_3d_joints = keypoint_3d_loss(criterion_3d_keypoints, pred_smpl_3d_joints, gt_3d_joints, has_3d_joints, args.device)
            loss_smpl_2d_joints = keypoint_2d_loss(criterion_2d_keypoints, pred_smpl_2d_joints, gt_2d_joints, has_2d_joints)
            loss_smpl_vertices = vertices_loss(criterion_3d_vertices, pred_smpl_3d_vertices, gt_3d_vertices_fine, has_smpl, args.device)
            # compute smpl parameter loss
            loss_smpl = smpl_param_loss(criterion_smpl_param, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl, args.device) + loss_smpl_3d_joints + loss_smpl_2d_joints + loss_smpl_vertices
            if "H36m" in args.train_yaml:
                # mainly train smpl parameter regressor in h36m training
                loss = (args.smpl_param_loss_weight * loss_smpl) + (1e-8 * loss)
            else:
                # train both in 3dpw fine-tuning
                loss = (args.smpl_param_loss_weight * loss_smpl) + loss

        # update logs
        log_loss_3d_joints.update(loss_3d_joints.item(), batch_size)
        log_loss_3d_vertices.update(loss_3d_vertices.item(), batch_size)
        log_loss_edge_normal.update(loss_edge_normal.item(), batch_size)
        log_loss_2d_joints.update(loss_2d_joints.item(), batch_size)
        log_losses.update(loss.item(), batch_size)

        # back-propagation
        optimizer.zero_grad()
        loss.backward() 
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(FastMETRO_model.parameters(), args.clip_max_norm)
        optimizer.module.step()
        batch_time.update(time.time() - end)
        end = time.time()
        
        
        # logging
        # if (iteration % (iters_per_epoch//2) == 0) and is_main_process():
        #     print("Complete", iteration, "th iterations!")
        # if ((iteration == 10) or (iteration == 100) or ((iteration % args.logging_steps) == 0) or (iteration == max_iter)) and is_main_process():
        #     logger.info(
        #         ' '.join(
        #         ['epoch: {ep}', 'iterations: {iter}',]
        #         ).format(ep=epoch, iter=iteration,) 
        #         + ' loss: {:.4f}, 3D-joint-loss: {:.4f}, 3D-vertex-loss: {:.4f}, edge-normal-loss: {:.4f}, 2D-joint-loss: {:.4f}, lr: {:.6f}'.format(
        #             log_losses.avg, log_loss_3d_joints.avg, log_loss_3d_vertices.avg, log_loss_2d_joints.avg,
        #             optimizer.module.param_groups[0]['lr'])
        #     )
        if iteration % args.logging_steps == 0 or iteration == max_iter:
        # if True:
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ' '.join(
                ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, ep=epoch, iter=iteration, 
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '  loss: {:.4f}, 2d joint loss: {:.4f}, 3d joint loss: {:.4f}, vertex loss: {:.4f}, edge-normal-loss: {:.4f}, compute: {:.4f}, data: {:.4f}, lr: {:.6f}'.format(
                    log_losses.avg, log_loss_2d_joints.avg, log_loss_3d_joints.avg, log_loss_3d_vertices.avg, log_loss_edge_normal.avg, batch_time.avg, data_time.avg, 
                    optimizer.module.param_groups[0]['lr'])
            )

            #aml_run.log(name='Loss', value=float(log_losses.avg))
            #aml_run.log(name='3d joint Loss', value=float(log_loss_3d_joints.avg))
            #aml_run.log(name='2d joint Loss', value=float(log_loss_2d_joints.avg))
            #aml_run.log(name='vertex Loss', value=float(log_loss_3d_vertices.avg))
            # visualize estimation results during training
            if args.visualize_training and (iteration >= args.logging_steps):
                visual_imgs = visualize_mesh(renderer,
                                            annotations['ori_img'].detach(),
                                            pred_3d_vertices_fine.detach(), 
                                            pred_cam.detach())
                visual_imgs = visual_imgs.transpose(0,1)
                visual_imgs = visual_imgs.transpose(1,2)
                visual_imgs = np.asarray(visual_imgs)
                if is_main_process():
                    stamp = str(epoch) + '_' + str(iteration)
                    temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                    if args.use_opendr_renderer:
                        visual_imgs[:,:,::-1] = visual_imgs[:,:,::-1]*255
                    cv2.imwrite(temp_fname, np.asarray(visual_imgs[:,:,::-1]))

        # save checkpoint
        if (epoch != 0) and ((epoch % args.saving_epochs) == 0) and ((iteration % iters_per_epoch) == 0):
            checkpoint_dir = save_checkpoint(FastMETRO_model, args, 0, 0)
        if (iteration % iters_per_epoch) == 0:
            lr_scheduler.module.step()
            
            val_result = run_validate(args, val_dataloader, FastMETRO_model, epoch, smpl, mesh_sampler)
            val_mPVPE, val_mPJPE, val_PAmPJPE = val_result['val_mPVPE'], val_result['val_mPJPE'], val_result['val_PAmPJPE']
            if args.use_smpl_param_regressor:
                val_smpl_mPVPE, val_smpl_mPJPE, val_smpl_PAmPJPE = val_result['val_smpl_mPVPE'], val_result['val_smpl_mPJPE'], val_result['val_smpl_PAmPJPE']
            
            if val_PAmPJPE < log_eval_metrics_pampjpe.PAmPJPE:
                checkpoint_dir = save_checkpoint(FastMETRO_model, args, epoch, iteration)
                log_eval_metrics_pampjpe.update(val_mPVPE, val_mPJPE, val_PAmPJPE, epoch)
            if val_mPJPE < log_eval_metrics_mpjpe.mPJPE:
                checkpoint_dir = save_checkpoint(FastMETRO_model, args, epoch, iteration)
                log_eval_metrics_mpjpe.update(val_mPVPE, val_mPJPE, val_PAmPJPE, epoch)

            # if is_main_process():
            if True:
                if args.use_smpl_param_regressor:
                    logger.info(
                        ' '.join(['Validation', 'Epoch: {ep}',]).format(ep=epoch) 
                        + '  MPVPE: {:6.2f}, MPJPE: {:6.2f}, PA-MPJPE: {:6.2f} '.format(1000*val_mPVPE, 1000*val_mPJPE, 1000*val_PAmPJPE)
                        + '  <SMPL> MPVPE: {:6.2f}, <SMPL> MPJPE: {:6.2f}, <SMPL> PA-MPJPE: {:6.2f} '.format(1000*val_smpl_mPVPE, 1000*val_smpl_mPJPE, 1000*val_smpl_PAmPJPE)
                        )
                else:
                    logger.info(
                        ' '.join(['Validation', 'Epoch: {ep}',]).format(ep=epoch) 
                        + ' MPVPE: {:6.2f}, MPJPE: {:6.2f}, PA-MPJPE: {:6.2f}'.format(1000*val_mPVPE, 1000*val_mPJPE, 1000*val_PAmPJPE)
                    )
                logger.info(
                        'Best Results (PA-MPJPE):'
                        + ' <MPVPE> {:6.2f} <MPJPE> {:6.2f}, <PA-MPJPE> {:6.2f}, at Epoch {:6.2f}'.format(1000*log_eval_metrics_pampjpe.mPVPE, 1000*log_eval_metrics_pampjpe.mPJPE, 1000*log_eval_metrics_pampjpe.PAmPJPE, log_eval_metrics_pampjpe.epoch)
                )
                logger.info(
                        'Best Results (MPJPE):'
                        + ' <MPVPE> {:6.2f}, <MPJPE> {:6.2f}, <PA-MPJPE> {:6.2f}, at Epoch {:6.2f}'.format(1000*log_eval_metrics_mpjpe.mPVPE, 1000*log_eval_metrics_mpjpe.mPJPE, 1000*log_eval_metrics_mpjpe.PAmPJPE, log_eval_metrics_mpjpe.epoch)
                )  
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(total_time_str, total_training_time / max_iter))
    # checkpoint_dir = save_checkpoint(FastMETRO_model, args, epoch, iteration)

    logger.info(
        'Best Results: (PA-MPJPE)'
        + ' <MPVPE> {:6.2f} <MPJPE> {:6.2f}, <PA-MPJPE> {:6.2f} at Epoch {:6.2f}'.format(1000*log_eval_metrics_pampjpe.mPVPE, 1000*log_eval_metrics_pampjpe.mPJPE, 1000*log_eval_metrics_pampjpe.PAmPJPE, log_eval_metrics_pampjpe.epoch)
    )
    logger.info(
        'Best Results: (MPJPE)'
        + ' <MPVPE> {:6.2f} <MPJPE> {:6.2f}, <PA-MPJPE> {:6.2f} at Epoch {:6.2f}'.format(1000*log_eval_metrics_mpjpe.mPVPE, 1000*log_eval_metrics_mpjpe.mPJPE, 1000*log_eval_metrics_mpjpe.PAmPJPE, log_eval_metrics_mpjpe.epoch)
    )


def run_eval(args, val_dataloader, FastMETRO_model, smpl, renderer):
    smpl.eval()

    epoch = 0
    if args.distributed:
        FastMETRO_model = torch.nn.parallel.DistributedDataParallel(
            FastMETRO_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    FastMETRO_model.eval()
    
    val_result = run_validate(args, val_dataloader, 
                              FastMETRO_model, 
                              epoch, 
                              smpl,
                              renderer)
    val_mPVPE, val_mPJPE, val_PAmPJPE = val_result['val_mPVPE'], val_result['val_mPJPE'], val_result['val_PAmPJPE']
    
    if args.use_smpl_param_regressor:
        val_smpl_mPVPE, val_smpl_mPJPE, val_smpl_PAmPJPE = val_result['val_smpl_mPVPE'], val_result['val_smpl_mPJPE'], val_result['val_smpl_PAmPJPE']
        logger.info(
            ' '.join(['Validation', 'Epoch: {ep}',]).format(ep=epoch) 
            + '  MPVPE: {:6.2f}, MPJPE: {:6.2f}, PA-MPJPE: {:6.2f} '.format(1000*val_mPVPE, 1000*val_mPJPE, 1000*val_PAmPJPE)
            + '  <SMPL> MPVPE: {:6.2f}, <SMPL> MPJPE: {:6.2f}, <SMPL> PA-MPJPE: {:6.2f} '.format(1000*val_smpl_mPVPE, 1000*val_smpl_mPJPE, 1000*val_smpl_PAmPJPE)
            )
    else:
        logger.info(
            ' '.join(['Validation', 'Epoch: {ep}',]).format(ep=epoch) 
            + '  MPVPE: {:6.2f}, MPJPE: {:6.2f}, PA-MPJPE: {:6.2f} '.format(1000*val_mPVPE, 1000*val_mPJPE, 1000*val_PAmPJPE)
            )
    
    logger.info("The experiment completed successfully. Finalizing run...")

    return

def run_validate(args, val_loader, FastMETRO_model, epoch, smpl, renderer):
    mPVPE = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    smpl_mPVPE = AverageMeter()
    smpl_mPJPE = AverageMeter()
    smpl_PAmPJPE = AverageMeter()
    # switch to evaluation mode
    FastMETRO_model.eval()
    smpl.eval()
    with torch.no_grad():
        print("val_data_num:",val_loader.__len__()) 
        for iteration, (img_keys, images, annotations) in enumerate(val_loader):    
            #iteration 8  128images/8 16段视频    一段视频8帧，视频帧的for语句 out all_imgaes 8张图像
            #out[8] mesh 三维坐标  补帧  【1，3，5，7，8】  五帧
            #out[8] ver  补帧
            #输出成一段视频 opencv  保存到文件夹
            #原图像生成视频8帧
            #mesh生成视频 8帧输出去生成8帧视频
            #ver  8帧 joint关节点 是5帧做补帧的
            #xyz  (x1+x3)/2=x2



            #print(iteration,img_keys)
            #continue
            # compute output
            images = images.cuda(args.device)
            # gt 3d joints
            gt_3d_joints = annotations['joints_3d'].cuda(args.device) # batch_size X 24 X 4 (last for confidence)
            B,S,point_num,point_xyz = gt_3d_joints.shape
            gt_3d_joints = gt_3d_joints.view(B*S,point_num,point_xyz)
            gt_3d_pelvis = gt_3d_joints[:,cfg.J24_NAME.index('Pelvis'),:3] # batch_size X 3
            gt_3d_joints = gt_3d_joints[:,cfg.J24_TO_J14,:] # batch_size X 14 X 4
            gt_3d_joints[:,:,:3] = gt_3d_joints[:,:,:3] - gt_3d_pelvis[:, None, :] # batch_size X 14 X 4
            has_3d_joints = annotations['has_3d_joints'].cuda(args.device) # batch_size

            # gt params for smpl
            gt_pose = annotations['pose'].cuda(args.device) # batch_size X 72
            gt_betas = annotations['betas'].cuda(args.device) # batch_size X 10
            has_smpl = annotations['has_smpl'].cuda(args.device) # batch_size 

            B,S,C,H,W = images.shape
            batch_size = B*S
            images = images.view(B*S,C,H,W)
    
            gt_pose = gt_pose.view(B*S,-1)
            gt_betas = gt_betas.view(B*S,-1) 
            has_smpl = has_smpl.view(B*S)
            has_3d_joints = has_3d_joints.view(B*S) 
            # generate simplified mesh
            gt_3d_vertices_fine = smpl(gt_pose, gt_betas) # batch_size X 6890 X 3

            # normalize ground-truth vertices & joints (based on smpl's pelvis)
            # smpl.get_h36m_joints: from vertex to joint (using smpl)
            gt_smpl_3d_joints = smpl.module.get_h36m_joints(gt_3d_vertices_fine) # batch_size X 17 X 3
            gt_smpl_3d_pelvis = gt_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:] # batch_size X 3
            gt_3d_vertices_fine = gt_3d_vertices_fine - gt_smpl_3d_pelvis[:, None, :] # batch_size X 6890 X 3

            # forward-pass
            out = FastMETRO_model(images)
            pred_cam, pred_3d_vertices_fine = out['pred_cam'], out['pred_3d_vertices_fine']

            # obtain 3d joints, which are regressed from the full mesh
            pred_3d_joints_from_smpl = smpl.module.get_h36m_joints(pred_3d_vertices_fine) # batch_size X 17 X 3
            pred_3d_joints_from_smpl_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl[:,cfg.H36M_J17_TO_J14,:] # batch_size X 14 X 3
            # normalize predicted vertices 
            pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 6890 X 3
            # normalize predicted joints 
            pred_3d_joints_from_smpl = pred_3d_joints_from_smpl - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 14 X 3

            if args.use_smpl_param_regressor:
                pred_rotmat, pred_betas = out['pred_rotmat'], out['pred_betas']
                pred_smpl_3d_vertices = smpl(pred_rotmat, pred_betas) # batch_size X 6890 X 3
                pred_smpl_3d_joints = smpl.module.get_h36m_joints(pred_smpl_3d_vertices) # batch_size X 17 X 3
                pred_smpl_3d_joints_pelvis = pred_smpl_3d_joints[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
                pred_smpl_3d_joints = pred_smpl_3d_joints[:,cfg.H36M_J17_TO_J14,:] # batch_size X 14 X 3
                pred_smpl_3d_vertices = pred_smpl_3d_vertices - pred_smpl_3d_joints_pelvis[:, None, :] # batch_size X 6890 X 3
                pred_smpl_3d_joints = pred_smpl_3d_joints - pred_smpl_3d_joints_pelvis[:, None, :] # batch_size X 14 X 3
                # measure errors
                error_smpl_vertices = mean_per_vertex_position_error(pred_smpl_3d_vertices, gt_3d_vertices_fine, has_smpl)
                error_smpl_joints = mean_per_joint_position_error(pred_smpl_3d_joints, gt_3d_joints,  has_3d_joints)
                error_smpl_joints_pa = reconstruction_error(pred_smpl_3d_joints.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)
                if len(error_smpl_vertices) > 0:
                    smpl_mPVPE.update(np.mean(error_smpl_vertices), int(torch.sum(has_smpl)) )
                if len(error_smpl_joints) > 0:
                    smpl_mPJPE.update(np.mean(error_smpl_joints), int(torch.sum(has_3d_joints)) )
                if len(error_smpl_joints_pa) > 0:
                    smpl_PAmPJPE.update(np.mean(error_smpl_joints_pa), int(torch.sum(has_3d_joints)) )
            
            # measure errors
            error_vertices = mean_per_vertex_position_error(pred_3d_vertices_fine, gt_3d_vertices_fine, has_smpl)
            error_joints = mean_per_joint_position_error(pred_3d_joints_from_smpl, gt_3d_joints,  has_3d_joints)
            error_joints_pa = reconstruction_error(pred_3d_joints_from_smpl.cpu().numpy(), gt_3d_joints[:,:,:3].cpu().numpy(), reduction=None)
            if len(error_vertices) > 0:
                mPVPE.update(np.mean(error_vertices), int(torch.sum(has_smpl)) )
            if len(error_joints) > 0:
                mPJPE.update(np.mean(error_joints), int(torch.sum(has_3d_joints)) )
            if len(error_joints_pa) > 0:
                PAmPJPE.update(np.mean(error_joints_pa), int(torch.sum(has_3d_joints)) )
            FRAMENUM=32
            # visualization
            if args.run_eval_and_visualize:
                if (iteration % 20) == 0:
                    print("Complete {} iterations!! (visualization)".format(iteration))
                pred_2d_joints_from_smpl = orthographic_projection(pred_3d_joints_from_smpl, pred_cam) # batch_size X 14 X 2
                pred_3d_vertices_fine1 = pred_3d_vertices_fine.detach()
                batch_size, num_points, dim = pred_3d_vertices_fine1.size()
                pred_3d_vertices_fine1 = pred_3d_vertices_fine1.view(batch_size // FRAMENUM, FRAMENUM, num_points, dim)
                pred_3d_vertices_fine1[:, FRAMENUM-1, :, :] = (pred_3d_vertices_fine1[:, FRAMENUM-2, :, :] + pred_3d_vertices_fine1[:, FRAMENUM-1, :, :]) / 2.0
                for k in range(FRAMENUM-1):
                    if k%2==1:
                        pred_3d_vertices_fine1[:, k, :, :] = (pred_3d_vertices_fine1[:, k-1, :, :] + pred_3d_vertices_fine1[:, k+1, :, :]) /2.0
                pred_3d_vertices_fine1=pred_3d_vertices_fine1.view(batch_size,num_points,dim)
                if args.use_smpl_param_regressor:
                    pred_smpl_2d_joints = orthographic_projection(pred_smpl_3d_joints, pred_cam) # batch_size X 14 X 2
                    visual_imgs = visualize_mesh_with_smpl(renderer,
                                                           annotations['ori_img'].detach(),
                                                           pred_3d_vertices_fine.detach(), 
                                                           pred_cam.detach(),
                                                           pred_smpl_3d_vertices.detach())
                else:

                    # visual_imgs = visualize_mesh(renderer,
                    #                             annotations['ori_img'].detach(),
                    #                             pred_3d_vertices_fine.detach(), 
                    #                             pred_cam.detach())
                    visual_imgs = visualize_mesh(renderer,
                                                annotations['ori_img'].detach(),
                                                pred_3d_vertices_fine1, 
                                                pred_cam.detach())
                    center = annotations['center']
                    center = center.reshape(-1,2)
                    #org1000 = annotations['ori_img_1000']
                    new_boxx = np.zeros([128,4])
                    for ii in range(128):
                        new_boxx[ii,0] = center[ii,0]
                        new_boxx[ii,1] = center[ii,1]
                        new_boxx[ii,2] = 224
                        new_boxx[ii,3] = 224

                    # bboxes = np.array([cen_x,cen_y,224,224])
                    
                    #orig_cam = convert_crop_cam_to_orig_img(cam=pred_cam.detach().cpu().numpy(),bbox=new_boxx,img_width=org1000.shape[2],img_height=org1000.shape[3])
                    #visual_imgs2 = visualize_mesh2(renderer,
                    #        annotations['ori_img_1000'].detach(),
                    #        pred_3d_vertices_fine.detach(), 
                    #        orig_cam)
                
                all_image = out['all_image']
                backbone_output = out['backbone_output']
                SD_output = out['SD_output']
                # sample_1 = out['sample_1']
                # sample_2 = out['sample_2']
                #ema_S_out = out['ema_S_out']
                #x = out['x']
                #cnn_featmap,ema_featmap,ems_featmap,std_featmap = make_all_heatmap(all_image,first_featmap,ema_out,ema_S_out,x)
                SD_featmap_list, backbone_output_list = make_all_heatmap(all_image,backbone_output,SD_output)
                
                # save_path = "/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/test_byt_3dpw_downtown_weeklyMarket_00/"
                # save_path = "/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/test_byt_3dpw_downtown_stairs_00/"
                # save_path = "/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/test_byt_3dpw_downtown_bus_00/"
                # save_path = "/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/test_byt_3dpw_outdoors_fencing_01/"
                # save_path = "/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/test_byt_3dpw_flat_packBags_00/"
                # save_path = "/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/test_byt_3dpw_downtown_windowShopping_00/"
                save_path = "/root/fengzehui/test_output_images/"
                video_path="/root/fengzehui/test_output_video/"+str(FRAMENUM)+"/video_"+str(FRAMENUM)+"/"
                if not os.path.exists(save_path):
                    #os.mkdir(save_path)
                    os.makedirs(save_path)
                if not os.path.exists(video_path):
                    #os.mkdir(video_path)
                    os.makedirs(video_path)
                all_vis = visual_imgs
                image = annotations['ori_img'].detach()
                #image_1000 = annotations['ori_img_1000'].detach()

                #image_1000 = image_1000.reshape([16*8,*image_1000.shape[2:]])
                image = image.reshape([16*8, 3, 224, 224])
                videoID=0
                stamp = str(epoch) + '_' + str(iteration)
                # out = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"mesh.avi", cv2.VideoWriter_fourcc(*'DIVX'), 8, (224,224))
                # out1 = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"cat.avi", cv2.VideoWriter_fourcc(*'DIVX'), 8, (224,224))
                out = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"mesh.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 8, (224,224))
                out1 = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"cat.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 8, (224,224))

                # visual_imgs = np.asarray(visual_imgs)
                for ii in range(len(all_vis)):
                    this_vis = all_vis[ii]
                    this_image = image[ii,:,:,:]
                    #this_image_1000 = image_1000[ii,:,:,:]

                    #this_image_1000 = np.asarray(this_image_1000)

                    this_image = this_image.transpose(0,1)
                    this_image = this_image.transpose(1,2)
                    this_image = np.asarray(this_image)


                    visual_imgs = this_vis.transpose(0,1)#torch.Size([28930, 228])
                    visual_imgs = visual_imgs.transpose(1,2)
                    visual_imgs = np.asarray(visual_imgs)

                    # cat_image = np.concatenate([this_image*255,visual_imgs*255],axis=1)
                    cat_image = this_image*255
                    mesh_image = visual_imgs*255
                    if is_main_process():
                        stamp = str(epoch) + '_' + str(iteration)
                        temp_fname_cat =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'cat.png'
                        temp_fname_mesh =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'mesh.png'
                        # temp_fname_mesh =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'only_mesh.png'
                        temp_fname_org =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'org.png'

                        backbone_output_heatmap =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'backbone_output_heatmap.png'
                        SD_output_heatmap =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'SD_output_heatmap.png'
                        # sample_1_heatmap =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'sample_1_heatmap.png'
                        # sample_2_heatmap =  save_path + 'visual_' + stamp +"_"+'%05d'%ii+"_"+ 'sample_2_heatmap.png'

                        # if args.use_opendr_renderer:
                        #     visual_imgs[:,:,::-1] = visual_imgs[:,:,::-1]*255
                        #cat_image = cv2.cvtColor(cat_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        mesh_image = cv2.cvtColor(mesh_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                        #this_image_1000 = cv2.cvtColor(this_image_1000.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(temp_fname_cat, cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(temp_fname_mesh, mesh_image)
                        #cv2.imwrite(temp_fname_org, this_image_1000)

                        cv2.imwrite(SD_output_heatmap, cv2.cvtColor(SD_featmap_list[ii], cv2.COLOR_BGR2RGB))
                        cv2.imwrite(backbone_output_heatmap, cv2.cvtColor(backbone_output_list[ii], cv2.COLOR_BGR2RGB))
                        # cv2.imwrite(sample_1_heatmap, cv2.cvtColor(sample_1_list[ii], cv2.COLOR_BGR2RGB))
                        # cv2.imwrite(sample_2_heatmap, cv2.cvtColor(sample_2_list[ii], cv2.COLOR_BGR2RGB))
                        out.write(mesh_image)
                        cat_image_temp=cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB)
                        cat_image_temp = cv2.convertScaleAbs(cat_image_temp)
                        out1.write(cat_image_temp)
                        if ii%FRAMENUM==FRAMENUM-1:
                            out.release()
                            out1.release()
                            if ii!=FRAMENUM-1:
                                videoID+=1
                                # out = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"mesh.avi", cv2.VideoWriter_fourcc(*'DIVX'), 16, (224,224))
                                # out1 = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"cat.avi", cv2.VideoWriter_fourcc(*'DIVX'), 16, (224,224))
                                out = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"mesh.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 16, (224,224))
                                out1 = cv2.VideoWriter(video_path+stamp+'_'+str(videoID)+'_'+"cat.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 16, (224,224))

                        #/HOME/HOME/Zhongzhangnan/FastMETRO_EMA_adapt/src/tools/run_fastmetro_bodymesh_dp_3dpw_more_data_pips_S_adapt_test.pycv2.imwrite(final_heatmap, cv2.cvtColor(std_featmap[ii], cv2.COLOR_RGB2BGR))

            # visual_imgs = visualize_mesh(   renderer,
            #                     annotations['ori_img'].detach(),
            #                     annotations['joints_2d'].detach(),
            #                     pred_3d_vertices_fine.detach(), 
            #                     pred_cam.detach(),
            #                     pred_2d_joints_from_smpl.detach())
            # visual_imgs = visual_imgs.transpose(0,1)
            # visual_imgs = visual_imgs.transpose(1,2)
            # visual_imgs = np.asarray(visual_imgs)

    val_mPVPE = all_gather(float(mPVPE.avg))
    val_mPVPE = sum(val_mPVPE)/len(val_mPVPE)
    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)
    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)
    val_count = all_gather(float(mPVPE.count))
    val_count = sum(val_count)
    val_result = {}
    val_result['val_mPVPE'] = val_mPVPE
    val_result['val_mPJPE'] = val_mPJPE
    val_result['val_PAmPJPE'] = val_PAmPJPE
    val_result['val_count'] = val_count



    
    if args.use_smpl_param_regressor:
        val_smpl_mPVPE = all_gather(float(smpl_mPVPE.avg))
        val_smpl_mPVPE = sum(val_smpl_mPVPE)/len(val_smpl_mPVPE)
        val_smpl_mPJPE = all_gather(float(smpl_mPJPE.avg))
        val_smpl_mPJPE = sum(val_smpl_mPJPE)/len(val_smpl_mPJPE)
        val_smpl_PAmPJPE = all_gather(float(smpl_PAmPJPE.avg))
        val_smpl_PAmPJPE = sum(val_smpl_PAmPJPE)/len(val_smpl_PAmPJPE)
        val_result['val_smpl_mPVPE'] = val_smpl_mPVPE
        val_result['val_smpl_mPJPE'] = val_smpl_mPJPE
        val_result['val_smpl_PAmPJPE'] = val_smpl_PAmPJPE
        
    return val_result
def visualize_mesh2(renderer, images, pred_vertices, pred_cam):
    rend_imgs = []
    B,S,C,X,Y = images.shape
    images = images.reshape(B*S, C, X, Y)
    batch_size = pred_vertices.shape[0]
    
    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # img = np.zeros_like(img)
        # Get predicted vertices for the particular example
        vertices = pred_vertices[i]
        cam = pred_cam[i]
        # Visualize reconstruction
        if args.use_opendr_renderer:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_opendr(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_opendr(img, vertices, cam, renderer)
        else:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_pyrender(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_pyrender(img, vertices, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))

    # rend_imgs = make_grid(rend_imgs, nrow=1)
    
    return rend_imgs

def visualize_mesh(renderer, images, pred_vertices, pred_cam):
    rend_imgs = []
    B,S,C,X,Y = images.shape
    images = images.reshape(B*S, C, X, Y)
    batch_size = pred_vertices.shape[0]
    
    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose(1,2,0)
        #img = np.zeros_like(img)
        # Get predicted vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_cam[i].cpu().numpy()
        # Visualize reconstruction
        if args.use_opendr_renderer:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_opendr(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_opendr(img, vertices, cam, renderer)
        else:
            if args.visualize_multi_view:
                rend_img = visualize_reconstruction_multi_view_pyrender(img, vertices, cam, renderer)
            else:
                rend_img = visualize_reconstruction_pyrender(img, vertices, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))

    # rend_imgs = make_grid(rend_imgs, nrow=1)
    
    return rend_imgs

def visualize_mesh_with_smpl(renderer, images, pred_vertices, pred_cam, pred_smpl_vertices):
    rend_imgs = []
    B,S,C,X,Y = images.shape
    images = images.reshape(B*S, C, X, Y)
    batch_size = pred_vertices.shape[0]
    
    for i in range(batch_size):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get predicted vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        smpl_vertices = pred_smpl_vertices[i].cpu().numpy()
        cam = pred_cam[i].cpu().numpy()
        # Visualize reconstruction
        if args.use_opendr_renderer:
            rend_img = visualize_reconstruction_smpl_opendr(img, vertices, cam, renderer, smpl_vertices)
        else:
            rend_img = visualize_reconstruction_smpl_pyrender(img, vertices, cam, renderer, smpl_vertices)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    # rend_imgs = make_grid(rend_imgs, nrow=1)
    
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
    parser.add_argument("--val_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/test_has_gender.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    # parser.add_argument("--val_yaml", default='/HOME/HOME/data/PointHMR/datasets/3dpw/train.yaml', type=str, required=False,
    #                     help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=8, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='output_3dpw_step8_ema_back_S_adapt_loss/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--saving_epochs", default=1, type=int)
    parser.add_argument("--resume_checkpoint", default="/root/fengzehui/output_3dpw_step8_ema_back_S_adapt_loss_test/checkpoint-best/state_dict.bin", type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--resume_epoch", default=0, type=int)
    parser.add_argument("--resume_mpjpe_best_epoch", default=0, type=float)
    parser.add_argument("--resume_mpjpe_best_mpvpe", default=0, type=float)
    parser.add_argument("--resume_mpjpe_best_mpjpe", default=0, type=float)
    parser.add_argument("--resume_mpjpe_best_pampjpe", default=0, type=float)
    parser.add_argument("--resume_pampjpe_best_epoch", default=0, type=float)
    parser.add_argument("--resume_pampjpe_best_mpvpe", default=0, type=float)
    parser.add_argument("--resume_pampjpe_best_mpjpe", default=0, type=float)
    parser.add_argument("--resume_pampjpe_best_pampjpe", default=0, type=float)

    
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-6, type=float, 
                        help="The initial lr.")
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.3, type=float,
                        help='gradient clipping maximal norm')
    parser.add_argument("--num_train_epochs", default=600, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--uniform", default=True, type=bool)
    # Loss coefficients
    parser.add_argument("--joints_2d_loss_weight", default=100.0, type=float)
    parser.add_argument("--vertices_3d_loss_weight", default=100.0, type=float)
    parser.add_argument("--edge_normal_loss_weight", default=100.0, type=float)
    parser.add_argument("--joints_3d_loss_weight", default=1000.0, type=float)
    parser.add_argument("--vertices_fine_loss_weight", default=0.25, type=float) 
    parser.add_argument("--vertices_intermediate_loss_weight", default=0.50, type=float) 
    parser.add_argument("--vertices_coarse_loss_weight", default=0.25, type=float)
    parser.add_argument("--edge_gt_loss_weight", default=5.0, type=float) 
    parser.add_argument("--edge_self_loss_weight", default=1e-4, type=float) 
    parser.add_argument("--normal_loss_weight", default=0.1, type=float)
    parser.add_argument("--smpl_param_loss_weight", default=1000.0, type=float)
    # Model parameters
    parser.add_argument("--model_name", default='FastMETRO-L', type=str,
                        help='Transformer architecture: FastMETRO-S, FastMETRO-M, FastMETRO-L')
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
    # Others
    #########################################################
    parser.add_argument("--run_evaluation", default=True, action='store_true',) 
    parser.add_argument("--run_eval_and_visualize", default=True, action='store_true',)
    parser.add_argument('--logging_steps', type=int, default=1000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument('--model_save', default=False, action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--exp", default='FastMETRO', type=str, required=False)
    parser.add_argument("--visualize_training", default=False, action='store_true',)
    parser.add_argument("--visualize_multi_view", default=False, action='store_true',)
    parser.add_argument("--use_opendr_renderer", default=True, action='store_true',)
    parser.add_argument("--num_joints", default=14, action='store_true',) 
    parser.add_argument("--num_vertices", default=431, action='store_true',) 
    parser.add_argument("--embed_dim_ratio", default=32, action='store_true',)     
    parser.add_argument("--depth", default=1, action='store_true',)     
    parser.add_argument("--frames", default=8, action='store_true',) 
    parser.add_argument("--number_of_kept_frames", default=4, action='store_true',)     
    parser.add_argument("--number_of_kept_coeffs", default=4, action='store_true',) 

    args = parser.parse_args()
    return args


def main(args):
    print("FastMETRO for 3D Human Mesh Reconstruction!")
    USE_MULTI_GPU = True
    import os 
    # 检测机器是否有多张显卡
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        MULTI_GPU = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        device_ids = [0, 1]
    else:
        MULTI_GPU = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # args.distributed = args.num_gpus > 1
    args.distributed = False
    args.device = torch.device(args.device)
    # if args.distributed:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    #     args.local_rank = int(os.environ['LOCAL_RANK'])
    #     print("Init distributed training on local rank {} ({}), world size {}".format(args.local_rank, int(os.environ["LOCAL_RANK"]), args.num_gpus))
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(
    #         backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    #     )
    #     local_rank = int(os.environ["LOCAL_RANK"])
    #     args.device = torch.device("cuda", local_rank)
    #     torch.distributed.barrier()

    mkdir(args.output_dir)
    logger = setup_logger("FastMETRO", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL()
    smpl = torch.nn.DataParallel(smpl,device_ids=device_ids)
    smpl.to(device)
    mesh_sampler = Mesh()
    mesh_sampler = torch.nn.DataParallel(mesh_sampler,device_ids=device_ids)
    mesh_sampler.to(device)

    smpl_intermediate_faces = torch.from_numpy(np.load('/root/fengzehui/src/modeling/data/smpl_1723_faces.npy', encoding='latin1', allow_pickle=True).astype(np.int64)).to(args.device)
    smpl_intermediate_edges = torch.from_numpy(np.load('/root/fengzehui/src/modeling/data/smpl_1723_edges.npy', encoding='latin1', allow_pickle=True).astype(np.int64)).to(args.device)
 
    # Renderer for visualization
    if args.use_opendr_renderer:
        renderer = OpenDR_Renderer(faces=smpl.module.faces.cpu().numpy())
    else:
        renderer = PyRender_Renderer(faces=smpl.module.faces.cpu().numpy())
    
    logger.info("Training Arguments %s", args)
    
    # Load model
    # if args.run_evaluation and (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and ('state_dict' not in args.resume_checkpoint):
    #     # if only run eval, load checkpoint
    #     logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
    #     _FastMETRO_Network = torch.load(args.resume_checkpoint)
    # else:
    #     # init ImageNet pre-trained backbone model
    #     if args.arch == 'hrnet-w64':
    #         hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    #         hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
    #         hrnet_update_config(hrnet_config, hrnet_yaml)
    #         backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
    #         logger.info('=> loading hrnet-v2-w64 model')
    #     elif args.arch == 'resnet50':
    #         logger.info("=> using pre-trained model '{}'".format(args.arch))
    #         backbone = models.__dict__[args.arch](pretrained=True)
    #         # remove the last fc layer
    #         backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    #     else:
    #         assert False, "The CNN backbone name is not valid"

    #     _FastMETRO_Network = FastMETRO_Network(args, backbone, mesh_sampler)
    #     # number of parameters
    #     overall_params = sum(p.numel() for p in _FastMETRO_Network.parameters() if p.requires_grad)
    #     backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    #     transformer_params = overall_params - backbone_params
    #     logger.info('Number of CNN Backbone learnable parameters: {}'.format(backbone_params))
    #     logger.info('Number of Transformer Encoder-Decoder learnable parameters: {}'.format(transformer_params))
    #     logger.info('Number of Overall learnable parameters: {}'.format(overall_params))

    #     if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None'):
    #         # for fine-tuning or resume training or inference, load weights from checkpoint
    #         logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
    #         cpu_device = torch.device('cpu')
    #         state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
    #         _FastMETRO_Network.load_state_dict(state_dict, strict=False)
    #         del state_dict
    hrnet_yaml = '/root/fengzehui/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
    hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
    hrnet_update_config(hrnet_config, hrnet_yaml)
    backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
    logger.info('=> loading hrnet-v2-w64 model')
    _FastMETRO_Network = FastMETRO_Network(args, backbone, mesh_sampler,smpl)
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
    

    #for name,para in _FastMETRO_Network.named_parameters():
    #    # with torch.no_grad():
    #    para = para + (torch.rand(para.size())-0.5) * 0.15 * torch.std(para)
    _FastMETRO_Network = torch.nn.DataParallel(_FastMETRO_Network,device_ids=device_ids)
    _FastMETRO_Network.to(args.device)

    val_dataloader = make_data_loader(args, args.val_yaml, args.distributed, is_train=False, scale_factor=args.img_scale_factor)
    if args.run_evaluation:
        run_eval(args, val_dataloader, _FastMETRO_Network, smpl, renderer)
    else:
        train_dataloader = make_data_loader(args, args.train_yaml, args.distributed, is_train=True, scale_factor=args.img_scale_factor)
        run_train(args, train_dataloader, val_dataloader, _FastMETRO_Network, smpl, mesh_sampler, renderer, smpl_intermediate_faces, smpl_intermediate_edges)


if __name__ == "__main__":
    args = parse_args()
    main(args)