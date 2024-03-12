# ----------------------------------------------------------------------------------------------

"""
End-to-End inference codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import torch
import torchvision.models as models
import numpy as np
import cv2
import skimage.io as io
import sys 
# print(sys.path)
sys.path.append("/home/zjlab1/workspace/fengzehui/")
from PIL import Image
from torchvision import transforms
from src.modeling.model.modeling_PBET_EMA_4_S_adapt_train import PBET_Body_Network as _PBET_Network
from src.modeling._smpl import SMPL, Mesh
from src.modeling.model.hrnet_cls_net_featmaps_adapt_addSD import get_cls_net
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.utils.logger import setup_logger
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.geometric_layers import orthographic_projection, rodrigues
from src.utils.renderer_opendr import OpenDR_Renderer, visualize_reconstruction_opendr, visualize_reconstruction_smpl_opendr
try:
    from src.utils.renderer_pyrender import PyRender_Renderer, visualize_reconstruction_pyrender, visualize_reconstruction_smpl_pyrender
except:
    print("Failed to import renderer_pyrender. Please see docs/Installation.md")


transform = transforms.Compose([
transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])
def create_video_from_images(image_folder, output_video_path, fps=10):
    # 获取文件夹中的图片文件，并按文件名排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])

    # 获取第一张图片的宽度和高度
    img = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = img.shape

    # 定义视频编码器并创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历图片文件，并将每一帧添加到视频中
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频写入对象
    video_writer.release()
def run_inference(args, image_list, PBET_model, smpl, renderer):
    # switch to evaluate mode
    PBET_model.eval()

    frame_count = 0 
    batch_imgs_list = []
    batch_visual_imgs_list = []
    frame_num = 0
    for image_file in image_list:
        if 'pred' not in image_file:
            frame_num = frame_num + 1
            img = Image.open(image_file)
       
            img_tensor = transform(img)
            img_visual = transform_visualize(img)

            batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
            batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
            batch_imgs_list.append(batch_imgs)
            batch_visual_imgs_list.append(batch_visual_imgs)
            final_batch = torch.cat(batch_imgs_list, dim=0) 
            if frame_num%8==0:
                B,C,H,W = final_batch.shape
                #images = final_batch.view(B//8,8,C,H,W)    
                # forward-pass
                out = PBET_model(final_batch)
                pred_cam, pred_3d_vertices_fine, pred_3d_vertices_coarse, pred_3d_joints= out['pred_cam'], out['pred_3d_vertices_fine'],out['pred_3d_vertices_coarse'],out['pred_3d_joints']
                for i in range(0,8):    
                # obtain 3d joints, which are regressed from the full mesh
                    pred_2d_vertices_from_token = orthographic_projection(pred_3d_vertices_coarse, pred_cam)
                    pred_2d_joints = orthographic_projection(pred_3d_joints, pred_cam)
                    pred_3d_joints_from_smpl = smpl.get_h36m_joints(pred_3d_vertices_fine) # batch_size X 17 X 3
                    pred_3d_joints_from_smpl_pelvis = pred_3d_joints_from_smpl[:,cfg.H36M_J17_NAME.index('Pelvis'),:]
                    pred_3d_vertices_fine = pred_3d_vertices_fine - pred_3d_joints_from_smpl_pelvis[:, None, :] # batch_size X 6890 X 3
                    visual_img = visualize_mesh(renderer,
                                                torch.squeeze(batch_visual_imgs_list[i], 0).cuda(),
                                                pred_3d_vertices_fine[i].detach(), 
                                                pred_cam[i].detach())
                    temp_fname = "out/prediction_{}".format(frame_count)+ '.jpg'
                    print('save to ', temp_fname)
                    visual_img = visual_img.transpose(1,2,0)
                    visual_img = np.asarray(visual_img).copy()
                    if args.use_opendr_renderer:
                        visual_img[:,:,::-1] = visual_img[:,:,::-1]*255
                    cv2.imwrite(temp_fname, np.asarray(visual_img[:,:,::-1]))
                    frame_count = frame_count + 1
                batch_imgs_list = []
                batch_visual_imgs_list = []

    create_video_from_images("out/", "out/video_prediction.mp4")
    logger.info("The inference completed successfully. Finalizing run...")
    
    return 

def visualize_mesh(renderer, image, pred_vertices, pred_cam):
    img = image.cpu().numpy().transpose(1,2,0)

    # Get predicted vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    cam = pred_cam.cpu().numpy()
    
    # Visualize reconstruction
    if args.use_opendr_renderer:
        
        rend_img = visualize_reconstruction_opendr(img,vertices, cam, renderer)
        
    else:
        rend_img = visualize_reconstruction_pyrender(img, vertices, cam, renderer)
    rend_img = rend_img.transpose(2,0,1)
    
    return rend_img


def visualize_mesh_with_smpl(renderer, image, pred_vertices, pred_cam, pred_smpl_vertices):
    img = image.cpu().numpy().transpose(1,2,0)

    # Get predicted vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    smpl_vertices = pred_smpl_vertices.cpu().numpy()
    cam = pred_cam.cpu().numpy()

    # Visualize reconstruction
    if args.use_opendr_renderer:
        rend_img = visualize_reconstruction_smpl_opendr(img, vertices, cam, renderer, smpl_vertices)
    else:
        rend_img = visualize_reconstruction_smpl_pyrender(img, vertices, cam, renderer, smpl_vertices)
    rend_img = rend_img.transpose(2,0,1)
    
    return rend_img
def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """ 
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d

def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--video_file_or_path", default='video/video.mp4', type=str, 
                        help="test data")
    #########################################################
    # Loading/Saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--resume_checkpoint", default="models/P-BET-checkpoint.bin", type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument("--model_name", default='PBET-L', type=str,
                        help='Transformer architecture: PBET-S, PBET-M, PBET-L')
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
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
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
def extract_frames_from_video(video_path):
    # 创建一个空列表来存储图像路径
    image_list = []
    if not os.path.exists("out/"):
        os.makedirs("out/")
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 确保视频文件被正确打开
    if not cap.isOpened():
        raise ValueError("无法打开视频文件 {}".format(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 计算应保留的帧数
    num_frames_to_keep = (total_frames // 8) * 8
    if not os.path.exists("out/images/"):
        os.makedirs("out/images/")
    # 逐帧读取视频，并将每一帧转换成图像并保存路径
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= num_frames_to_keep:
            break
        # 仅保存八的倍数的帧数
        transform_visualize = transforms.Compose([transforms.Resize(224),
                                                transforms.CenterCrop(224)])
        frame = transform_visualize(Image.fromarray(frame.astype('uint8')))
        # 生成图像的文件名
        image_filename = "out/images/frame_{}.png".format(frame_count)
        print(os.path.dirname(args.video_file_or_path))
        image_path = image_filename
        print(image_path)
        # 保存图像
        cv2.imwrite(image_path, np.asarray(frame))
        # 将图像路径添加到列表中
        image_list.append(image_path)
        frame_count += 1
    # 释放视频对象
    cap.release()

    return image_list

def main(args):
    print("PBET for 3D Human Mesh Reconstruction!")
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    mkdir(args.output_dir)
    logger = setup_logger("PBET Inference", args.output_dir, 0)
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    smpl = SMPL()
    mesh_sampler = Mesh()
    # Renderer for visualization
    if args.use_opendr_renderer:
        renderer = OpenDR_Renderer(faces=smpl.faces.cpu().numpy())
    else:
        renderer = PyRender_Renderer(faces=smpl.faces.cpu().numpy())

    # Load pretrained model    
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None') and ('state_dict' not in args.resume_checkpoint):
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        PBET_Network = torch.load(args.resume_checkpoint)
    else:
        # init ImageNet pre-trained backbone model
        if args.arch == 'hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        elif args.arch == 'resnet50':
            logger.info("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        else:
            assert False, "The CNN backbone name is not valid"
        smpl = SMPL().to(args.device)
        PBET_Network = _PBET_Network(args, backbone, mesh_sampler,smpl)
        # number of parameters
        overall_params = sum(p.numel() for p in PBET_Network.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        transformer_params = overall_params - backbone_params
        logger.info('Number of CNN Backbone learnable parameters: {}'.format(backbone_params))
        logger.info('Number of Transformer Encoder-Decoder learnable parameters: {}'.format(transformer_params))
        logger.info('Number of Overall learnable parameters: {}'.format(overall_params))

        if (args.resume_checkpoint != None) and (args.resume_checkpoint != 'None'):
            # for fine-tuning or resume training or inference, load weights from checkpoint
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            cpu_device = torch.device('cpu')
            state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
            PBET_Network.load_state_dict(state_dict, strict=False)
            del state_dict

    PBET_Network.to(args.device)
    logger.info("Run inference")

    image_list = []
    if not args.video_file_or_path:
        raise ValueError("video_file_or_path not specified")
    if op.isfile(args.video_file_or_path):
        if args.video_file_or_path.endswith(".mp4"):
            image_list = extract_frames_from_video(args.video_file_or_path)
    run_inference(args, image_list, PBET_Network, smpl, renderer)    

if __name__ == "__main__":
    args = parse_args()
    main(args)