# ----------------------------------------------------------------------------------------------
# ProGraph Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
ProGraph model.
"""
from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from .transformer import build_transformer
from .position_encoding import build_position_encoding
from .smpl_param_regressor import build_smpl_parameter_regressor
import torch
from torch import nn
from einops import rearrange
from omegaconf import OmegaConf
from .unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler
from .ReferenceEncoder import ReferenceEncoder
from .Visual_Prompt import visual_prompt
import clip
from diffusers.models.vae import DiagonalGaussianDistribution
from .pose_hrnet import PoseHighResolutionNet
from .model_poseformer import PoseTransformerV2
from .attention import GraphResBlock
class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)
class coarse2intermediate_upsample(nn.Module):
    def __init__(self, inchannels, outchannels, scale_factor):
        super().__init__() 
        self.transpose_conv = nn.ConvTranspose1d(
            in_channels=inchannels*scale_factor,
            out_channels=outchannels,
            kernel_size=4,
            stride=4,
            padding=0
        )
        self.scale_factor = scale_factor
        self.linear_layer = nn.Linear(in_features=12, out_features=3)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.interpolate(input=x, scale_factor = self.scale_factor, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)
        x = self.transpose_conv(x)
        x = self.linear_layer(x)
        return x
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=4):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Assuming self.conv1 is a convolutional layer with appropriate parameters
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        # Additional convolutional layers
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Transpose convolutional layers for upsampling
        self.transconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(64)

        self.transconv2 = nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2, padding=0)
        self.bn5 = nn.BatchNorm2d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Upsample using transpose convolutions
        m = self.transconv1(x)
        m = self.bn4(m)
        m = F.relu(m)
        m = self.transconv2(m)
        m = self.bn5(m)
        m = F.relu(m)

        v = self.transconv1(x)
        v = self.bn4(v)
        v = F.relu(v)
        v = self.transconv2(v)
        v = self.bn5(v)
        v = F.relu(v)

        return m,v

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(512, 2048, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(2048)


    def forward(self, z):


        z = F.relu(self.bn1(self.conv1(z)))
        z = F.relu(self.bn2(self.conv2(z)))
        z = F.relu(self.bn3(self.conv3(z)))
        z = F.relu(self.bn4(self.conv4(z)))
        z = F.relu(self.bn5(self.conv5(z)))

        return z

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.SVD =SVD()
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, images ):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)
        z = self.SVD(z,images)
        z = rearrange(z, "b c f h w -> (b f) c h w", f=8)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar


class SVD(nn.Module):
    def __init__(self):
        super(SVD, self).__init__()
        self.noise_scheduler_kwargs = dict(
            num_train_timesteps=1000,
            beta_start=0.00004,
            beta_end=0.0006,
            beta_schedule='scaled_linear',
            steps_offset=1,
            clip_sample=False
        )
        self.omega_conf_obj = OmegaConf.create(self.noise_scheduler_kwargs)
        self.noise_scheduler = DDIMScheduler(**OmegaConf.to_container(self.omega_conf_obj))
        #self.clip_image_encoder = ReferenceEncoder(model_path="clip-vit-base-patch32")#VedioCLIP预训练模型
        self.unet_additional_kwargs = dict(
            use_motion_module=True,
            motion_module_resolutions=[1, 2, 4, 8],
            unet_use_cross_frame_attention=False,
            unet_use_temporal_attention=False,
            motion_module_type='Vanilla',
            motion_module_kwargs=dict(
                num_attention_heads=1,
                num_transformer_block=1,
                attention_block_types=['Temporal_Self', 'Temporal_Self'],
                temporal_position_encoding=True,
                temporal_position_encoding_max_len=24,
                temporal_attention_dim_div=1,
                zero_initialize=True
            )
        )
        self.unet_conf_obj = OmegaConf.create(self.unet_additional_kwargs)
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path="/home/zjlab1/workspace/fengzehui/models/stable-diffusion-v1-5", subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(self.unet_conf_obj)
        )
        # print(("=> loading checkpoint '{}'".format(config.pretrain)))
        checkpoint = torch.load('/home/zjlab1/workspace/fengzehui/models/vit-32-8f.pt')
        self.ActionCLIP, self.clip_state_dict = clip.load('ViT-B/32', device='cuda', jit=False, tsm=False,T=8, dropout=0,emb_dropout=0)
        #self.fusion_model = visual_prompt('Transf', self.clip_state_dict, 8)
        #self.model_image = ImageCLIP(self.ActionCLIP)
        self.ActionCLIP.load_state_dict(checkpoint['model_state_dict'])
        self.linear_transform = torch.nn.Linear(512, 768).to('cuda')

    def forward(self,img_features,images):
        #video_length = x.shape[1]
        #clip_ref_image = batch["clip_ref_image"]
        # Use RGB image as input
        #x = batch['img']#原先的代码
        #batch_size = pixel_values_pose.shape[0]

        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio



        #conditioning_feats = self.backbone(x[:,:,:,32:-32])
        with torch.no_grad():
            #pixel_values_pose = rearrange(pixel_values_pose, "b f c h w -> (b f) c h w")
            # latents = DiagonalGaussianDistribution(img_features)
            # latents = latents.sample()
            latents = rearrange(img_features, "(b f) c h w -> b c f h w", f=8)
            latents = latents * 0.18215
            

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler_kwargs['num_train_timesteps'], (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)



        # with torch.no_grad():
            # prompt_ids = tokenizer(
            #     batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            # ).input_ids.to(latents.device)
            # encoder_hidden_states = text_encoder(prompt_ids)[0]
        #    encoder_hidden_states = self.clip_image_encoder(x).unsqueeze(1)#              TODO:这里的clip_image_encoder换成VideoCLIP 输入是视频
        with torch.no_grad():
        
                image = rearrange(images, "(b f) c h w -> b f c h w", f=8)
                b, t, c, h, w = image.size()
                image_input = image.contiguous().view(-1, c, h, w)
                encoder_hidden_states = self.ActionCLIP.encode_image(image_input).view(b, t, -1)
                encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float32)
        encoder_hidden_states = self.linear_transform(encoder_hidden_states)
        
        conditioning_feats= self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return conditioning_feats

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img  
class CorrBlock:
    def __init__(self, fmaps, num_levels=4, radius=4):
        B, S, C, H, W = fmaps.shape
        # print('fmaps', fmaps.shape)
        self.S, self.C, self.H, self.W = S, C, H, W

        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        # print('fmaps', fmaps.shape)

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels-1):
            fmaps_ = fmaps.reshape(B*S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)
            # print('fmaps', fmaps.shape)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert(D==2)

        x0 = coords[:,0,:,0].round().clamp(0, self.W-1).long()
        y0 = coords[:,0,:,1].round().clamp(0, self.H-1).long()

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i] # B, S, N, H, W
            _, _, _, H, W = corrs.shape
            
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device) 

            centroid_lvl = coords.reshape(B*S*N, 1, 1, 2) / 2**i
            delta_lvl = delta.reshape(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(corrs.reshape(B*S*N, 1, H, W), coords_lvl)
            corrs = corrs.reshape(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1) # B, S, N, LRR*2
        return out.contiguous().float()

    def corr(self, targets):
        B, S, N, C = targets.shape
        assert(C==self.C)
        assert(S==self.S)

        fmap1 = targets

        self.corrs_pyramid = []
        for fmaps in self.fmaps_pyramid:
            _, _, _, H, W = fmaps.shape
            fmap2s = fmaps.reshape(B, S, C, H*W)
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.reshape(B, S, N, H, W) 
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)
def taubin_smoothing(L, v, maxiter=10, Labmda=0.5, Mu=-0.53):
    L = L.to(v.device)
    batch_size, n_vertex, _ =v.shape

    for iter in range(maxiter):
        laplacian_coords = L.mm(v.permute(1, 0, 2).reshape(n_vertex, batch_size * 3)).reshape(n_vertex,
                                                                                          batch_size,
                                                                                          3).permute(1, 0, 2)

        v = v - Labmda * laplacian_coords
        laplacian_coords = L.mm(v.permute(1, 0, 2).reshape(n_vertex, batch_size * 3)).reshape(n_vertex,
                                                                                              batch_size,
                                                                                              3).permute(1, 0, 2)
        v = v - Mu * laplacian_coords

    return v

def calc_Laplacian(vertices, faces):

    nvertex = vertices.shape[0]
    nface = faces.shape[0]

    V = np.ones((2* nface))
    I = np.arange(2* nface).tolist()
    J1 = np.c_[faces[:, 0], faces[:, 2]].flatten().tolist()
    J2 = np.c_[faces[:, 1], faces[:, 1]].flatten().tolist()

    #print(J1.shape)
    i1 = torch.tensor([I,J1])
    i2 = torch.tensor([I,J2])
    V = torch.tensor(V)
    G1 = torch.sparse_coo_tensor(i1, V, (2*nface, nvertex))
    G2 = torch.sparse_coo_tensor(i2, V, (2*nface, nvertex))

    #G1 = opt.spmatrix(V,
    #                  np.array(I, dtype=int),
    #                  np.array(J1, dtype=int),
    #                  (3 * self.nface, self.nvertex))

    #G2 = opt.spmatrix(V,
    #                  np.array(I, dtype=int),
    #                  np.array(J2, dtype=int),
    #                  (3 * self.nface, self.nvertex))

    G = G1 - G2
    GT = G.transpose(1,0)
    L = torch.mm(GT,G.to_dense()).to_sparse().to(torch.float32)

    idx=torch.arange(nvertex).repeat(2,1)
    v=1/torch.diag(L.to_dense())
    print(v)
    M = torch.sparse.FloatTensor(idx,
                                 v, (nvertex, nvertex))

    L = M.mm(L.to_dense())

    return L



class ProGraph_Body_Network(nn.Module):
    """ProGraph for 3D human pose and mesh reconstruction from a single RGB image"""
    def __init__(self, args, backbone, mesh_sampler,smpl):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.mesh_sampler = mesh_sampler
        self.num_joints = self.args.num_joints
        self.num_vertices = self.args.num_vertices
        self.backbone = backbone

        self.hidden_dim  = 256
        self.latent_dim  = 3
        self.corr_levels = 4
        self.corr_radius = 3
        self.levels=0.9
        # the number of transformer layers
        if 'ProGraph-S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'ProGraph-M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'ProGraph-L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"
    
        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead, 
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        
        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'], hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'], hidden_dim=self.transformer_config_2['model_dim'])
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        
        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        zeros_1 = torch.tensor(np.zeros((self.num_vertices, self.num_joints)).astype(bool)) 
        zeros_2 = torch.tensor(np.zeros((self.num_joints, (self.num_joints + self.num_vertices))).astype(bool)) 
        self.adjacency_indices = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_indices.pt')
        self.adjacency_matrix_value = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_values.pt')
        self.adjacency_matrix_size = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(self.adjacency_indices, self.adjacency_matrix_value, size=self.adjacency_matrix_size).to_dense()
        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
        self.coarse2intermediate_upsample = nn.Linear(431, 1723)
        self.conv_transform_down = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1)
        self.conv_layers2=nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)
        self.Poseformer_cam =PoseTransformerV2(args=args,num_frame=8,num_joints=1)
        self.Poseformer_ver1 =PoseTransformerV2(args=args,num_frame=8,num_joints=14)
        self.Graph = GraphResBlock(in_channels=128,out_channels=128)
        if args.use_smpl_param_regressor:
            self.smpl_parameter_regressor = build_smpl_parameter_regressor()

        vertices = smpl.v_template.detach().cpu()
        self.faces = smpl.faces.detach().cpu()
        with torch.no_grad():
            self.L = torch.tensor(calc_Laplacian(vertices, self.faces).cuda(args.device), requires_grad=False)
        self.bn4 = nn.BatchNorm2d(2048, momentum=0.1)
        
        self.relu = nn.ReLU(inplace=True)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.VAE1 = VAE(self.encoder, self.decoder)           
        
        
        
        # self.SVD=SVD()
    def forward(self, images):#torch.Size([16, 3, 224, 224])
        #[B, 3, 224, 224]
        device = images.device
        batch_size = images.size(0)
        B, C, H, W = images.shape   # 64, 3, 244, 244
        S = 8
        B = B//S                     # 8*8, 3, 244, 244
        N = 431
        H8 = 56
        W8 = 56
        # preparation   
        #[1, 512] to [1, B, 512]
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # 1 X batch_size X 512 
        #[14, 512] , [431, 512] to [445, B, 512]
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(1).repeat(1, batch_size, 1) # (num_joints + num_vertices) X batch_size X 512
        #[445, 445]
        attention_mask = self.attention_mask.to(device) # (num_joints + num_vertices) X (num_joints + num_vertices)
        
        # extract image features through a CNN backbone
        #[B, 3, 224, 224]to [B, 64, 56, 56] [B, 2048, 7, 7]
        #x,img_features = self.backbone(images) # batch_size X 2048 X 7 X 7
        #x,img_features = self.backbone(images)
        all_images = images
        with torch.no_grad():
            x, img_features, backbone_output = self.backbone(images)
        reconstructed_x, mu, logvar = self.VAE1(img_features, images)
        img_features = img_features *self.levels + (1-self.levels) * reconstructed_x
        SD_output = img_features
        _, _, h, w = img_features.shape
        # img_features = img_features + residual1
        
        # #[B, 2048, 7, 7] to [B, 512, 7, 7] to [49, B, 512]
        img_features = self.conv_1x1(img_features).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512
        
        # positional encodings
        #[49, B, 512]
        pos_enc_1 = self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512
        #[49, B, 128]
        pos_enc_2 = self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 128

        # first transformer encoder-decoder
        #input :[49, B, 512] ,[1, B, 512] , [445, B, 512],[49, B, 512],[445, 445]
        #output: [1, B, 512],[49, B, 512],[445, B, 512]
        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=attention_mask)
        
        # progressive dimensionality reduction
        # [1, B, 512] to [1, B, 128]
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1) # 1 X batch_size X 128 
        # [49, B, 512] to [49, B, 128]
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1) # 49 X batch_size X 128 
        #[445, B, 512] to [445, B, 128]
        reduced_jv_features_1 = self.dim_reduce_dec(jv_features_1) # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        #input :[49, B, 128],[1, B, 128],[445, B, 128],[49, B, 128],[445, 445]
        #output: [1, B, 128],[49, B, 128],[445, B, 128]
        cam_features_2, _, jv_features_2 = self.transformer_2(reduced_enc_img_features_1, reduced_cam_features_1, reduced_jv_features_1, pos_enc_2, attention_mask=attention_mask) 

        # estimators
        #[1, B, 128] to [B, 3]
        pred_cam1 = self.cam_predictor(cam_features_2).view(batch_size, 3) # batch_size X 3
        #[B 8, f 8, C 128]
        cam_features_2 = cam_features_2.view(-1, 8, 1, 128)
        pred_cam2 = self.Poseformer_cam(cam_features_2).view(batch_size, 3)
        pred_cam = pred_cam1 * 0.8 + pred_cam2 * 0.2
        #self.weight1.to(pred_cam1.device)
        #pred_cam2.to(pred_cam1.device)
        #pred_cam = self.weight1 * pred_cam1 + (1 - self.weight1) * pred_cam2
        #[445, B, 128] to [B, 445, 3]
        #pred_3d_coordinates1 = self.xyz_regressor(jv_features_2.transpose(0, 1)) # batch_size X (num_joints + num_vertices) X 3
        #[B 8, f 8, C 445, _ 128]
        jv_features_2 = jv_features_2.permute(1, 0, 2)
        jv_features_joints = jv_features_2[:,:self.num_joints,:]
        jv_features_joints_v1 = jv_features_joints.view(-1, 8, 14, 128)
        jv_features_joints_poseformer = self.Poseformer_ver1(jv_features_joints_v1)# B,14,3 output
        jv_features_joints_poseformer = jv_features_joints_poseformer.view(-1, 14 ,3)
        jv_features_vertices = jv_features_2[:,self.num_joints:,:]
        #[B,431,128]   [B,431,128]
        jv_features_vertices,graph_431_64 = self.Graph(jv_features_vertices)
        jv_features_vertices = jv_features_vertices.to(jv_features_joints.device)
        jv_features_2 = torch.cat([jv_features_joints, jv_features_vertices], dim=1)
        #jv_features_2 = jv_features_2.view(-1, 8, 445, 128)
        pred_3d_coordinates1 = self.xyz_regressor(jv_features_2)
        jv_features_vertices = pred_3d_coordinates1[:,self.num_joints:,:]
        jv_features_joints = pred_3d_coordinates1[:,:self.num_joints,:]
        jv_features_joints = jv_features_joints_poseformer * 0.2+jv_features_joints *0.8
        pred_3d_joints = jv_features_joints
        #[B, 445, 3] to [B, 431, 3]
        #pred_3d_vertices_coarse = pred_3d_coordinates[:,self.num_joints:,:] # batch_size X num_vertices(coarse) X 3
        pred_3d_vertices_coarse = jv_features_vertices
        # coarse-to-intermediate mesh upsampling
        # [B, 431, 3] to  [B, 3, 431] to [B, 3, 1723] to [B, 1723, 3]
        pred_3d_vertices_intermediate = self.coarse2intermediate_upsample(pred_3d_vertices_coarse.transpose(1,2)).transpose(1,2) # batch_size X num_vertices(intermediate) X 3
 
        #pred_3d_vertices_intermediate = self.coarse2intermediate_upsample(pred_3d_vertices_coarse) # batch_size X num_vertices(intermediate) X 3
        # intermediate-to-fine mesh upsampling
        #[B, 1723, 3] to [B, 6890, 3]
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_intermediate, n1=1, n2=0) # batch_size X num_vertices(fine) X 3
        pred_3d_vertices_fine = taubin_smoothing(self.L, pred_3d_vertices_fine, maxiter=10, Labmda=0.75, Mu=-0.5)       
        out = {}
        out['pred_cam'] = pred_cam
        out['pred_3d_joints'] = pred_3d_joints
        out['pred_3d_vertices_coarse'] = pred_3d_vertices_coarse
        out['pred_3d_vertices_intermediate'] = pred_3d_vertices_intermediate
        out['pred_3d_vertices_fine'] = pred_3d_vertices_fine
        out['all_image'] = all_images

 

        # (optional) regress smpl parameters
        if self.args.use_smpl_param_regressor:
            pred_rotmat, pred_betas = self.smpl_parameter_regressor(pred_3d_vertices_intermediate.clone().detach())
            out['pred_rotmat'] = pred_rotmat
            out['pred_betas'] = pred_betas

        return out