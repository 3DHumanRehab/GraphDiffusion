# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..  
# *************************************************************************

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, AdaLayerNorm
from diffusers.models.attention import Attention as CrossAttention

from einops import rearrange, repeat
import math
@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        use_graph = None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels
        #self.GCN=GraphResBlock(self.in_channels,self.in_channels)
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.use_graph = use_graph
        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    use_graph = self.use_graph,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,

                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # JH: need not repeat when a list of prompts are given 
        if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
            encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)
        
        #
        residual = hidden_states
        batch, channel, height, weight = hidden_states.shape

        

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
class Graph_rank2_Block(nn.Module):
    def __init__(self):
        super(Graph_rank2_Block, self).__init__()


        # 定义卷积和 GCN 操作
        self.conv1 = nn.Conv2d(1280, 431, kernel_size=1)
        self.GCN_p = Graph_p_ResBlock(16 , 16)  # 你需要定义一个 GCN 模块类
        self.conv3 = nn.Conv2d(431, 1280, kernel_size=1)


    def forward(self, hidden_states):
        video_length = hidden_states.shape[2]
        hidden_states = hidden_states.reshape(-1, 1280, 4, 4)
        hidden_states = self.conv1(hidden_states)
        hidden_states = hidden_states.reshape(-1, 431, 16)
        hidden_states = self.GCN_p(hidden_states)
        hidden_states = hidden_states.reshape(-1, 431, 4, 4)
        # self.conv3.weight.to(hidden_states.cuda())
        # self.conv3.bias.to(hidden_states.cuda())
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states.reshape(-1, 1280, video_length, 4, 4)

        return hidden_states
  

class Graph_rank1_Block(nn.Module):
    def __init__(self):
        super(Graph_rank1_Block, self).__init__()


        # 定义卷积和 GCN 操作
        self.conv1 = nn.Conv2d(1280, 431, kernel_size=1)
        self.GCN = GraphResBlock(16 , 16)  # 你需要定义一个 GCN 模块类
        self.conv3 = nn.Conv2d(431, 1280, kernel_size=1)


    def forward(self, hidden_states):
        video_length = hidden_states.shape[2]
        hidden_states = hidden_states.reshape(-1, 1280, 4, 4)
        hidden_states = self.conv1(hidden_states)
        hidden_states = hidden_states.reshape(-1, 431, 16)
        hidden_states,_ = self.GCN(hidden_states)
        hidden_states = hidden_states.reshape(-1, 431, 4, 4)
        # self.conv3.weight.to(hidden_states.cuda())
        # self.conv3.bias.to(hidden_states.cuda())
        hidden_states = self.conv3(hidden_states)
        hidden_states = hidden_states.reshape(-1, 1280, video_length, 4, 4)

        return hidden_states


class GraphLinear(torch.nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels),requires_grad=True)
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels),requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        # return torch.matmul(self.W[None, :].to(x.device), x) + self.b[None, :, None].to(x.device)
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]
    
class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight.to(x.device) * x + self.bias.to(x.device)

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


class GraphConvolution(torch.nn.Module):
    instance_count = 0
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, mesh='body', bias=True):
        # out_channels=[56*56,56*56,28*28,28*28,14*14,14*14,7*7,14*14,14*14,14*14,28*28,28*28,28*28,56*56,56*56,56*56]
        # super(GraphConvolution, self).__init__()
        # #device=torch.device('cuda')
        # self.in_features = in_features
        # self.out_features = out_features
        # self.num=0
        # if mesh=='body':
        #     adj_indices = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_indices.pt')
        #     adj_mat_value = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_values.pt')
        #     adj_mat_size = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_size.pt')
        #     #adj_indices = self.adjacency_indices
        #     #adj_mat_value = self.adjacency_matrix_value
        #     #adj_mat_size = self.adjacency_matrix_size
        # self.adjmat = torch.sparse_coo_tensor(adj_indices, adj_mat_value, size=adj_mat_size)
        # self.weight1= torch.nn.Parameter(torch.FloatTensor(431,out_channels[GraphConvolution.instance_count]))
        # self.weight = torch.nn.Parameter(torch.FloatTensor(out_channels[GraphConvolution.instance_count], 431))
        # GraphConvolution.instance_count += 1
        # print("+++++++++++++++++++++++++",GraphConvolution.instance_count)
        # if bias:
        #     self.bias = torch.nn.Parameter(torch.FloatTensor(431))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()
        
        super(GraphConvolution, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features

        if mesh=='body':
            adj_indices = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_indices.pt',map_location='cuda')  # 431 
            adj_mat_value = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_values.pt',map_location='cuda')
            adj_mat_size = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/smpl_431_adjmat_size.pt',map_location='cuda')
        elif mesh=='hand':
            adj_indices = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/mano_195_adjmat_indices.pt')
            adj_mat_value = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/mano_195_adjmat_values.pt')
            adj_mat_size = torch.load('/home/zjlab1/workspace/fengzehui/src/modeling/data/mano_195_adjmat_size.pt')

        self.adjmat = torch.sparse_coo_tensor(adj_indices, adj_mat_value, size=adj_mat_size)

        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            support1 = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                # output.append(torch.matmul(self.adjmat, support))
                if self.adjmat.device!=support.device:
                    output.append(spmm(self.adjmat.to(support.device), support))
                    
                    #output.append(spmm(self.adjmat, support))
                else:
                    output.append(spmm(self.adjmat, support))
                support1.append(support)
            
            if output:
                output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias.to(output.device)
            tensor_list = [torch.tensor(array) for array in support1]
            # 将列表中的张量堆叠成一个新的维度
            stacked_tensor = torch.stack(tensor_list)
            return output,stacked_tensor

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Graph_p_ResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, mesh_type='body'):
        super(Graph_p_ResBlock, self).__init__()
        device=torch.device('cuda')
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.lin1 = torch.nn.Linear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, mesh_type)
        # self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.lin2 = torch.nn.Linear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)
        # self.convlayer=torch.nn.Conv2d(in_channels//2, out_channels=431,kernel_size=1).to('cuda')
        # self.convlayer1=torch.nn.Conv2d(in_channels=431, out_channels=out_channels//2 ,kernel_size=1).to('cuda')
    def forward(self, x):
        trans_y = F.relu(self.pre_norm(x))
        y = self.lin1(trans_y)

        y = F.relu(self.norm1(y))
        y, _ = self.conv(y)
        y, _ = self.conv(y)
        

        trans_y = F.relu(self.norm2(y))
        y = self.lin2(trans_y)

        z = x.to(y.device)+y
        return z
    
class GraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, mesh_type='body'):
        super(GraphResBlock, self).__init__()
        device=torch.device('cuda')
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.lin1 = torch.nn.Linear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, mesh_type)
        # self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.lin2 = torch.nn.Linear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels)
        self.norm1 = BertLayerNorm(out_channels // 2)
        self.norm2 = BertLayerNorm(out_channels // 2)
        # self.convlayer=torch.nn.Conv2d(in_channels//2, out_channels=431,kernel_size=1).to('cuda')
        # self.convlayer1=torch.nn.Conv2d(in_channels=431, out_channels=out_channels//2 ,kernel_size=1).to('cuda')
    def forward(self, x):
        # b,c,h,w=x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        # m=x
        # x=x.transpose(1,3) #16,56,56,320
        # trans_y = F.relu(self.pre_norm(x))
        # trans_y=trans_y.transpose(1,3) #16,320,56,56
        # trans_y = trans_y.view(b, c, -1) #16,320,56*56
        # y = self.lin1(trans_y) 
        # y = y.view(b, int(c//2), h, w)
        # y = y.transpose(1,3)
        # y = F.relu(self.norm1(y))
        # y = y.transpose(1,3)
        # y = self.convlayer(y)
        # y = y.view(b, 431, -1)
        # y = self.conv(y)
        # trans_y = y.view(b, 431, h, w) #16,320 56 56
        # trans_y = trans_y.to('cuda')
        # trans_y = self.convlayer1(trans_y)
        # trans_y=trans_y.transpose(1,3)#16, 56 56 320
        # trans_y = F.relu(self.norm2(trans_y))
        # trans_y=trans_y.transpose(1,3)#16, 320,56,56
        # trans_y = trans_y.view(b, int(c//2), -1) #16,320,56*56
        # trans_y = self.lin2(trans_y)
        # y = trans_y.view(b, c, h, w)
        # z = m+y
        trans_y = F.relu(self.pre_norm(x))
        y = self.lin1(trans_y)

        y = F.relu(self.norm1(y))
        y, support = self.conv(y)

        trans_y = F.relu(self.norm2(y))
        y = self.lin2(trans_y)

        z = x.to(y.device)+y
        return z,support
    

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_graph:bool = False,        
        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention

        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            self.attn1 = SparseCausalAttention2D(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        self.use_graph =use_graph
        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)
        self.use_ada_layer_norm_zero = False
        
        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, *args, **kwargs):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention

        norm_hidden_states = (self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states))

        # if self.only_cross_attention:
        #     hidden_states = (
        #         self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
        #     )
        # else:
        #     hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        # pdb.set_trace()
        #norm_hidden_states=GraphResBlock(norm_hidden_states)

        if self.unet_use_cross_frame_attention:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

    


        # Feed-forward torch.Size([64, 3136, 320])
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
