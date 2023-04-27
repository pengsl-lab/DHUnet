# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}


# 更换维度，convert HWIO to OIHW
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

# 激活函数gelu，relu，swish
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

# # Multi-Head Attention Module，返回的weights是什么？
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis 
        self.num_attention_heads = config.transformer["num_heads"] # num_heads多头数
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads) #每一个att head的dim，例如 768//12
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # QKV,用全连接层Linear生成
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size) # W0
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # [batch_size, num_patches, hidden_size] -> [batch_size, num_patches, num_attention_heads,attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [batch_size, num_patches, num_attention_heads,attention_head_size] -> [batch_size,num_attention_heads,num_patches,attention_head_size]
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # [batch_size, num_patches, hidden_size]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # [batch_size, num_patches, hidden_size]

        # [batch_size, num_patches, hidden_size] -> [batch_size,num_attention_heads,num_patches,attention_head_size]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 多矩阵乘法 a*b*c 与 a*c*d 相乘为a*b*d(后两维作矩阵乘法)
        # -> [batch_size,num_attention_heads,num_patches,num_patches]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Q*K^T
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # /sqrt(d)
        attention_probs = self.softmax(attention_scores)#softmax
        weights = attention_probs if self.vis else None # 如果可视化的话，将注意力权重赋予给weights，方便查案
        attention_probs = self.attn_dropout(attention_probs) #dropout

        # -> [batch_size,num_attention_heads,num_patches,attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer) #softmax后乘以矩阵V

        # -> [batch_size,num_patches,num_attention_heads,attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #内存连续
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [batch_size,num_patches,+,all_head_size]
        # [batch_size,num_patches,all_head_size] 或者说是[batch_size,num_patches,total_embed_dim]
        context_layer = context_layer.view(*new_context_layer_shape) # 完成concat拼接
        attention_output = self.out(context_layer) # 乘以W0全连接层
        # print('attention_output:',attention_output.size()) 
        # print:[24, 196, 768] -> [batch_size,num_patches,hidden_size]
        attention_output = self.proj_dropout(attention_output) #dropout
        return attention_output, weights

# MLP Module
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()
    # 初始化fc1和fc2的权值和偏置
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)   # Linear
        x = self.act_fn(x) # GELU
        x = self.dropout(x) # Dropout
        x = self.fc2(x) # Linear
        x = self.dropout(x) # Dropout
        return x

# Patch embedding ,may be some other operate 从开始 Patch Embedding 到 Transformer encoder 以前的模块都在Embeddings类中
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size) # patch_embeddings，Conv2d 16*16 s16
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size)) # position_embeddings

        self.dropout = Dropout(config.transformer["dropout_rate"]) # Dropout


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2) # (B, hidden, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features
    # 对比原官方vit的forward函数,少了cls_tokens##
    # B = x.shape[0]       
    # cls_tokens = self.cls_token.expand(B, -1, -1) # cls_tokens -> (B,1,768)还是(B,1,768/B)?
    # x = torch.cat((cls_tokens, x), dim=1) #Concat

# Encoder Block * L
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        # 第一部分
        h = x  # 复制一份以便concat拼接
        x = self.attention_norm(x) # LayerNorm
        x, weights = self.attn(x) # Multi-Head Attention 模块，返回的weights是什么？
        x = x + h # 拼接
        # 第二部分 
        h = x # 复制一份以便concat拼接
        x = self.ffn_norm(x) # LayerNorm
        x = self.ffn(x) # MLP
        x = x + h # 拼接
        return x, weights

    # ？？？？？有何作用
    def load_from(self, weights, n_block):
        # print("##########vit_seg_mdeling.Block load from ###############")
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):  # Encoder Block * L 堆叠
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer)) # 深复制，即将被复制对象完全再复制一遍作为独立的新个体单独存在

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states) 
            if self.vis:# vis 参数有什么用???可视化attention模块产生的权重？
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states) # layer norm
        return encoded, attn_weights

# Transformer 由 Embedding 和 Encoder 构成
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids) # x,features = hybrid_model(x)
        #features 来自 "# x,features = hybrid_model(x)"
        # features 是从cnn获得的特征，用于skip-connection跟上采样的特征拼接
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True, # use_batchnorm
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True) # ReLU激活函数

        bn = nn.BatchNorm2d(out_channels) # BatchNorm2d 和 LayerNorm的异同？？？

        super(Conv2dReLU, self).__init__(conv, bn, relu)

#Feature Concatenation, up-conv 2x2 and conv 3x3, ReLU and conv 3x3, ReLU
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1) # skip-connection跳跃连接
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# SegmentationHead（由一个Conv2d和UpsamplingBilinear2d组成)
class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)

# Cascaded Upsampler(CUP)模块
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512 # head_channels ??? 最底层的channel为512，后面依次为(256, 128, 64, 16)
        self.conv_more = Conv2dReLU(
            config.hidden_size, # vit-B的D hidden_size设置为768
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1]) # 例如[512, 256, 128, 64]
        out_channels = decoder_channels #例如 (256, 128, 64, 16)

        if self.config.n_skip != 0: # 根据跳跃链接数量设置是否每一层的skip_channels
            # print(self.config)
            skip_channels = self.config.skip_channels # 例如 skip_channels为[512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        # 例如
        # in_channels : [512, 256, 128, 64]
        # out_channels :[256, 128, 64, 16]
        # skip_channels :[512,256, 64, 0]????，由于R50网络得到的feature分别为28*28*512，56*56*256，56*56*64 shape
        print('in_channels',in_channels)
        print('out_channels',out_channels)
        print('skip_channels',skip_channels)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        #features 来自 "# x,features = hybrid_model(x)"
        # features 是从cnn获得的特征，用于skip-connection跟上采样的特征拼接

        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1) # (B, n_patch, hidden) -> (B, hidden, n_patch)  
        x = x.contiguous().view(B, hidden, h, w) # contiguous()内存连续，(B, hidden, n_patch) -> (B, hidden, h, w)
        
        # U型结构的最底层Conv2dReLU
        x = self.conv_more(x)
        # 一些列上采样解码层DecoderBlocks
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

# Transformer 和 Linear 等最终构成VisionTransformer
class TransUnet(nn.Module):
    def __init__(self, config=CONFIGS['R50-ViT-B_16'], img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(TransUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head # ？？？
        self.classifier = config.classifier # 划分任务
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=num_classes,   #config['n_classes'],
            kernel_size=3, # Unet中的segmenthead选的kernel_size为1
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # ？？
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
    
    #？？？ 加载预训练权重
    def load_from(self, weights):
        print("##########vit_seg_mdeling.VisionTransformer load from ###############")
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


