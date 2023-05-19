import torch
import torch.nn as nn
import numpy as np
from .SwinTransformer.SwinTransformer import SwinTransformer
from .ConvNeXt.ConvNeXt import ConvNeXt

class PatchExpand_X2_cel(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, patch_size=[2,4], factor=2):
        super().__init__()
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size 
        self.norm = norm_layer(dim) 
        # W,H,C
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = ( dim // 2 ** i) // factor # 1,out_dim=C/4
            else:
                out_dim = (dim // 2 ** (i + 1)) // factor # 0,out_dim=C/4
            stride = 2
            padding = (ps - stride) // 2 # 0,0;1,1
            self.reductions.append(nn.ConvTranspose2d(dim, out_dim, kernel_size=ps, stride=stride, padding=padding))
            # 0,2W,2H,C/4
            # 1,2W,2H,C/4
    def forward(self, x):
        # B,C,W,H
        x = self.norm(x) 
        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        # B,C/2,2W,2H
        return x

class FinalPatchExpand_X4_cel(nn.Module):
    def __init__(self,dim, norm_layer=nn.BatchNorm2d, patch_size=[4,8,16,32]):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim) 
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
     
        # W,H,C
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = ( dim // 2 ** i) # 1,out_dim=C/2
            else:
                out_dim = (dim // 2 ** (i + 1))  # 0,out_dim=C/2
            stride = 4
            padding = (ps - stride) // 2 # 0,0;1,1
            self.reductions.append(nn.ConvTranspose2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding)) # out=(w−1)×stride−2×p+k
            # 0,4W,4H,C/2
            # 1,4W,4H,C/2
    def forward(self, x):
        """
        x: B,C,H,W
        """
        x = self.norm(x) 
        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1)
        # x: B,C,4H,4W
        return x

def Get_encoder(config, model_name="ConvNeXt"):
    if model_name == "ConvNeXt":
        encoder_model = ConvNeXt(in_chans=3, 
                                depths=[3, 3, 3, 3], 
                                dims=[96, 192, 384, 768], 
                                drop_path_rate=0.1,
                                layer_scale_init_value=1e-6, 
                                out_indices=[0, 1, 2, 3])
    elif model_name == "SwinTransformer":
        encoder_model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                        # img_size=448,
                                        # patch_size = 8,
                                        patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                        in_chans=config.MODEL.SWIN.IN_CHANS,
                                        # num_classes=self.num_classes,
                                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                        depths=config.MODEL.SWIN.DEPTHS,
                                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                                        window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                                        drop_rate=config.MODEL.DROP_RATE,
                                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                        ape=config.MODEL.SWIN.APE,
                                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise AssertionError("Not implemented model")
    return encoder_model

class Attention_Fusion_Module(nn.Module):
    def __init__(self, setup, layer_i, in_dim, out_dim):
        super().__init__()
        self.setup = setup
        self.layer_i = layer_i
        self.in_dim = in_dim
        self.out_dim = out_dim

        ######## fuse ############
        if setup["fuse"] == "AFF":
            self.fuse = AFF(channels=self.in_dim, r=4)
        elif setup["fuse"] == "iAFF":
            self.fuse = iAFF(channels=in_dim, r=4)
        elif setup["fuse"] == "DAF":
            self.fuse = DAF()
        else:
            self.fuse = MyAFF(channels=self.in_dim)

        ######## pxd #########
        if setup["pxd"]:
            if layer_i == 3:
                self.pxd = nn.Sequential(
                                    nn.Conv2d(in_dim*2, in_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),
                                    FinalPatchExpand_X4_cel(dim=in_dim))# final cel x4   # out_dim = in_dim
            elif layer_i == 0:
                self.pxd = nn.Sequential(
                                    nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_dim),
                                    nn.ReLU(),
                                    PatchExpand_X2_cel(dim=in_dim, factor=2),)  # out_dim = in_dim // 2
            else:
                self.pxd = nn.Sequential(
                                    nn.Conv2d(in_dim*2, in_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_dim),
                                    nn.ReLU(),
                                    PatchExpand_X2_cel(dim=in_dim, factor=2),) # out_dim = in_dim // 2
        # conv and upsample        
        else:      
            if layer_i == 3:
                self.pxd = nn.Sequential(
                                    nn.Conv2d(in_dim*2, out_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),
                                    nn.UpsamplingBilinear2d(scale_factor=4),)# final upx4                      
            elif layer_i == 0:
                self.pxd = nn.Sequential(
                                    nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),
                                    nn.UpsamplingBilinear2d(scale_factor=2),)  
            else:   # 1,2
                self.pxd = nn.Sequential(
                                    nn.Conv2d(in_dim*2, out_dim, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(out_dim),
                                    nn.ReLU(),
                                    nn.UpsamplingBilinear2d(scale_factor=2),)

    def forward(self, l_x, g_x, f_out=None, cls_token=None):
        out = self.fuse(l_x, g_x) 
        if f_out is not None:
            out = torch.cat([out, f_out],dim=1)
        out = self.pxd(out)
        return out

class DAF(nn.Module):
    '''
    DirectAddFuse
    '''

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual

class iAFF(nn.Module):
    '''
    iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 1 local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 1 global attention
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 2 local attention
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 2 global attention
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

class AFF(nn.Module):
    '''
    AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class DHUnet(nn.Module):
    def __init__(self,config, Global_branch="SwinTransformer", Local_branch="ConvNeXt", num_classes=1000):
        super().__init__()

        ##### Experiment settings ####
        setup = {
            "fuse" : "DAF", # DAF AFF iAFF MyAFF
            "pxd" : True, # skip-connection patchExpand cel, default upsample
            "deep_supervision":True,  # deep supervision
        }
        self.setup = setup
        #####################

        encoder_depth = len(config.MODEL.SWIN.DEPTHS)
        embed_dim = config.MODEL.SWIN.EMBED_DIM

        self.encoder_depth = encoder_depth
        self.decoder_depth = encoder_depth
        self.embed_dim = embed_dim
        
        # Global and Local encoder branch
        self.L_encoder = Get_encoder(config, Local_branch)
        self.G_encoder = Get_encoder(config, Global_branch)

        # attention fusion decoder
        self.Att_fusion = nn.ModuleList()
        for i in range(encoder_depth):
            input_dim = embed_dim*2**(encoder_depth - i - 1)
            att_fusion = Attention_Fusion_Module(
                                                setup=setup,
                                                layer_i=i,
                                                in_dim=input_dim, 
                                                out_dim=input_dim//2 if i < encoder_depth - 1 else input_dim
                                                )
            self.Att_fusion.append(att_fusion)
        
        # Segmentation Head
        self.segment = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=1, bias=False)

        ######## deep_supervision #########
        self.ds = setup["deep_supervision"]
        if self.ds:
            self.deep_supervision = nn.ModuleList([
                nn.Sequential(nn.Conv2d(embed_dim*4, num_classes, 3, padding=1), nn.Upsample(scale_factor=16)),
                nn.Sequential(nn.Conv2d(embed_dim*2, num_classes, 3, padding=1), nn.Upsample(scale_factor=8)),
                nn.Sequential(nn.Conv2d(embed_dim, num_classes, 3, padding=1), nn.Upsample(scale_factor=4)),
                nn.Sequential(nn.Conv2d(embed_dim, num_classes, 3, padding=1))
            ])
    def forward(self, x_l,x_g):
        if x_l.size()[1] == 1:
            x_l = x_l.repeat(1,3,1,1)
        if x_g.size()[1] == 1:
            x_g = x_g.repeat(1,3,1,1)
        
        # Obtain the intermediate layer features obtained by the encoder model (from bottom to top)
        L_features, local_ape = self.L_encoder(x_l)
        G_features = self.G_encoder(x_g, local_ape)

        assert len(G_features) == len(L_features), "the length of encoder does not match!"

        # deep supervision
        if self.ds:
            self.ds_out = []

        # The decoder fuses the features and restores the image resolution
        for idx in range(self.decoder_depth):
            if idx == 0:
                out = self.Att_fusion[idx](L_features[idx], G_features[idx], None, None)
            else:
                out = self.Att_fusion[idx](L_features[idx], G_features[idx], out, None) 
            if self.ds:
                self.ds_out.append(self.deep_supervision[idx](out))

        # Segmentation Head
        out = self.segment(out)

        if self.ds:
            self.ds_out.append(out)
            return (self.ds_out)[::-1]
        else:
            return out,None,None,None,None

    def load_from(self, config):
        pretrained_path_G = config.MODEL.PRETRAIN_CKPT_G
        pretrained_path_L = config.MODEL.PRETRAIN_CKPT_L
        print(" G_encoder, L_encoder load pretrained weights: ", pretrained_path_L, pretrained_path_G)
        # pretrain
        self.G_encoder.load_from(pretrained_path_G)
        self.L_encoder.load_from(pretrained_path_L)