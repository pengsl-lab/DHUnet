from cv2 import norm
from numpy import pad
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .convNet import Block,ConvNeXt,LayerNorm
from functools import partial
from einops import rearrange

class ConvNeXtUnet(nn.Module):
    def __init__(self,config, in_chans=3, num_classes=9, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                drop_path_rate=0.1, layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                **kwargs):
        super().__init__()
        depths = [3,3,3,3]
        self.encoder = ConvNeXt(in_chans=in_chans, depths=depths, dims=dims, 
                                drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                                 out_indices=out_indices)
        self.decoder = Unet_Decoder3(embed_dim=dims[0],depths=depths,drop_path_rate=drop_path_rate,  
                                          layer_scale_init_value=layer_scale_init_value,num_classes=num_classes)                             
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print(x.size())
        x,features = self.encoder(x)
        logits = self.decoder(x,features)
        
        return logits

    def load_from(self, config):
        import copy
        # pretrained_path = config.MODEL.PRETRAIN_CKPT
        pretrained_path = "./pretrained_ckpt/convnext_tiny_1k_224.pth"
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            print("---start load pretrained modle---")
            pretrained_dict = pretrained_dict['model']
            model_dict = self.state_dict()
            with open('pretrained_dict.txt','w') as f:
                for k, v in sorted(pretrained_dict.items()):
                    f.write(k + '\n')
            with open('model_dict.txt','w') as f:
                for k, v in sorted(model_dict.items()):
                    f.write(k + '\n')

            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "norm" != k[:4] and "head" != k[:4]:
                    encoder_k = "encoder." + k
                    full_dict.update({encoder_k:v})  
                    
                    if "stages.2" in k:
                        num = int(k.split(".",3)[2])
                        if (num + 1)%3 == 0:
                            print(num)
                            divnum = (num + 1) // 3
                            encoder_k = "encoder.stages.2." + str(divnum) + "." + k.split(".",3)[-1]
                            print(encoder_k)
                            full_dict.update({encoder_k:v})  

                    if "stages" in k:
                        num = 3 - int(k[7:8])
                        decoder_k = "decoder.layers_up." + str(num) + ".blocks" + k[8:]
                        # print("---",decoder_k)
                        full_dict.update({decoder_k:v})  

            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
                    else:
                        # print("{} matched!".format(k))
                        continue
                else:
                    del full_dict[k]

            with open('full_dict.txt','w') as f:
                for k, v in sorted(full_dict.items()):
                    f.write(k + '\n')
            msg = self.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")

class PatchExpand(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, dim_scale=2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Conv2d(in_channels=dim,out_channels=2*dim,kernel_size=1,bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # print("dim:",self.dim)
        x = self.expand(x) 
        B,C,H,W = x.shape
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C//4)
        # print(x.shape)
        x= self.norm(x) # norm_layer
        return x

class PatchExpandCel(nn.Module):
    def __init__(self, dim,norm_layer=nn.LayerNorm, patch_size=[2,4], input_resolution=[],dim_scale=2, num_input_patch_size=1):
        super().__init__()
        
        self.dim = dim
        self.norm = norm_layer(dim)# channel first

        self.reductions = nn.ModuleList()
        self.patch_size = patch_size # [2,4]
        # W/8,H/8,2C
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = ( dim // 2 ** i) // 2 # 1,out_dim=C/2
            else:
                out_dim = (dim // 2 ** (i + 1)) // 2 # 0,out_dim=C/2
            stride = 2
            padding = (ps - stride) // 2 # 0,0;1,1
            self.reductions.append(nn.ConvTranspose2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding))
            # 0,W/4,H/4,2C
            # 1,W/4,H/4,2C
    def forward(self, x):
        """
        x: B, C, H, W
        """
        # print(x.shape)
        # print(self.dim)
        x = self.norm(x)

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x) 
            xs.append(tmp_x)
        x = torch.cat(xs, dim=1) # B, C, H, W
        #print(x.size())
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self,dim,norm_layer=nn.LayerNorm, dim_scale=4):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Conv2d(in_channels=dim,out_channels=16*dim,kernel_size=1,bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B,C,H,W
        """
        x = self.expand(x) # [B, 16C,H, W]
        _,C,_,_ = x.shape
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)
        return x

class FinalPatchExpand_X4_cel(nn.Module):
    def __init__(self,dim,norm_layer=nn.LayerNorm, patch_size=[4,8], input_resolution=[], dim_scale=4, num_input_patch_size=1):
        super().__init__()

        self.dim = dim
        self.norm = norm_layer(dim) # channel first

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
                                                stride=stride, padding=padding))
            # 0,4W,4H,C
            # 1,4W,4H,C
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
        return x

class Conv2dAct(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_layernorm=True, # use_layernorm
            use_GELU=False
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_layernorm),
        )
        if use_GELU:
            act = nn.GELU()
        else:
            act = nn.ReLU(inplace=True)

        if not use_layernorm:
            norm = nn.BatchNorm2d(out_channels) 
        else:
            norm = use_layernorm(out_channels)

        super(Conv2dAct, self).__init__(conv, norm, act)
#Feature Concatenation, up-conv 2x2 and conv 3x3, ReLU and conv 3x3, ReLU
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_layernorm = None
    ):
        super().__init__()
        self.conv1 = Conv2dAct(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_layernorm=use_layernorm,
            use_GELU=False
        )
        self.conv2 = Conv2dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_layernorm=use_layernorm,
            use_GELU=False
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up = nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels//2,kernel_size=2,stride=2)

    def forward(self, x, skip=None):
        # x = self.up(x) 
        if skip is not None:
            x = torch.cat([x, skip], dim=1) 
        # print(x.shape)
        # print("309:",x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = self.up(x) 

        return x

class Final3DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_layernorm = None
    ):
        super().__init__()
        self.conv1 = Conv2dAct(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_layernorm=use_layernorm,
            use_GELU=False
        )
        self.conv2 = Conv2dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_layernorm=use_layernorm,
            use_GELU=False
        )
        
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.upx4 = nn.ConvTranspose2d(in_channels=out_channels,out_channels=out_channels,kernel_size=4,stride=4)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1) 

        x = self.conv1(x)
        x = self.conv2(x)

        return x
class Unet_Decoder3(nn.Module):
    def __init__(self, embed_dim=96, depths=[3,3,9,3], drop_path_rate=0.1,num_classes=1000,
                 norm_layer=nn.LayerNorm,use_checkpoint=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim 
        self.depths = depths
        self.num_layers = len(depths)
        self.num_classes = num_classes
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")

        self.upBlocks = nn.ModuleList()
        for i_layer in range(len(self.depths)-1):
            in_dim = embed_dim*2**(self.num_layers-1-i_layer)
            upblock = DecoderBlock(in_channels=in_dim,out_channels=in_dim//2,skip_channels=in_dim,use_layernorm=norm_layer)
            self.upBlocks.append(upblock)

        upblock = Final3DecoderBlock(in_channels=embed_dim,out_channels=embed_dim,skip_channels=embed_dim,use_layernorm=norm_layer)
        self.upBlocks.append(upblock)

        self.norm_encoder = norm_layer(self.embed_dim * 2 ** (self.num_layers-1))
        self.norm_up= norm_layer(self.embed_dim) 

        self.upx4 = FinalPatchExpand_X4(dim=embed_dim,norm_layer=norm_layer)
        self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        # print("330:",x.shape)
        x = self.norm_encoder(x)
        for inx, layer_up in enumerate(self.upBlocks):
            # if inx < len(self.upBlocks)-1:  
            # print("516:",x.shape,x_downsample[3-inx].shape)
            x = layer_up(x,x_downsample[3-inx]) 
        x = self.norm_up(x)
        return x

    def forward(self, x, x_downsample):
        x = self.forward_up_features(x,x_downsample) # #Dencoder and Skip connection
        # print("338:",x.shape)
        x = self.upx4(x)
        # print("340:",x.shape)
        x = self.output(x) 
        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, depth, drop_path=0.,
                norm_layer=nn.LayerNorm, upsample=None,use_checkpoint=False,layer_scale_init_value=1e-6,
                input_resolution=[],num_heads=0, window_size=0,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path_global=0.):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint                 

        # build blocks
        self.blocks = nn.ModuleList(
            [Block(dim=dim,drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale_init_value=layer_scale_init_value) 
                        for i in range(depth)])
        
        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x
class ConvNeXt_Decoder(nn.Module):
    def __init__(self, embed_dim=96, depths=[3,3,9,3], drop_path_rate=0.1,num_classes=1000,
                 norm_layer=nn.LayerNorm,use_checkpoint=False, **kwargs):
                
        super().__init__()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        depths = [3,3,3,3]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        # build decoder layers
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        # decoder_depth = [3,3,9,3] 
        for i_layer in range(self.num_layers):# num_layers = 4
            # concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),int(embed_dim*2**(self.num_layers-1-i_layer)))
            concat_linear = nn.Conv2d(in_channels=2*int(embed_dim*2**(self.num_layers-1-i_layer)),
                            out_channels=int(embed_dim*2**(self.num_layers-1-i_layer)),
                            kernel_size=3,stride=1,padding=1)# if i_layer > 0 else nn.Identity() 

            if i_layer == 0:
                layer_up = PatchExpandCel(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                       norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                depth=depths[(self.num_layers-1-i_layer)],
                                drop_path=dp_rates[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpandCel if (i_layer < self.num_layers - 1) else FinalPatchExpand_X4_cel, 
                                use_checkpoint=use_checkpoint)
            
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)
        
        self.norm_encoder = norm_layer(self.embed_dim * 2 ** (self.num_layers-1))
        self.norm_up= norm_layer(self.embed_dim) 

        # self.upx4 = FinalPatchExpand_X4_cel(dim=embed_dim,norm_layer=norm_layer)
        self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        x = self.norm_encoder(x)
        for inx, layer_up in enumerate(self.layers_up):
            # print(x.shape,x_downsample[3-inx].shape)
            x = torch.cat([x,x_downsample[3-inx]],1)
            x = self.concat_back_dim[inx](x) 
            # print("300:",x.shape)
            x = layer_up(x) 
            
        x = self.norm_up(x)
        return x

    def forward(self, x, x_downsample):
        x = self.forward_up_features(x,x_downsample) # #Dencoder and Skip connection
        # x = self.upx4(x) 
        x = self.output(x) 
        return x
