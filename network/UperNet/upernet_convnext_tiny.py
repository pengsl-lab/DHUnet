import torch
import torch.nn as nn
from .backbone_ConvNeXt import ConvNeXt
from mmseg.models.decode_heads import UPerHead

class upernet_ConvNeXt_tiny(nn.Module):
    def __init__(self, in_chans=3, out_chans=1000, pretrained='./pretrained_ckpt/convnext_tiny_1k_224.pth'):
        super().__init__()
        self.backbone = ConvNeXt(in_chans=in_chans,
                                depths=[3, 3, 9, 3], 
                                dims=[96, 192, 384, 768],
                                out_indices=[0, 1, 2, 3],
                                drop_path_rate=0.4, 
                                layer_scale_init_value=1.0 
                                )
        self.decoder = UPerHead(
                        in_channels=[96, 192, 384, 768],
                        in_index=[0, 1, 2, 3],
                        pool_scales=(1, 2, 3, 6),
                        channels=384,
                        num_classes=out_chans
                        )
        self.upx4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        # self.upx4_trans = nn.ConvTranspose2d(in_channels=out_chans,out_channels=out_chans,kernel_size=4,stride=4)
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
    def forward(self, x):
        backbone_out = self.backbone(x)
        out = self.upx4(self.decoder(backbone_out))
        return out

if __name__ == '__main__':
    data = torch.randn(2,3,224,224).cuda()
    a = upernet_ConvNeXt_tiny(3,4).cuda()
    s = a(data)
    print(s.shape)

# class upernet_ConvNeXt_tiny(nn.Module):
#     def __init__(self, in_chans=3, out_chans=10000, pretrained='./pretrained_ckpt/convnext_tiny_1k_224.pth'):
#         super().__init__()
#         self.backbone = ConvNeXt(in_chans=in_chans,
#                                 depths=[3, 3, 9, 3], 
#                                 dims=[96, 192, 384, 768],
#                                 drop_path_rate=0.2, 
#                                 layer_scale_init_value=1.0,
#                                 out_indices=[0, 1, 2, 3],
#                                 )
#         norm_cfg = dict(type='SyncBN', requires_grad=True)
#         self.decoder = UPerHead(
#                         in_channels=[128, 256, 512, 1024],
#                         in_index=[0, 1, 2, 3],
#                         pool_scales=(1, 2, 3, 6),
#                         channels=512,
#                         dropout_ratio=0.1,
#                         num_classes=out_chans,
#                         norm_cfg=norm_cfg,
#                         align_corners=False,
#                         loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                         )
#         self.upx4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
#         # self.upx4_trans = nn.ConvTranspose2d(in_channels=out_chans,out_channels=out_chans,kernel_size=4,stride=4)
#         if pretrained:
#             self.backbone.init_weights(pretrained=pretrained)
#     def forward(self, x):
#         backbone_out = self.backbone(x)
#         out = self.upx4(self.decoder(backbone_out))
#         return out

# if __name__ == '__main__':
#     data = torch.randn(2,3,224,224).cuda()
#     a = upernet_ConvNeXt_tiny(3,4).cuda()
#     s = a(data)
#     print(s.shape)