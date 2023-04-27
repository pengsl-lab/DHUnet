import torch
import torch.nn as nn
from .backbone_BEiT import BEiT 
from mmseg.models.decode_heads import UPerHead

class upernet_BEiT_tiny(nn.Module):
    def __init__(self, in_chans=3, out_chans=10000, pretrained='./pretrained_ckpt/beit_base_patch16_224_pt22k_ft1k.pth'):
        super().__init__()
        self.backbone = BEiT(in_chans=in_chans,
                            patch_size=16,
                            embed_dim=384,
                            depth=12,
                            num_heads=8,
                            mlp_ratio=4,
                            qkv_bias=True,
                            use_abs_pos_emb=True,
                            use_rel_pos_bias=False)
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.decoder = UPerHead(
                        in_channels=[384, 384, 384, 384],
                        in_index=[0, 1, 2, 3],
                        pool_scales=(1, 2, 3, 6),
                        channels=512,
                        dropout_ratio=0.1,
                        num_classes=out_chans,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                        )
        self.upx4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        # self.upx4_trans = nn.ConvTranspose2d(in_channels=out_chans,out_channels=out_chans,kernel_size=4,stride=4)
        # if pretrained:
        #     self.backbone.init_weights(pretrained=pretrained)
    def forward(self, x):
        backbone_out = self.backbone(x)
        out = self.upx4(self.decoder(backbone_out))
        return out

if __name__ == '__main__':
    data = torch.randn(2,3,224,224).cuda()
    a = upernet_BEiT_tiny(3,4).cuda()
    s = a(data)
    print(s.shape)