import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BasicBlock(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
        base_width = 64, dilation = 1, norm_layer = None):
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample= None,
        groups = 1, base_width = 64, dilation = 1, norm_layer = None,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(
        self,block, layers,num_classes = 1000, zero_init_residual = False, groups = 1,
        width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
 
    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = stride
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,  planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
 
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
 
    def _forward_impl(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        out.append(x)
        x = self.layer2(x)
        out.append(x)
        x = self.layer3(x)
        out.append(x)
        x = self.layer4(x)
        out.append(x)
        return out
 
    def forward(self, x) :
        return self._forward_impl(x)
    def _resnet(block, layers, pretrained_path = None, **kwargs,):
        model = ResNet(block, layers, **kwargs)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path),  strict=False)
        return model
    
    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3],pretrained_path,**kwargs)
    
    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3],pretrained_path,**kwargs)

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )     
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts
 
    
class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6],num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, 4*self.out_channels, kernel_size=1),
            nn.BatchNorm2d(4*self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out
 
class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=512):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)
        
        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels//2, channels//2, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(channels//2 + channels, channels//2, 1),
            nn.BatchNorm2d(channels//2),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels//4, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )    
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(channels//2 + channels//4, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        
        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels//8, channels//8, 1),
            nn.BatchNorm2d(channels//8),
            nn.ReLU()
        ) 
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(channels//4 + channels//8, channels//8, 1),
            nn.BatchNorm2d(channels//8),
            nn.ReLU()
        )
    
        self.fuse_all = nn.Sequential(
            nn.Conv2d(channels*2 - channels//8, channels//4, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU()
        )
        
    def forward(self, input_fpn):
        x1 = self.PPMHead(input_fpn[-1])
        
        x = nn.functional.interpolate(x1, size=(x1.size(2)*2, x1.size(3)*2),mode='bilinear', align_corners=True)
        x = torch.cat([x, self.Conv_fuse1(input_fpn[-2])], dim = 1)
        x2 = self.Conv_fuse1_(x)
        
        x = nn.functional.interpolate(x2, size=(x2.size(2)*2, x2.size(3)*2),mode='bilinear', align_corners=True)
        x = torch.cat([x, self.Conv_fuse2(input_fpn[-3])], dim = 1)
        x3 = self.Conv_fuse2_(x)  
        
        x = nn.functional.interpolate(x3, size=(x3.size(2)*2, x3.size(3)*2),mode='bilinear', align_corners=True)
        x = torch.cat([x, self.Conv_fuse3(input_fpn[-4])], dim = 1)
        x4 = self.Conv_fuse3_(x)
 
        x1 = F.interpolate(x1, x4.size()[-2:],mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:],mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:],mode='bilinear', align_corners=True)
 
        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))
        
        return x


class UPerNet(nn.Module):
    def __init__(self, in_chans, num_classes):
        super(UPerNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = ResNet.resnet50(replace_stride_with_dilation=[1,2,4])
        self.in_channels = 2048
        self.channels = 512
        self.decoder = FPNHEAD(2048, 512)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, self.num_classes, kernel_size=3, padding=1),
        )
        
    def forward(self, x):
        x = self.backbone(x) 
        x = self.decoder(x)
        
        x = nn.functional.interpolate(x, size=(x.size(2)*4, x.size(3)*4),mode='bilinear', align_corners=True)
        x = self.cls_seg(x)
        return x

if __name__ == '__main__':
    data = torch.randn(2,3,224,224).cuda()
    a = UPerNet(3, 4).cuda()
    s = a(data)
    print(s.shape)