import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from network.DHUnet import DHUnet
from network.SwinTransformer import SwinUnet
from network.TransUnet import TransUnet
from network.ConvNeXt import ConvNeXtUnet
from network.UNet import Unet, Unet2Plus, Unet3Plus
from network.FCN import fcn_resnet50 as FCN
from network.ResUnet import ResUnet, ResUnet2Plus
from network.MedT import axialunet,gated,MedT,logo
from network.TransFuse import TransFuse_S
from network.UperNet import upernet_ConvNeXt_tiny as ConvNeXt
from network.DeeplabV3.DeeplabV3 import deeplabv3_resnet50 as DeeplabV3

from trainer import trainer
from trainer import trainer_KFold
from config import get_config
import logging
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/wl/lian/Medical_Image/transUnet/data/Synapse', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='model',help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=301, help='random seed')
parser.add_argument('--cfg', type=str, required=False, default='configs/MOA_tiny_patch4_window14_224.yaml',metavar="FILE", help='path to config file', )

# add
parser.add_argument('--network', type=str, default='DHUnet',help='the model of network')  
parser.add_argument('--fold_no', type=int,default=-1, help='the i th fold')
parser.add_argument('--total_fold', type=int,default=5, help='total k fold cross-validation')


parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Liver': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Liver',
            'num_classes': 3,
        },
        'BACH': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_BACH',
            'num_classes': 4,
        },
        'Gleason': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Gleason',
            'num_classes': 5,
        },
        'Gleason2018': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_Gleason2018',
            'num_classes': 5,
        },
        'WSSS4LUAD': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_WSSS4LUAD',
            'num_classes': 4,
        },
        'BCSS': {
            'root_path': args.root_path,
            'list_dir': './lists/lists_BCSS',
            'num_classes': 5,
        },
    }

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    
    if args.fold_no == -1:
        args.output_dir = os.path.join(args.output_dir, 'all')
    else:
        args.output_dir = os.path.join(args.output_dir, str(args.total_fold) + 'fold_' + str(args.fold_no))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.network == "DHUnet":
        net = DHUnet(config, num_classes=args.num_classes).cuda()
        net.load_from(config)
    elif args.network == "SwinUnet":
        net = SwinUnet(config,img_size=args.img_size,num_classes=args.num_classes).cuda()
        net.load_from(config)
    elif args.network == "TransUnet":
        net = TransUnet(img_size=args.img_size, num_classes=args.num_classes).cuda()
        net.load_from(weights=np.load("./pretrained_ckpt/R50+ViT-B_16.npz"))
    elif args.network == "Unet":
        net = Unet(num_classes=args.num_classes).cuda()
    elif args.network == "Unet2Plus":
        net = Unet2Plus(num_classes=args.num_classes).cuda()
    elif args.network == "Unet3Plus":
        net = Unet3Plus(num_classes=args.num_classes).cuda()
    elif args.network == "FCN":
        net = FCN(aux=False, num_classes=args.num_classes, pretrain_backbone=False).cuda()
    elif args.network == "ResUnet":
        net = ResUnet(num_classes=args.num_classes).cuda()
    elif args.network == "ResUnet2Plus":
        net = ResUnet2Plus(num_classes=args.num_classes).cuda()
    elif args.network == "MedT":
        # BCSS gated
        net = gated(img_size=args.img_size, imgchan=3, num_classes=args.num_classes).cuda()
        # other MedT
        # net = MedT(img_size=args.img_size, imgchan=3, num_classes=args.num_classes).cuda()
    elif args.network == "TransFuse":
        net = TransFuse_S(num_classes=args.num_classes, pretrained=True).cuda()
    elif args.network == "ConvNeXtUnet":
        net = ConvNeXtUnet(config=config, num_classes=args.num_classes).cuda()
        net.load_from(config)
    elif args.network == "ConvNeXt":
        net = ConvNeXt(out_chans=args.num_classes, pretrained='./pretrained_ckpt/convnext_tiny_1k_224.pth').cuda()
    elif args.network == "DeeplabV3":
        net = DeeplabV3(num_classes=args.num_classes, pretrain_backbone=True).cuda()
    else:
        raise NotImplementedError("NotImplemented network")

    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    print(str(net))
    trainer(args, net, args.output_dir)
