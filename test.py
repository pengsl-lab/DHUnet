import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import test_single_patch 
from utils import slide_concate # converge patches

from network.SwinTransformer import SwinUnet
from network.TransUnet import TransUnet
from network.ConvNeXt import ConvNeXtUnet
from network.UNet import Unet, Unet2Plus, Unet3Plus
from network.FCN import fcn_resnet50 as FCN
from network.ResUnet import ResUnet, ResUnet2Plus
from network.MedT import axialunet,gated,MedT,logo
from network.TransFuse import TransFuse_S
from network.DHUnet import DHUnet
from network.UperNet import upernet_convnext_tiny as ConvNeXt
from network.DeeplabV3.DeeplabV3 import deeplabv3_resnet50 as DeeplabV3

from trainer import trainer, get_dataloader
from config import get_config

from datasets.dataset import DHUnet_dataset
from utils import calculate_metric_perpatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

# add
parser.add_argument('--network', type=str, default='DHUnet',help='the model of network') 
parser.add_argument('--fold_no', type=int, default=-1, help='the i th fold')
parser.add_argument('--total_fold', type=int, default=5, help='total k fold cross-validation')


parser.add_argument("--opts",help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,nargs='+')
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
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)

def get_dataloader(args, fold_no=0, total_fold=5, split = "train", batch_size=1, shuffle = False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    db_data = DHUnet_dataset(list_dir=args.list_dir, split=split, fold_no=fold_no, total_fold=total_fold)

    logging.info("The length of {} {} set is: {}".format(args.dataset,split,len(db_data)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    dataloader = DataLoader(db_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)    
    return dataloader

def inference(args, model, test_save_path=None):
    split = "test"
    testloader = get_dataloader(args, fold_no=args.fold_no, total_fold=args.total_fold, split = split, batch_size=1, shuffle = False)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    eval_time = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):   
        image, label, case_name = sampled_batch["image"], sampled_batch["mask"], sampled_batch['case_name']
        eval_time += test_single_patch(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name,network=args.network)
        
    logging.info("Eval average time {} s".format(eval_time / len(testloader)))

    eval_save_dir = os.path.join(args.output_dir, split)
    concate_path_txt = os.path.join(args.list_dir, "test_concat.txt")
    logging.info('Start slide concate...')
    slide_concate(test_save_path, eval_save_dir, concate_path_txt)
    
    metric_list = []
    result_gt_dir = os.path.join('data', args.dataset, 'WSI')
    for test_pred_file in os.listdir(eval_save_dir):
        prediction_img = Image.open(os.path.join(eval_save_dir, test_pred_file))
        prediction = np.array(prediction_img, np.uint8) # .resize((prediction_img.size[0]//10, prediction_img.size[1]//10))

        if args.dataset == "Gleason":
            test_pred_file = test_pred_file.replace('.png', '_classimg_nonconvex.png')
        elif args.dataset == "Gleason2018":
            test_pred_file = 'mask_' + test_pred_file
        elif args.dataset == 'BCSS':
            test_pred_file = 'gt_' + test_pred_file
        elif args.dataset == "WSSS4LUAD":
            # add background
            back_ground_mask_dir = os.path.join('data', args.dataset, 'background-mask')
            background_img = Image.open(os.path.join(back_ground_mask_dir, test_pred_file))
            background = np.array(background_img, np.uint8)
            prediction *= background 
            # save result
            prediction_final = Image.fromarray(prediction)
            from utils import savePalette
            savePalette(prediction_final, os.path.join(eval_save_dir, test_pred_file))
            test_pred_file = test_pred_file.replace('.png', '_gt.png')  

        label_img = Image.open(os.path.join(result_gt_dir, test_pred_file))
        label = np.array(label_img, np.uint8) # .resize((label_img.size[0]//10, label_img.size[1]//10))
        if args.dataset == 'BCSS':
            prediction[label == 0] = 0
            label[label==5] = 0
        metric = [] 
        for i in range(1, args.num_classes):
            metric.append(calculate_metric_perpatch(prediction==i, label==i)) 
        
        print(metric)
        mean_metric = np.nanmean(metric, axis=0)
        logging.info('case %s dice %f yc %f acc %f' % (test_pred_file, mean_metric[0], mean_metric[1], mean_metric[2]))
        metric_list.append(np.array(metric))

    metric_array = np.array(metric_list) 
    mean_metric = np.nanmean(metric_array, axis=0)
    for i in range(1, args.num_classes):
        logging.info('class %d dice %f yc %f acc %f' % (i, mean_metric[i-1][0], mean_metric[i-1][1], mean_metric[i-1][2]))
    performance = np.nanmean(mean_metric, axis=0) 
    logging.info('mean dice %f yc %f acc %f' % (performance[0], performance[1], performance[2]))
    logging.info('Test Finished!')
    return "Test Finished!" 

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

    dataset_config = {
        'Synapse': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'AGGC': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_AGGC',
            'num_classes': 6,
            'z_spacing': 1,
        },
        'Liver': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Liver',
            'num_classes': 3,
            'z_spacing': 1,
        },
        'Gleason': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Gleason',
            'num_classes': 5,
            'z_spacing': 1,
        },
        'Gleason2018': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Gleason2018',
            'num_classes': 5,
            'z_spacing': 1,
        },
        'WSSS4LUAD': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_WSSS4LUAD',
            'num_classes': 4,
            'z_spacing': 1,
        },
        'BCSS': { 
            'Dataset': DHUnet_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_BCSS',
            'num_classes': 5,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    if args.network == "DHUnet":
        net = DHUnet(config, num_classes=args.num_classes)
    elif args.network == "SwinUnet":
        net = SwinUnet(config,img_size=args.img_size,num_classes=args.num_classes)
    elif args.network == "TransUnet":
        net = TransUnet(img_size=args.img_size, num_classes=args.num_classes)
    elif args.network == "Unet":
        net = Unet(num_classes=args.num_classes)
    elif args.network == "Unet2Plus":
        net = Unet2Plus(num_classes=args.num_classes)
    elif args.network == "Unet3Plus":
        net = Unet3Plus(num_classes=args.num_classes)
    elif args.network == "FCN":
        net = FCN(aux=False, num_classes=args.num_classes, pretrain_backbone=False)
    elif args.network == "ResUnet":
        net = ResUnet(num_classes=args.num_classes)
    elif args.network == "ResUnet2Plus":
        net = ResUnet2Plus(num_classes=args.num_classes)
    elif args.network == "MedT":
        # net = MedT(img_size=args.img_size, imgchan=3, num_classes=args.num_classes)
        net = gated(img_size=args.img_size, imgchan=3, num_classes=args.num_classes)
    elif args.network == "TransFuse":
        net = TransFuse_S(num_classes=args.num_classes, pretrained=True)
    elif args.network == "ConvNeXtUnet":
        net = ConvNeXtUnet(config=config, num_classes=args.num_classes)
    elif args.network == "ConvNeXt":
        net = ConvNeXt(out_chans=args.num_classes, pretrained='./pretrained_ckpt/convnext_tiny_1k_224.pth') 
    elif args.network == "DeeplabV3":
        net = DeeplabV3(num_classes=args.num_classes, pretrain_backbone=True)
    else:
        raise NotImplementedError("NotImplemented network")

    if args.fold_no == -1:
        args.output_dir = os.path.join(args.output_dir, 'all')
    else:
        args.output_dir = os.path.join(args.output_dir, str(args.total_fold) + 'fold_' + str(args.fold_no)) 

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=net.to(device)
    msg = net.load_state_dict(torch.load(snapshot,map_location=device))
    print("self trained DHUnet ",msg)

    # Calculate FLOPs and parameters
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    from thop import profile
    input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(net, inputs=(input, input))[:2] 
    print('FLOPs:%.2fG'%(flops/1e9))

    snapshot_name = snapshot.split('/')[-1]
    log_folder = args.output_dir + '/test_log_/' + dataset_name
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
