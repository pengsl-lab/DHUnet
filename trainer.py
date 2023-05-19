import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss,FocalLossV2
from utils import validate_single_patch
from datasets.dataset import DHUnet_dataset

class Structure_Loss(nn.Module):
    def __init__(self, num_classes):
        super(Structure_Loss, self).__init__()
        self.num_classes = num_classes 
        self.ce_loss = CrossEntropyLoss() 
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, outputs, label_batch):  
        loss_ce = self.ce_loss(outputs, label_batch[:].long())
        loss_dice = self.dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_dice + 0.5 * loss_ce
        return loss

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

def validate(args, model, valloader):
    logging.info("{} val iterations per epoch".format(len(valloader)))
    model.eval()
    metric_list = []
    for i_batch, sampled_batch in tqdm(enumerate(valloader)):
        image, label, case_name = sampled_batch["image"], sampled_batch["mask"], sampled_batch['case_name'][0]
        metric_i = validate_single_patch(image, label, model, classes=args.num_classes, test_save_path=None, case=case_name, network=args.network)
        metric_i = np.array(metric_i)
        metric_list.append(metric_i) 

        # mean_metric_i = np.nanmean(metric_i,axis=0)
        # logging.info('idx %d case %s dice %f yc %f acc %f' % (i_batch, case_name, mean_metric_i[0], mean_metric_i[1], mean_metric_i[2]))
    
    metric_array = np.array(metric_list) 
    mean_metric = np.nanmean(metric_array, axis=0) 
    for i in range(1, args.num_classes):
        logging.info('class %d dice %f yc %f acc %f' % (i, mean_metric[i-1][0], mean_metric[i-1][1], mean_metric[i-1][2]))
    
    performance = np.mean(mean_metric, axis=0) 
    logging.info('mean dice %f yc %f acc %f' % (performance[0], performance[1], performance[2]))
    return performance

def trainer_KFold(args, model, snapshot_path, trainloader):
    base_lr = args.base_lr
    num_classes = args.num_classes
    structure_loss = Structure_Loss(num_classes=num_classes) 
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    wandb_log = wandb.init(
        project="DHUnet-Ablation",
        name=args.network + '_' + args.dataset + '_lr' + str(args.base_lr) + '_ep' + str(args.max_epochs),
        config=args,
    )

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["mask"]
            label_batch = label_batch.squeeze(1).cuda() 
            image_batch = image_batch.cuda()
            if args.dataset == 'BCSS': 
                mask_label = label_batch.unsqueeze(1).repeat(1,3,1,1)
                mask_label[mask_label > 0] = 1
                image_batch = image_batch * mask_label
                label_batch[label_batch==5] = 0
            if args.network == "DHUnet":
                outputs0, outputs1, outputs2, outputs3, outputs4 = model(image_batch, image_batch)
            else:
                outputs = model(image_batch)
            if args.network == "DHUnet":
                loss0 = structure_loss(outputs0, label_batch)
                loss1 = structure_loss(outputs1, label_batch)
                loss2 = structure_loss(outputs2, label_batch)
                loss3 = structure_loss(outputs3, label_batch)
                loss4 = structure_loss(outputs4, label_batch)
                loss = 0.5*loss0 + 0.3*loss2 + 0.2*loss4 
            else:
                loss = structure_loss(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
            
            if args.network == "DHUnet":
                logging.info('iteration %d : total_loss : %f ,loss0 : %f ,loss1 : %f ,loss2 : %f ,loss3 : %f ,loss4 : %f' 
                        % (iter_num, loss.item(), loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()))
            else:
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            wandb_log.log({'train':{'loss': loss, 'iter': iter_num, 'learning rate': lr_}})
        save_interval = 5
        if (epoch_num + 1) % save_interval == 0 or epoch_num >= max_epoch - 1:
            if epoch_num >= max_epoch - 1:
                save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            valloader = get_dataloader(args, fold_no=args.fold_no, total_fold=args.total_fold, split = "val", batch_size=1, shuffle = False)
            if len(valloader) > 0:
                model.eval()        
                performance = validate(args, model, valloader)
                wandb_log.log({'val':{'dice': performance[0], 'epoch': epoch_num}})
                wandb_log.log({'val':{'yc': performance[1], 'epoch': epoch_num}})
                wandb_log.log({'val':{'acc': performance[2], 'epoch': epoch_num}})
    return "Training Finished!"

def trainer(args, model, snapshot_path):
    # 9折训练
    trainloader = get_dataloader(args, fold_no=args.fold_no, total_fold=args.total_fold, split = "train", batch_size=args.batch_size, shuffle = True)
    trainer_KFold(args, model, snapshot_path, trainloader)
    return "Finished!"