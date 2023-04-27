import numpy as np
import torch
from medpy import metric
import torch.nn as nn
from PIL import Image
import os
from torch.autograd import Variable
import time

def TimeOnCuda():
    torch.cuda.synchronize()
    return time.time()

# taken from https://github.com/JunMa11/SegLoss/blob/master/test/nnUNetV2/loss_functions/focal_loss.py
class FocalLossV2(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=torch.softmax, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLossV2, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit,dim=1)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
        # print((alpha==0.5).sum()) # 224*224*2 = 100352
        # print(gamma)
        # print(pt.cpu().detach().numpy())
        # print(logpt.cpu().detach().numpy())
        # print(loss.cpu().detach().numpy())
        # print(loss.mean().item())
        # exit(0)
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = 1e-6
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = (input).log() + self.smooth
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, n_classes,weight=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        # print(score.shape, target.shape)
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if self.weight is None:
            self.weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weight[i]
        return loss / self.n_classes

def calculate_IoU_binary(y_pred, y_true):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = 1e-9
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_Dice_binary(y_pred, y_true):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = 1e-9
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

def calculate_F1_binary(pred, true):
    """
    F1 score:
        Accuracy =(TP+TN)/(TP+TN+FP+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Precision*Recall)/(Precision+Recall)
    """
    epsilon = 1e-9
    TP = true * pred
    FP = pred ^ TP
    FN = true ^ TP
    precision = TP.sum() / (TP.sum() + FP.sum() + epsilon)
    recall = TP.sum() / (TP.sum() + FN.sum() + epsilon)
    F1 = (2 * precision * recall) / (precision + recall + epsilon)
    return F1

def calculate_Acc_binary(y_pred, y_true):
    """
    compute accuracy for binary segmentation map via numpy
    """
    w, h = y_pred.shape
    smooth = 1e-9
    acc = (np.sum(y_true == y_pred) + smooth) / (h * w + smooth)
    return acc

def calculate_metric_perpatch(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = calculate_Dice_binary(pred, gt)
        yc = metric.binary.jc(pred, gt) # jaccard == iou 完全一样
        acc = calculate_Acc_binary(pred, gt)
        return [dice, yc, acc]
    elif pred.sum() == 0 and gt.sum() == 0:
        return [1, 1, 1]
    elif pred.sum() == 0 and gt.sum() > 0: 
        return [0, 0, 0]
    else: # pred.sum() > 0 and gt.sum() == 0:
        return [np.NaN, np.NaN, np.NaN]

def validate_single_patch(image, label, net, classes, test_save_path=None, case=None, network = "DHUnet"):
    label = label.squeeze(0).cpu().detach().numpy() # 去掉batch维度[224,224]
    image = image.cuda() # 放到cuda上
    net.eval()
    with torch.no_grad():
        if network == "DHUnet":
            out = torch.argmax(torch.softmax(net(image, image)[0], dim=1), dim=1).squeeze(0)
        else:
            out = torch.argmax(torch.softmax(net(image), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    
    if test_save_path is not None:
        save_pred_path = test_save_path + '/' + case[0].split('.')[0] + '.png'
        if not os.path.exists(os.path.dirname(save_pred_path)): # 检查目录是否存在
            os.makedirs(os.path.dirname(save_pred_path)) # 如果不存在则创建目录
        print(save_pred_path)
        mask = Image.fromarray(np.uint32(prediction))
        mask.save(save_pred_path)
    
    metric = []
    for i in range(1, classes):
        if (label==i).sum() > 0: # 只针对patch含有的类别计算
            metric.append(calculate_metric_perpatch(prediction==i, label==i)) 
        else:
            metric.append([np.NaN, np.NaN, np.NaN])
    return metric


def test_single_patch(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, network="DHUnet"):
    # 记录推理时间
    eval_start_time = 0
    eval_end_time = 0
    
    # label = label.squeeze(0).cpu().detach().numpy() # 去掉batch维度
    print(case, os.path.getsize('data/' + case[0]) / 1024)
    if os.path.getsize('data/'  + case[0]) / 1024 >= 3: # 小于3KB 非组织区域不预测(？？？可能设置的有点大了，试试3)
        image = image.cuda() # 放到 cuda 上
        net.eval()
        with torch.no_grad():
            if network == "DHUnet":
                eval_start_time = TimeOnCuda()
                net_out = net(image, image)[0]
                eval_end_time = TimeOnCuda()
                out = torch.argmax(torch.softmax(net_out, dim=1), dim=1).squeeze(0)
            else:
                eval_start_time = TimeOnCuda()
                net_out = net(image)
                eval_end_time = TimeOnCuda()
                out = torch.argmax(torch.softmax(net_out, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    else:
        prediction = np.zeros((patch_size), np.uint8)
        
    if test_save_path is not None:
        ext = case[0].split('.')[-1]
        save_pred_path = test_save_path + '/' + case[0].split('/', 2)[-1].split(ext)[0] + 'png'
        if not os.path.exists(os.path.dirname(save_pred_path)):#检查目录是否存在
            os.makedirs(os.path.dirname(save_pred_path))#如果不存在则创建目录
        print(save_pred_path)
        mask = Image.fromarray(np.uint32(prediction))
        mask.save(save_pred_path)
    
    return eval_end_time - eval_start_time

def slide_concate(test_save_path, eval_save_dir, concate_path_txt):
    with open(concate_path_txt, 'r') as f:
        eval_slides = f.readlines()
    for eval_slide in eval_slides:
        eval_slide = eval_slide.strip('\n')[::-1].split("_", 5)[::-1]
        print(eval_slide)
        IMAGES_PATH = os.path.join(test_save_path, eval_slide[0][::-1]) # 
        IMAGE_SAVE_PATH = os.path.join(eval_save_dir, eval_slide[0][::-1].split('/')[-1] + '.png') 
        IMAGE_SIZE = int(eval_slide[1][::-1]), int(eval_slide[2][::-1]) # 48000 90000
        patch_size = int(eval_slide[3][::-1]) # 1000
        overlap = int(eval_slide[4][::-1]) # 500
        os.makedirs(os.path.dirname(IMAGE_SAVE_PATH), exist_ok=True)
        print(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap)
        image_concate(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap)
        print("saved path ",IMAGE_SAVE_PATH)

# 将IMAGES_PATH路径下的小patch还原成原始IMAGE_SIZE图片
def image_concate(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap):
    # 获取图片集地址下的所有图片名称
    image_names = sorted(os.listdir(IMAGES_PATH))

    # 图片的行列数
    step_size = patch_size - overlap
    step_x_max = int(np.ceil((IMAGE_SIZE[0] - step_size)/step_size)) # columns(向上取整)
    step_y_max = int(np.ceil((IMAGE_SIZE[1] - step_size)/step_size)) # ceil rows(向上取整) floor 向下取整
    assert step_x_max * step_y_max == len(image_names), "文件数量错误"
    
    # 定义图像拼接函数
    to_image = Image.new('L', IMAGE_SIZE)  # 创建一个新图

    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for x in range(step_x_max):
        for y in range(step_y_max):
            path = IMAGES_PATH + '/' + "%03d_%03d.png"%(x, y)
            from_image = Image.open(path).resize((patch_size, patch_size), Image.NEAREST)
            
            position = [x*step_size, y*step_size]
            if position[0] + patch_size >= IMAGE_SIZE[0]:
                position[0] = IMAGE_SIZE[0] - patch_size - 1
            if position[1] + patch_size >= IMAGE_SIZE[1]:
                position[1] = IMAGE_SIZE[1] - patch_size - 1
            to_image.paste(from_image, position)
    # to_image.save(IMAGE_SAVE_PATH)
    savePalette(to_image, IMAGE_SAVE_PATH) # colors

def savePalette(image_array, save_path):
    mask = image_array.convert("L")
    palette=[]
    for j in range(256):
        palette.extend((j,j,j))    
        palette[:3*10]=np.array([
                                [0, 0, 0], # 黑色非组织区域 label 0
                                [0,255,0], # 绿色 label 1 
                                [0,0,255], # 蓝色：label 2
                                [255,255,0], # 黄色 label 3 
                                [255,0,0], # 红色：label 4
                                [0,255,255],# 淡蓝色：label 5
                            ], dtype='uint8').flatten()
    mask = mask.convert('P')
    mask.putpalette(palette)
    if not os.path.exists(os.path.dirname(save_path)):#检查目录是否存在
        os.makedirs(os.path.dirname(save_path))#如果不存在则创建目录
    mask.save(save_path)

