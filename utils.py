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
        yc = metric.binary.jc(pred, gt) # calculate_IoU_binary(pred, gt) # metric.binary.jc(pred, gt) # jaccard == iou
        acc = calculate_Acc_binary(pred, gt)
        return [dice, yc, acc]
    elif pred.sum() == 0 and gt.sum() == 0:
        return [1, 1, 1]
    elif pred.sum() == 0 and gt.sum() > 0: 
        return [0, 0, 0]
    else: # pred.sum() > 0 and gt.sum() == 0:
        return [np.NaN, np.NaN, np.NaN]

def validate_single_patch(image, label, net, classes, test_save_path=None, case=None, network = "DHUnet"):
    label = label.squeeze(0).cpu().detach().numpy() 
    image = image.cuda() 
    net.eval()
    with torch.no_grad():
        if network == "DHUnet":
            out = torch.argmax(torch.softmax(net(image, image)[0], dim=1), dim=1).squeeze(0)
        else:
            out = torch.argmax(torch.softmax(net(image), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    
    if test_save_path is not None:
        save_pred_path = test_save_path + '/' + case[0].split('.')[0] + '.png'
        if not os.path.exists(os.path.dirname(save_pred_path)): 
            os.makedirs(os.path.dirname(save_pred_path)) 
        print(save_pred_path)
        mask = Image.fromarray(np.uint32(prediction))
        mask.save(save_pred_path)
    
    metric = []
    for i in range(1, classes):
        if (label==i).sum() > 0:
            metric.append(calculate_metric_perpatch(prediction==i, label==i)) 
        else:
            metric.append([np.NaN, np.NaN, np.NaN])
    return metric


def test_single_patch(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, network="DHUnet"):
    
    eval_start_time = 0
    eval_end_time = 0
    
    
    print(case, os.path.getsize('data/' + case[0]) / 1024)
    if os.path.getsize('data/'  + case[0]) / 1024 >= 3: 
        image = image.cuda() 
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
        if not os.path.exists(os.path.dirname(save_pred_path)):
            os.makedirs(os.path.dirname(save_pred_path))
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

# Restore the small patch under the IMAGES_PATH path to the original IMAGE_SIZE image
def image_concate(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap):

    image_names = sorted(os.listdir(IMAGES_PATH))

    # The number of rows and columns of the image
    step_size = patch_size - overlap
    step_x_max = int(np.ceil((IMAGE_SIZE[0] - step_size)/step_size)) 
    step_y_max = int(np.ceil((IMAGE_SIZE[1] - step_size)/step_size)) 
    assert step_x_max * step_y_max == len(image_names), "Wrong number of files."
    
    # Define the image stitching function
    to_image = Image.new('L', IMAGE_SIZE) 

    # Loop through and paste each picture to the corresponding position in order
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
    # Different tags are marked with different colors
    savePalette(to_image, IMAGE_SAVE_PATH) 

def savePalette(image_array, save_path):
    mask = image_array.convert("L")
    palette=[]
    for j in range(256):
        palette.extend((j,j,j))    
        palette[:3*10]=np.array([
                                [0, 0, 0], # label 0
                                [0,255,0], # label 1 
                                [0,0,255], # label 2
                                [255,255,0], # label 3 
                                [255,0,0], # label 4
                                [0,255,255],# label 5
                            ], dtype='uint8').flatten()
    mask = mask.convert('P')
    mask.putpalette(palette)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    mask.save(save_path)

