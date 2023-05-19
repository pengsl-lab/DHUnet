import os 
import numpy as np
from PIL import Image
import logging
import sys
import random

random.seed(301)
np.random.seed(301)

def count_stats(lists_dir, mask_txt, n_classes = 3, dataset = 'Gleason'):
    # statistical training set
    mask_txt = os.path.join(lists_dir, mask_txt)
    # Statistical class ratio
    seg_counter = np.zeros((n_classes,1))

    with open(mask_txt,'r') as f:
        files = f.readlines()
        for file in files:
            file = file.strip('\n')
            mask = np.array(Image.open(file)) 
            h, w = mask.shape
            for i in range(0, n_classes): 
                sum = (mask == i).sum()
                if i == 0 and sum == (h*w):
                    seg_counter[0] += 1
                    break
                if sum > 0:
                    seg_counter[i] += 1
    logging.info("-"*100)
    logging.info("Under {} path, {} dataset, the distribution ratio of each category: {}".format(mask_txt, dataset, seg_counter.flatten()))
    return seg_counter

def liver_preprocess(lists_dir):
    leave_image = []
    leave_mask = []
    with open(os.path.join(lists_dir, "train_images.txt"),'r') as f:
        image_files = f.readlines()
    with open(os.path.join(lists_dir, "train_masks.txt"),'r') as f:
        mask_files = f.readlines()
        
    for mask, image in zip(mask_files, image_files):
        size = os.path.getsize(image.strip('\n')) / 1024
        # print(image, size)
        if size >= 3:  # 3KB unorganized region no prediction no training
            leave_image.append(image)
            leave_mask.append(mask)
    print(len(leave_mask))
    with open(os.path.join(lists_dir, "train_images_new.txt"),'w') as f:
        f.writelines(leave_image)
    with open(os.path.join(lists_dir, "train_masks_new.txt"),'w') as f:
        f.writelines(leave_mask)

def bach_preprocess(lists_dir):
    leave_image = []
    leave_mask = []
    with open(os.path.join(lists_dir, "train_images.txt"),'r') as f:
        image_files = f.readlines()
    with open(os.path.join(lists_dir, "train_masks.txt"),'r') as f:
        mask_files = f.readlines()
        
    for mask, image in zip(mask_files, image_files):
        size = os.path.getsize(image.strip('\n')) / 1024
        if size >= 3:  # 3KB unorganized region no prediction no training
            leave_image.append(image)
            leave_mask.append(mask)
    print(len(leave_mask))
    with open(os.path.join(lists_dir, "train_images_new.txt"),'w') as f:
        f.writelines(leave_image)
    with open(os.path.join(lists_dir, "train_masks_new.txt"),'w') as f:
        f.writelines(leave_mask)

def gleason_preprocess(lists_dir):
    leave_image = []
    leave_mask = []
    with open(os.path.join(lists_dir, "train_images.txt"),'r') as f:
        image_files = f.readlines()
    with open(os.path.join(lists_dir, "train_masks.txt"),'r') as f:
        mask_files = f.readlines()
        
    for mask, image in zip(mask_files, image_files):
        size = os.path.getsize(image.strip('\n')) / 1024
        msk = np.array(Image.open(mask.strip('\n')))
        if size >= 3 or (msk>0).sum() > 0:  # 3KB unorganized region no prediction no training
            leave_image.append(image)
            leave_mask.append(mask)
    print(len(leave_mask))
    with open(os.path.join(lists_dir, "train_images_new.txt"),'w') as f:
        f.writelines(leave_image)
    with open(os.path.join(lists_dir, "train_masks_new.txt"),'w') as f:
        f.writelines(leave_mask)

def bcss_preprocess(lists_dir):
    leave_image = []
    leave_mask = []
    with open(os.path.join(lists_dir, "train_images.txt"),'r') as f:
        image_files = f.readlines()
    with open(os.path.join(lists_dir, "train_masks.txt"),'r') as f:
        mask_files = f.readlines()
        
    for mask, image in zip(mask_files, image_files):
        msk = np.array(Image.open(mask.strip('\n')))
        if (msk>0).sum() != 0:  # Mask all 0 without training (outside_roi)
            leave_image.append(image)
            leave_mask.append(mask)
    print(len(leave_mask))
    with open(os.path.join(lists_dir, "train_images_new.txt"),'w') as f:
        f.writelines(leave_image)
    with open(os.path.join(lists_dir, "train_masks_new.txt"),'w') as f:
        f.writelines(leave_mask)
if __name__ == "__main__":
    logging.basicConfig(filename="data_count.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    lists_dir = "lists/lists_BCSS"
    n_classes = 6
    dataset = 'BCSS'
    # The first step is to count the distribution ratio of each category
    # Liver 3 BACH 4 Gleason 5 WSSS4LUAD 4 BCSS 6
    train_seg_counter = count_stats(lists_dir, "train_masks.txt", n_classes, dataset)
    test_seg_counter = count_stats(lists_dir, "test_masks.txt", n_classes, dataset)

    if dataset == 'Liver': # Remove the non-organized area in the training set on the basis of the original
        liver_preprocess(lists_dir)
        logging.info("----{}数据集训练集样本平衡后-----".format(dataset))
        count_stats(lists_dir, "train_masks_new.txt", n_classes, dataset)
    if dataset == 'BACH':
        bach_preprocess(lists_dir) # Remove the non-organized area in the training set on the basis of the original
        logging.info("----{}数据集训练集样本平衡后-----".format(dataset))
        count_stats(lists_dir, "train_masks_new.txt", n_classes, dataset)
    if dataset == 'Gleason': # Remove the non-organized area in the training set on the basis of the original
        gleason_preprocess(lists_dir)
        logging.info("----{}数据集训练集样本平衡后-----".format(dataset))
        count_stats(lists_dir, "train_masks_new.txt", n_classes, dataset)    

    if dataset == 'BCSS': # Remove the non-organized area in the training set on the basis of the original
        bcss_preprocess(lists_dir)
        logging.info("----{}数据集训练集样本平衡后-----".format(dataset))
        count_stats(lists_dir, "train_masks_new.txt", n_classes, dataset)  
