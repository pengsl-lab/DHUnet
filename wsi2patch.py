from PIL import Image
import os
import glob,sys
import argparse
import openslide as slide
import time
import random
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def split_data(data,train_ratio=0.7,test_ratio=0.3):
    # specific the test for BCSS dataset
    if args.dataset == 'BCSS':
        train = []
        test = []
        for image in data:
            image_set = image.split('-')[1]
            if image_set in ['OL', 'LL', 'E2', 'EW', 'GM', 'S3']: # References to the division of the original data set literature
                test.append(image)
            else:
                train.append(image)
        return train, test

    np.random.shuffle(data)
    train_set_size = int(len(data)*train_ratio)
    test_set_size = len(data) - train_set_size

    train = data[:train_set_size]
    test = data[train_set_size:]
    return train,test

def get_bbox(position, dimension, patch_size):
    left,upper = position
    assert left>=0 and upper >=0, position
    right = left + patch_size
    lower = upper + patch_size
    if left + patch_size >= dimension[0]:
        right = dimension[0]-1
        left = right - patch_size
    if upper + patch_size >= dimension[1]:
        lower = dimension[1]-1
        upper = lower - patch_size
        
    return (left, upper, right, lower)

def crop_slide(img, save_image_path, bbox=(0,0,0,0), step=(0, 0), resize=(224,224)):
    img_crop = img.crop(bbox).resize(resize)
    patch_name = "%03d_%03d"%(step[0], step[1])
    out_file = os.path.join(save_image_path, patch_name + ".jpg")
    img_crop = img_crop.convert("RGB")
    img_crop.save(out_file)

def crop_mask(mask, save_mask_path, bbox=(0, 0,0,0), step=(0, 0),resize=(224,224)):
    crop_mask = mask.crop(bbox).resize(resize,Image.NEAREST) 
    patch_name = "%03d_%03d"%(step[0], step[1])
    out_file = os.path.join(save_mask_path, patch_name + ".png")

    if args.dataset == 'Gleason2018':
        # {"background": 4(white), "Benign":0(green), "Gleason3":1(blue), "Gleason4":2(yellow), "Gleason5":3(red)} -> {"background": 0, "Benign":1, "Gleason3":2, "Gleason4":3, "Gleason5":4} 
        label = np.asarray(crop_mask, dtype=np.uint8)
        mask_arr = np.zeros(label.shape, np.uint8)
        mask_arr[label==0] = 1
        mask_arr[label==1] = 2
        mask_arr[label==2] = 3
        mask_arr[label==3] = 4
        mask_arr[label==4] = 0
        crop_mask = Image.fromarray(mask_arr,mode='L')
    else:
        crop_mask = crop_mask.convert("L") 

    crop_mask.save(out_file)

def slide_to_patch(wsi_file, mask_file, args):
    img_bag_folder = os.path.join(args.output, "images")
    anno_bag_folder = os.path.join(args.output, "masks")
    os.makedirs(img_bag_folder, exist_ok=True)
    os.makedirs(anno_bag_folder, exist_ok=True)
    
    step_size = args.step
    st = time.time()

    basename_wsi = wsi_file.split(os.path.sep)[-1]
    img_name = basename_wsi.split('.' + basename_wsi.split('.')[-1])[0]
    basename_mk = mask_file.split(os.path.sep)[-1]
    mask_name = basename_mk.split('.' + basename_mk.split('.')[-1])[0]

    if args.dataset == 'Gleason':
        assert mask_name == img_name + '_classimg_nonconvex', (mask_name, img_name)
    elif args.dataset == 'Gleason2018':
        assert mask_name == 'mask_' + img_name, (mask_name, img_name)
    elif args.dataset == 'WSSS4LUAD':
        assert mask_name == img_name + '_gt', (mask_name, img_name)
    elif args.dataset == 'BCSS':
        assert mask_name == 'gt_' + img_name, (mask_name, img_name)
    else:
        assert mask_name == img_name, (mask_name, img_name)
    
    img_patch_path = os.path.join(img_bag_folder, img_name)
    anno_bag_path = os.path.join(anno_bag_folder, img_name)
    os.makedirs(img_patch_path, exist_ok=True)
    os.makedirs(anno_bag_path, exist_ok=True)

    # read image
    if args.dataset == 'Gleason' or args.dataset == 'Gleason2018' or args.dataset == 'WSSS4LUAD' or args.dataset == 'BCSS':
        img = Image.open(wsi_file)
    else:
        scan = slide.OpenSlide(wsi_file)
        img = scan.read_region((0,0),0,scan.dimensions)

    # read mask
    mask = Image.open(mask_file)
    assert mask.size == img.size, (mask.size, img.size)
    print(mask.size, img.size)
    dimension = img.size

    # step = (dimension[1] - step_size)/step_size
    step_y_max = int(np.ceil((dimension[1] - step_size)/step_size)) 
    step_x_max = int(np.ceil((dimension[0] - step_size)/step_size)) 
    print(step_x_max, step_y_max)

    for i in range(step_x_max): # columns
        for j in range(step_y_max): # rows
            position = (i*step_size, j*step_size)
            bbox = get_bbox(position, dimension, args.patch_size)
            resize = (args.img_size,args.img_size)
            # save patch img
            crop_slide(img, img_patch_path, bbox=bbox, step=(i,j), resize=resize)
            # save patch mask
            crop_mask(mask, anno_bag_path, bbox=bbox, step=(i,j), resize=resize)
        sys.stdout.write('\r Cropped: {}/{}'.format(i+1, step_x_max))
    print(img_name, " spend time: ",time.time()-st)       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate patches from svs slides')
    parser.add_argument('--dataset', type=str, required=True, choices=['Liver','BACH','Gleason','Gleason2018','WSSS4LUAD','BCSS'],help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/Liver/WSI', help='wsi and mask data dir')
    parser.add_argument('--output', type=str, default='data/Liver/', help='output images and masks')
    parser.add_argument('--lists_dir', type=str, default='lists/lists_Liver/', help='lists dir')
    parser.add_argument('--overlap', type=int, default=500)
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=224) # resize to img_size

    # for examples:
    # python wsi2patch.py --dataset Liver --data_dir data/Liver/WSI --output data/Liver --lists_dir lists/lists_Liver/  --lists_dir lists/lists_Liver/ # train:test = 0.8:0.2
    # python wsi2patch.py --dataset WSSS4LUAD --data_dir data/WSSS4LUAD/WSI --output data/WSSS4LUAD --lists_dir lists/lists_WSSS4LUAD/  --lists_dir lists/lists_WSSS4LUAD/ --overlap 224 --patch_size 448 # 0.7:0.3
    # python wsi2patch.py --dataset Gleason --data_dir data/Gleason/WSI --output data/Gleason --lists_dir lists/lists_Gleason/  --lists_dir lists/lists_Gleason/ # 0.7:0.3
    # python wsi2patch.py --dataset Gleason2018 --data_dir data/Gleason2018/WSI --output data/Gleason2018 --lists_dir lists/lists_Gleason2018/  --lists_dir lists/lists_Gleason2018/ --overlap 375 --patch_size 750 # 0.7:0.3
    # python wsi2patch.py --dataset BACH --data_dir data/BACH/WSI --output data/BACH --lists_dir lists/lists_BACH/  --lists_dir lists/lists_BACH/ # 0.8:0.2
    # python wsi2patch.py --dataset BCSS --data_dir data/BCSS/WSI --output data/BCSS --lists_dir lists/lists_BCSS/  --lists_dir lists/lists_BCSS/ --overlap 224 --patch_size 448
    # python wsi2patch.py --dataset BCSS --data_dir data/BCSS/WSI --output data/BCSS --lists_dir lists/lists_BCSS/  --lists_dir lists/lists_BCSS/ --overlap 500 --patch_size 1000

    # The meaning of the label pixel value
    # Liver {"background": 0, "non-tumor":1, "tumor":2} 
    # BACH {"background": 0, 'Benign':1,'in situ':2,'invasive':3}
    # Gleason {"background": 0, "Benign":1, "Gleason3":2, "Gleason4":3, "Gleason5":4}
    # Gleason2018 {"background": 4(white), "Benign":0(green), "Gleason3":1(blue), "Gleason4":2(yellow), "Gleason5":3(red)} -> {"background": 0, "Benign":1, "Gleason3":2, "Gleason4":3, "Gleason5":4} 
    # WSSS4LUAD {White background: 0, Normal tissue: 1, Tumor-associated stroma tissue: 2,Tumor epithelial tissue: 3}
    # BCSS {outside_roi: 0, tumor: 1, stroma: 2, lymphocytic_infiltrate: 3, necrosis_or_debris:4, background or others: 5}    
    
    mask_map = {}
    args = parser.parse_args()
    args.step = args.patch_size - args.overlap
    if args.dataset == 'Liver' or args.dataset == 'BACH':
        wsi_files = glob.glob(os.path.join(args.data_dir, '*.svs'))
        mask_files = glob.glob(os.path.join(args.data_dir, '*.png'))
    elif args.dataset == 'WSSS4LUAD':
        wsi_files = glob.glob(os.path.join(args.data_dir, '*.png'))
        mask_files = glob.glob(os.path.join(args.data_dir, '*_gt.png'))
        for m in mask_files:
            if m in wsi_files:
                wsi_files.remove(m)
    elif args.dataset == 'Gleason':
        wsi_files = glob.glob(os.path.join(args.data_dir, '*.jpg'))
        mask_files = glob.glob(os.path.join(args.data_dir, '*_classimg_nonconvex.png'))  
    elif args.dataset == 'Gleason2018':
        wsi_files = glob.glob(os.path.join(args.data_dir, '*.jpg'))
        mask_files = glob.glob(os.path.join(args.data_dir, '*.png'))   
    elif args.dataset == 'BCSS':
        wsi_files = glob.glob(os.path.join(args.data_dir, '*.png'))
        mask_files = glob.glob(os.path.join(args.data_dir, 'gt_*.png'))
        for m in mask_files:
            if m in wsi_files:
                wsi_files.remove(m)    
    assert len(wsi_files) == len(mask_files), (len(wsi_files), len(mask_files))

    for wsi,mask in zip(sorted(wsi_files), sorted(mask_files)):
        slide_to_patch(wsi, mask, args)
    
    data = os.listdir(os.path.join(args.output, 'images'))
    train, test = split_data(data, 0.7, 0.3) # Liver/BACH = 0.8, 0.2 others 0.7:0.3

    os.makedirs(args.lists_dir, exist_ok=True)
    f_train_images = open(os.path.join(args.lists_dir, 'train_images.txt'), 'w')
    f_train_masks = open(os.path.join(args.lists_dir, 'train_masks.txt'), 'w')
    print(train)
    for file_name in train:
        image_patches = sorted(os.listdir(os.path.join(args.output, 'images', file_name)))
        mask_patches = sorted(os.listdir(os.path.join(args.output, 'masks', file_name)))
        for img_patch, mask_patch in zip(image_patches, mask_patches):
            f_train_images.write(os.path.join(args.output, 'images', file_name, img_patch + '\n'))
            f_train_masks.write(os.path.join(args.output, 'masks', file_name, mask_patch + '\n'))
    
    f_test_images = open(os.path.join(args.lists_dir, 'test_images.txt'), 'w')
    f_test_masks = open(os.path.join(args.lists_dir, 'test_masks.txt'), 'w')
    print(test)
    for file_name in test:
        image_patches = sorted(os.listdir(os.path.join(args.output, 'images', file_name)))
        mask_patches = sorted(os.listdir(os.path.join(args.output, 'masks', file_name)))
        for img_patch, mask_patch in zip(image_patches, mask_patches):
            f_test_images.write(os.path.join(args.output, 'images', file_name, img_patch + '\n'))
            f_test_masks.write(os.path.join(args.output, 'masks', file_name, mask_patch + '\n'))

    f_test_concat_images = open(os.path.join(args.lists_dir, 'test_concat.txt'), 'w')
    for wsi,mask in zip(sorted(wsi_files), sorted(mask_files)):
        basename = wsi.split(os.path.sep)[-1]
        ext = basename.split('.')[-1]
        img_name = basename.split('.' + ext)[0]
        if img_name in test:
            img_size = Image.open(mask).size
            write_line = img_name + '_' + str(img_size[0]) + '_' + str(img_size[1])  + '_' + str(args.patch_size) + '_' + str(args.overlap) + '_' + str(args.img_size) + '\n'
            f_test_concat_images.write(write_line)
    
    f_train_images.close()
    f_train_masks.close()
    f_test_images.close()
    f_test_masks.close()
    f_test_concat_images.close()
