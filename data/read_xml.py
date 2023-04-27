# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""

import xml.etree.ElementTree as ET
import numpy as np
# from scipy.misc import imsave
import cv2
import os
import openslide
from PIL import Image

BACH_label = {
    'benign':1,
    'in situ':2,
    'invasive':3
}

Liver_label = {
    "non-tumor":1,
    "tumor":2
}

def findExtension(directory, extension='.xml'):
    files = []    
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files

def fillImage(image, coordinates,color=255):
   cv2.fillPoly(image, coordinates, color=color)
   return image
 
def readXML_BACH(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    regions = root[0][1].findall('Region')
    labels = []
    coords = []

    for r in regions:
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = BACH_label['benign']
        elif 'in situ' in label.lower():
            label = BACH_label['in situ']
        elif 'invasive' in label.lower():
            label = BACH_label['invasive']
        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            coord += [[x,y]]
        coords += [coord]
    return coords,labels

def readXML_Liver(filename):
    labels = []
    coords = []
    tree = ET.parse(filename)
    root = tree.getroot()
    annos = root.findall('Annotation')
    for anno in annos:
        name = anno.attrib['Name']
        if name not in Liver_label.keys(): # necrosis
            continue
        label = Liver_label[name]  
        regions = anno.findall('Regions/Region')

        for region in regions:
            coord = []
            vertices = region.findall('Vertices/Vertex')
            for v in vertices:
                x = int(v.get('X'))
                y = int(v.get('Y'))
                coord += [[x,y]]
            coords += [coord]
            labels += [label]
    return coords,labels

def saveImage(filename,dims,coordinates,labels,sample=1):
    use_Platte = False
    if use_Platte:
        colors = [(0,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,0)]
        img_size = (dims[1],dims[0],3)

        img = np.zeros(img_size, dtype=np.uint8)
        for c,l in zip(coordinates,labels):
            img = fillImage(img,[np.int32(np.stack(c))],color = colors[l])
    else:
        img_size = (dims[1],dims[0])
        img = np.zeros(img_size, dtype=np.uint8)
        for c,l in zip(coordinates,labels):
            img = fillImage(img,[np.int32(np.stack(c))],color = l)
    
    img = img[::sample,::sample,...]
    Image.fromarray(img).save(filename)
if __name__=='__main__':
    dataset = "BACH" # Liver
    # path to the dataset folder
    WSI_folder =  "data/%s/WSI" % dataset
    gt_thumbnails_folder  = "data/%s/gt_thumbnails" % dataset
    img_thumbnails_folder = "data/%s/img_thumbnails" % dataset

    files = findExtension(WSI_folder)
    for file in files:
        file_name = file[:-4]
        
        print('Reading scan',file_name)
        scan = openslide.OpenSlide(os.path.join(WSI_folder, file_name+'.svs'))
        dims = scan.dimensions
        print('Generating thumbnail ', dims[1],dims[0])
        if dataset == "Liver":
            coords,labels = readXML_Liver(os.path.join(WSI_folder, file))
        else:
            coords,labels = readXML_BACH(os.path.join(WSI_folder, file))
        saveImage(os.path.join(WSI_folder, file_name+'.png'), dims, coords, labels)
    
    