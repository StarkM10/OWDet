import os
import random
from math import ceil

import torch
from torch.utils.data import Dataset
from PIL import Image

random.seed(42)
class CustomDataset(Dataset):
    def __init__(self, txt_dir, transform=None, test = False, unk_test = False, task = None):
        self.transform = transform
        self.test = test
        self.data = []
        self.t1know_set = [42, 64, 34, 54, 55, 33, 24, 66, 47, 45, 43, 23, 3, 20, 74, 15, 41, 25, 16, 53]
        self.t2know_set = [42, 64, 34, 54, 55, 33, 24, 66, 47, 45, 43, 23, 3, 20, 74, 15, 41, 25, 16, 53, 8, 38, 75, 63, 49, 11, 30, 32, 39, 17, 36, 62, 78, 46, 12, 35, 26, 13, 37, 0]
        self.t3know_set = [42, 64, 34, 54, 55, 33, 24, 66, 47, 45, 43, 23, 3, 20, 74, 15, 41, 25, 16, 53, 8, 38, 75, 63, 49, 11, 30, 32, 39, 17, 36, 62, 78, 46, 12, 35, 26, 13, 37, 0,57,59,70,73,52,2,21,14,67,22,6,60,28,19,76,44,1,10,29,4]
        self.label_to_number = {'refrigerator': 0, 'hot dog': 1, 'baseball bat': 2, 'horse': 3, 'cake': 4, 'fork': 5, 'banana': 6, 'toothbrush': 7, 'truck': 8, 'vase': 9, 'pizza': 10, 'bench': 11, 'suitcase': 12, 'toaster': 13, 'skateboard': 14, 'pottedplant': 15, 'train': 16, 'giraffe': 17, 'spoon': 18, 'orange': 19, 'motorbike': 20, 'baseball glove': 21, 'tennis racket': 22, 'dog': 23, 'car': 24, 'sofa': 25, 'oven': 26, 'knife': 27, 'sandwich': 28, 'donut': 29, 'elephant': 30, 'mouse': 31, 'bear': 32, 'bus': 33, 'bird': 34, 'microwave': 35, 'backpack': 36, 'sink': 37, 'traffic light': 38, 'zebra': 39, 'hair drier': 40, 'sheep': 41, 'aeroplane': 42, 'diningtable': 43, 'carrot': 44, 'cow': 45, 'tie': 46, 'chair': 47, 'bowl': 48, 'parking meter': 49, 'scissors': 50, 'keyboard': 51, 'kite': 52, 'tvmonitor': 53, 'boat': 54, 'bottle': 55, 'toilet': 56, 'frisbee': 57, 'bed': 58, 'skis': 59, 'apple': 60, 'teddy bear': 61, 'umbrella': 62, 'stop sign': 63, 'bicycle': 64, 'wine glass': 65, 'cat': 66, 'surfboard': 67, 'clock': 68, 'laptop': 69, 'snowboard': 70, 'cup': 71, 'book': 72, 'sports ball': 73, 'person': 74, 'fire hydrant': 75, 'broccoli': 76, 'remote': 77, 'handbag': 78, 'cell phone': 79, 'unknown':80}
        all_txt_file = os.listdir(txt_dir)
        for txt_file in all_txt_file:
            if txt_file[:-4] not in self.label_to_number:
                continue
            if unk_test and self.label_to_number[txt_file[:-4]] in self.t1know_set :
                continue
            txt_file_path = os.path.join(txt_dir,txt_file)
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 5000:
                    lines = lines[:5000]
                num_lines = len(lines)
                if not self.test:
                    num_lines_80_percent = ceil(num_lines * 0.8)
                    lines = lines[:num_lines_80_percent]
                if self.test:
                    num_lines_20_percent = ceil(num_lines * 0.2)
                    lines = lines[-num_lines_20_percent:]
                    if len(lines) > 50:
                        lines = random.sample(lines, 50)
                for line in lines:
                    one_line= line.strip().split()  # 假设txt文件中以空格分隔图像名和bbox
                    image_name = one_line[0]
                    bbox = one_line[-4:]
                    self.data.append((txt_file[:-4],image_name, bbox))
                print("load "+str(len(lines))+" pic from "+txt_file)
        print("total_num  "+ str(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_num = ''
        cls_name, image_name, bbox = self.data[idx]
        number_label = self.label_to_number[cls_name]
        bbox = list(map(float, bbox))
        image_path = os.path.join('/media/D/datasets/VOC2007/JPEGImages',image_name+".jpg")
        # image_path = os.path.join('/mnt/nfs/data/home/1120220291/lwy/datas/JPEGImages',image_name+".jpg")
        image = Image.open(image_path).convert('RGB')
        cropped_image = image.crop(bbox)

        if self.transform:
            finial_image = self.transform(cropped_image)
        else:
            finial_image = cropped_image
            # f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
        # 在这里你可以根据需要进行图像预处理操作，如缩放、裁剪、归一化等
        # 然后将图像和边界框信息返回
        return finial_image, number_label, image_name, bbox[0], bbox[1], bbox[2], bbox[3]
