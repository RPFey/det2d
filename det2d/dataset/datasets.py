import glob
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from .names import COCO_CLASS_NAME
import random

class COCODataset(object):
    """
    The input dictionary should contain parameters as follow:
    `root`
    `annFile`

    You can find more details of the parameter here :
    We don't need to transform images here
    """

    def __init__(self, config):
        self.root = config['root']
        self.ann_path = config['annFile']
        self.coco_api = COCO(self.ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        self.data_info = self.coco_api.loadImgs(self.img_ids)
        self.class_name = COCO_CLASS_NAME

    def __len__(self):
        return len(self.data_info)

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        id = img_info['id']
        if not isinstance(id, int):
            raise TypeError('Image id must be int.')
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': id}
        return info

    def get_img_annotation(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)

        return anns

    def __getitem__(self, idx):
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.root, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        anns = self.get_img_annotation(idx)
        return img, anns


class VisDrone(object):
    """
    Input configuration dictionary should include:

    `img_path`: path to the image folder --
    `label_path`: path to the label folder
    `verbose`: verbose (for test cases)

    Please refer to the VisDrone.txt under `data` folder for more information.
    """

    def __init__(self, config):
        self.img_path = config['img_path']
        self.label_path = config['label_path']
        self.verbose = config['verbose'] if 'verbose' in config.keys() else False

        self.label_names = sorted(os.listdir(self.label_path))
        self.img_names = sorted(os.listdir(self.img_path))

        print("get samples : ", len(self.label_names))
        assert len(self.label_names) == len(self.img_names), "mismatch of images and labels"

        # for object detection method, just contains objects (1, 4, 5, 6), so we re format the target
        self.cat_ids = [1, 2, 3, 4]
        self.code = {
            1: 1,
            4: 2,
            5: 3,
            6: 4
        }

    def __len__(self):
        return len(self.label_names)

    def get_ids(self):
        return self.cat_ids

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_id = img_name.split('.')[0]
        img_name = os.path.join(self.img_path, img_name)

        img = cv2.imread(img_name)
        img = img[:, :, ::-1]
        assert img is not None, "img read fail"

        label_name = self.label_names[index]
        label_name = os.path.join(self.label_path, label_name)

        if self.verbose:
            print('image name, %s' % (img_name))
            print('label name, %s' % (label_name))

        with open(label_name, 'r') as f:
            lines = f.readlines()

        targets = []
        for line in lines:
            data = [int(x) for x in line.split(',')[:8]]

            if self.verbose:
                print(data)

            if data[5] not in self.code.keys():
                continue
            target = {
                'bbox': np.array(data[:4], dtype=np.float64),
                'image_id': img_id,
                'category_id': self.code[data[5]]
            }
            targets.append(target)

        return img, targets


class KITTI(object):
    """
    Input configuration dictionary should include:

    `img_path`: path to the image folder -- /path/to/kitti/image_2
    `label_path`: path to the label folder -- /path/to/kitti/label_2
    `ids`: train/val id -- /path/to/kitti/val.txt

    class2id = {
        'Misc': 1,
        'Car': 2,
        'Van': 3,
        'Truck': 4,
        'Pedestrian': 5,
        'Person_sitting': 6,
        'Cyclist': 7,
        'Tram': 8,
    }
    """

    def __init__(self, config):
        # parse all the parameters here.
        ids = config['ids']
        img_path = config['img_path']
        label_path = config['label_path']

        self.ids = ids
        self.img_path = img_path
        self.label_path = label_path

        with open(ids, 'r') as f:
            ids_file = f.readlines()
        id_num = [int(id_) for id_ in ids_file]

        img_list = sorted(os.listdir(img_path))
        label_list = sorted(os.listdir(label_path))

        self.img_list = []
        self.label_list = []

        for num in id_num:
            self.img_list.append(img_list[num])
            self.label_list.append(label_list[num])

        print("Get training samples : ", len(self.img_list))

        self.class2id = {
            'Misc': 1,
            'Car': 2,
            'Van': 3,
            'Truck': 4,
            'Pedestrian': 5,
            'Person_sitting': 6,
            'Cyclist': 7,
            'Tram': 8,
        }

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # get image and labels
        img_place = self.img_list[index]
        label_place = self.label_list[index]

        img = cv2.imread(os.path.join(self.img_path, img_place))
        img = img[:, :, ::-1]  # convert to RGB order

        label_file = open(os.path.join(self.label_path, label_place), 'r')
        label = label_file.readlines()
        labels = []
        cls_id = []

        for fields in label:
            fields = fields.strip()
            fields = fields.split(" ")
            cls_name = fields[0]
            cls_id.append(self.class2id[cls_name])
            labels.append(fields[1:])
        labels = np.array(labels)
        cls_id = np.array(cls_id)
        lens = len(labels)

        bbox = np.array(labels[:, [3, 4, 5, 6]])
        bbox = bbox.astype(np.float64)

        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
        targets = []

        for i in range(lens):
            target = {
                "bbox": bbox[i],
                "image_id": img_place,
                "category_id": cls_id[i]
            }
            targets.append(target)

        return img, targets