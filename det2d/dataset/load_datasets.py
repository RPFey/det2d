import torch
from torch.utils.data import Dataset

import numpy as np
import re
from .datasets import KITTI, COCODataset, VisDrone
from .dataset_utils import letterbox
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch._six import container_abcs, string_classes, int_classes
import cv2
import matplotlib.pyplot as plt

BaseDataset = {
    "COCO": COCODataset,
    "KITTI": KITTI,
    "VisDrone": VisDrone
}

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def create_dataset(config: dict):
    """
    To add other datasets, please follow the rules below.
    The dataset should take a disctionary as input, which includes `img_path`, `label_path` and other keys as parameters.
    Please specify these parameters in the documentation of your own dataset so that we can parse what you want.
    The dictionary must contain the `name` key , which indicates the base dataset. Add the name and dataset in
    the above `BaseDataset` dictionary.

    your dataset should return:
    image in np.array format:
        Note: dtype must be np.uint8 and the channel should be in `RGB` (H, W, 3)
    a dictionary with keys
        "bbox": (top_left_x, top_left_y, width, height)
        "image_id": corresponding image id/name
        "category_id": category_id of the bounding box
                    range (1 ~ nc)
    """
    return BaseDataset[config["name"]](config)


class LoadDataset(Dataset):
    """
    Input configuration dictionary must contain

    `base` : (dictionary) parameters for the base dataset. Please follow the documentation of the function
             `create_dataset` to set up the dictionary.
    `image_size`: (tuple or int) the input image size
    `hyper`: (dictionary) parameters for image augmentation, which must contain:
             `degrees`:  the amplitutde of random rotation degrees for image.
             `translate`:  the amplitutde of random translation for image.
             `scale`: the amplitutde of random scale for image.
             `shear`: random shear scale
    `augment`: whether to augment the image

    The augmentation includes random affine transformation, left-right flip and up-down flip with 0.5 probability.

    Return:
        img: torch.Tensor float, normalied in 0.~1. in RGB order
        label: (6, ) (batch_id, cls_id, x, y, w, h)
        img_id: the name of the img file
        shape: (h0, w0), ((h / h0, w / w0), pad) (original shape, (resize ratio), padding)

    TODO:
        ADD hsv and other augmentation
    """

    def __init__(self, config: dict, mode: str):
        super(LoadDataset, self).__init__()
        self.dataset = create_dataset(config['base'])
        self.img_size = config["image_size"]
        self.augment = config.get("augment", False) and mode == 'val'
        self.mode = mode
        self.hyp = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.mode == 'val' or self.mode == 'test':
            return self.get_val_data(index)
        else:
            while True:
                data = self.get_train_data(index)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    def get_val_data(self, index):
        return self.get_train_data(index)

    def get_train_data(self, index):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.dataset)-1)


class YoloDataset(LoadDataset):
    def __init__(self, config:dict, mode: str):
        super(YoloDataset, self).__init__(config, mode)
        if self.augment:
            assert 'hyper' in config.keys(), 'Please add the parameters for data augmentation !'
            self.hyp = config['hyper']

            self.transforms = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit = self.hyp.get('shift', default=0.02),
                    scale_limit = self.hyp.get('scale', default=0.01),
                    rotate_limit = self.hyp.get('rotate', default=10)
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', min_area=1024, min_visibility=0.7))
        else:
            self.transforms = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='yolo', min_area=256, min_visibility=0.7)
            )

    def get_train_data(self, index):
        img, labels = self.dataset[index]
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        h0, w0 = img.shape[:2]  # original size

        coors = []
        cls_id = []
        img_id = []
        for ann in labels:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.dataset.cat_ids:
                continue
            bbox = [x1, y1, x1 + w / 2, y1 + h / 2]
            if not ann['iscrowd']:
                coors.append(bbox)
                cls_id.append(ann['category_id'])
                img_id.append(ann['image_id'])

        coors = np.array(coors)
        cls_id = np.array(cls_id)
        img_id = np.array(img_id)

        img, ratio, pad = letterbox(img, self.img_size, scaleup=self.augment)
        h, w = img.shape[:2]  # current shape
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # convert width, height
        labels = coors.copy()
        if len(labels.shape) == 2:
            # convert to center coordinate
            labels[:, 0] = ratio[0] * (coors[:, 0] + coors[:, 2] / 2) + pad[0]  # pad width
            labels[:, 1] = ratio[1] * (coors[:, 1] + coors[:, 3] / 2) + pad[1]  # pad height

            # scale in current image
            labels[:, 2] = ratio[0] * coors[:, 2]
            labels[:, 3] = ratio[1] * coors[:, 3]

            # Normalize coordinates 0 - 1 (For Yolo training)
            labels[:, [1, 3]] /= img.shape[0]  # height
            labels[:, [0, 2]] /= img.shape[1]  # width

            transformed = self.transforms(image=img, bboxes=labels)
            img = transformed['image']
            labels = transformed['bbox']

        nL = len(labels)
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 2:] = torch.from_numpy(labels)
            labels_out[:, 1] = torch.from_numpy(cls_id)

        # Convert
        return img, labels_out, img_id, shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


class NanodetDataset(LoadDataset):
    def __init__(self, config: dict, mode: str):
        super(NanodetDataset, self).__init__(config, mode)
        if self.augment:
            assert 'hyper' in config.keys(), 'Please add the parameters for data augmentation !'
            self.hyp = config['hyper']
            self.transforms = A.Compose([
                A.SmallestMaxSize(max_size=min(self.img_size)),
                A.ShiftScaleRotate(
                    shift_limit=self.hyp.get('shift', 0.),
                    scale_limit=self.hyp.get('scale', 0.),
                    rotate_limit=self.hyp.get('rotate', 0.)
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomCrop(height=self.img_size[0], width=self.img_size[1]),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', min_area=256, min_visibility=0.2, label_fields=['gt_labels']))
        else:
            self.transforms = A.Compose(
                [
                    A.LongestMaxSize(max_size=max(self.img_size)),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='coco', min_area=256, min_visibility=0.2, label_fields=['gt_labels'])
            )

    def get_train_data(self, index):
        img, labels = self.dataset[index]
        img_id = self.dataset.data_info[index]['file_name']

        if not isinstance(img, np.ndarray):
            img = np.array(img)
        gt_bboxes = []
        gt_labels = []
        for ann in labels:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.dataset.cat_ids:
                continue
            bbox = [x1, y1, w, h]
            if ann['iscrowd']:
                pass
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.dataset.cat2label[ann['category_id']])

        if gt_bboxes:  # format is x, y, w, h
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        transformed = self.transforms(image=img, bboxes=gt_bboxes, gt_labels=gt_labels)
        img = transformed['image'].float() / 255.
        if img.shape[1:] != self.img_size:
                img_pad = torch.zeros((3, self.img_size[0], self.img_size[1]), dtype=img.dtype)
                img_pad[:, :img.shape[1], :img.shape[2]] = img
        else:
            img_pad = img

        gt_labels = np.array(transformed['gt_labels'])
        gt_bboxes = np.array(transformed['bboxes'])
        if gt_bboxes.dtype != np.float32:
            gt_bboxes = gt_bboxes.astype(np.float32)

        if len(gt_bboxes.shape) != 2:
            gt_bboxes = gt_bboxes[np.newaxis, :]
        gt_bboxes[:, 2:4] = gt_bboxes[:, :2] + gt_bboxes[:, 2:4] / 2 # change to x, y, x_center, y_center

        meta = dict(
            image=img_pad,
            img_info=img_id,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels
        )
        return meta

    def visualize_img(self, idx):
        """
        Visualize the idx training sample
        :param idx:
        :return:
        """

        meta = self.get_train_data(idx)
        img = meta['image'].permute(1, 2, 0).contiguous().numpy().copy()
        img = (img * 255).astype(np.uint8)
        bboxes = meta['gt_bboxes']
        category_ids = meta['gt_labels']

        for bbox, category_id in zip(bboxes, category_ids):
            class_name = self.dataset.class_name[category_id + 1]
            x_min, y_min, cen_x, cen_y = bbox
            x_min, x_max, y_min, y_max = int(x_min), int(2 * cen_x - x_min), int(y_min), int(2 * cen_y - y_min)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=1)
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (0, 255, 0), -1)
            cv2.putText(
                img,
                text=class_name,
                org=(x_min, y_min - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.35,
                color=(0, 0, 255),
                lineType=cv2.LINE_AA,
            )
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.savefig(meta['img_info'])

    @staticmethod
    def collate_fn(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            # TODO: support pytorch < 1.3
            # if torch.utils.data.get_worker_info() is not None:
            #     # If we're in a background process, concatenate directly into a
            #     # shared memory tensor to avoid an extra copy
            #     numel = sum([x.numel() for x in batch])
            #     storage = elem.storage()._new_shared(numel)
            #     out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                # return NanodetDataset.collate_fn([torch.as_tensor(b) for b in batch])
                return batch
            elif elem.shape == ():  # scalars
                # return torch.as_tensor(batch)
                return batch
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: NanodetDataset.collate_fn([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(NanodetDataset.collate_fn(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            transposed = zip(*batch)
            return [NanodetDataset.collate_fn(samples) for samples in transposed]

        from pprint import pprint
        pprint(batch)
        raise TypeError(default_collate_err_msg_format.format(elem_type))



