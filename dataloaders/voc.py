from base import BaseDataSet, BaseDataLoader
from base import BaseDataSet_SubCls
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json
import random


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image):
        image = np.asarray(image)
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        image = transforms.functional.to_pil_image(image)
        return image

class PairVOCDataset_StrongWeak(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 21

        self.datalist = kwargs.pop("datalist")
        self.stride = kwargs.pop('stride')

        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(PairVOCDataset_StrongWeak, self).__init__(**kwargs)

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            RandomGaussianBlur(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_transform_weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        prefix = "dataloaders/voc_splits{}".format(self.datalist)
        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_id

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path))
        image_id = self.files[index].split("/")[-1].split(".")[0]
        if self.use_weak_lables:
            label_path = os.path.join(self.weak_labels_output, image_id + ".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        h, w, _ = image.shape

        longside = random.randint(int(self.base_size * 0.8), int(self.base_size * 2.0))
        h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
        image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))

        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        crop_h, crop_w = self.crop_size, self.crop_size
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT, }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        x1 = random.randint(0, w + pad_w - crop_w)
        y1 = random.randint(0, h + pad_h - crop_h)

        image1 = image[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()
        label1 = label[y1:y1 + self.crop_size, x1:x1 + self.crop_size].copy()

        flip1 = False
        if random.random() < 0.5:
            image1 = np.fliplr(image1)
            label1 = np.fliplr(label1)
            flip1 = True
        flip = flip1

        image1_weak_aug = self.train_transform_weak(image1)
        image1_strong_aug = self.train_transform(image1)
        images = torch.stack([image1_weak_aug, image1_strong_aug])

        labels = torch.from_numpy(label1.copy()).unsqueeze(0)

        return images, labels, flip

class PairVoc_StrongWeak(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')
        self.dataset = PairVOCDataset_StrongWeak(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)

        super(PairVoc_StrongWeak, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                                 dist_sampler=dist_sampler)

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 21

        self.datalist = kwargs.pop("datalist")
        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        prefix = "dataloaders/voc_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_id

class VOC(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')

        self.dataset = VOCDataset(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)
        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                  dist_sampler=dist_sampler)

class VOC_SubCls(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255

        sampler_shuffle = kwargs.pop('shuffle')
        num_workers = kwargs.pop('num_workers')

        # self.dataset = VOCDataset(**kwargs)
        self.dataset = VOCDataset_SubCls(**kwargs)
        shuffle = False
        dist_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=sampler_shuffle)
        super(VOC_SubCls, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None,
                                  dist_sampler=dist_sampler)

class VOCDataset_SubCls(BaseDataSet_SubCls):
    def __init__(self, **kwargs):
        self.num_classes = 21
        self.datalist = kwargs.pop("datalist")
        self.palette = pallete.get_voc_pallete(self.num_classes)

        self.label_subcls = kwargs.pop("label_subcls")

        super(VOCDataset_SubCls, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        prefix = "dataloaders/voc_splits{}".format(self.datalist)

        if self.split == "val":
            file_list = os.path.join(prefix, f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(prefix, f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        # return image, label, image_id
        label_path_subcls = os.path.join(self.label_subcls, self.labels[index][1:].split('/')[-1])
        label_subcls = np.asarray(Image.open(label_path_subcls), dtype=np.int32)
        return image, label, label_subcls, image_id


