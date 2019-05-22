import matplotlib.pyplot as plt
from sys import path
#base_path = "/home/ph/taotao/Mask_RCNN-master"
path.append("F:/githubProject/Mask_RCNN/")
#base_path = os.path.dirname(os.path.realpath(__file__))
#os.chdir(base_path)

from config import Config
import model as modellib, utils
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import csv
import io
import cv2
from PIL import Image

class FashionConfig(Config):  #
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "fashion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 46  # COCO has 80 classes
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

def get_json(filename):
    with io.open(filename, "r", encoding='utf-8') as f:
        return json.load(f)


def decoding_mask(height, width, encode):
    """
    :param height:
    :param width:
    :param encode: mask
    :return: mask image
    """
    mask = np.zeros([height * width], dtype=np.uint8)
    pixel_list = list(map(int, encode.split(" ")))
    pixel = np.array(pixel_list).reshape((len(pixel_list) // 2, 2))
    for pair in pixel:
        mask[pair[0]:pair[0] + pair[1]] = 1
    return mask.reshape((width, height)).transpose((1, 0))  #


class FashionDataset(utils.Dataset):

    def load_fashion(self, root_dir, dataset_split, split_id):  # 
        train_dir = os.path.join(root_dir, "train.csv")
        label_dir = os.path.join(root_dir, "label_descriptions.json")
        # Add classes
        class_info = get_json(label_dir)
        class_info = class_info["categories"]
        for info in class_info:
            self.add_class("fashion", info["id"] + 1, info["name"])

        # split train / val

        dataset_split = np.array(dataset_split)
        sum = np.sum(dataset_split)
        min_id = np.sum(dataset_split[0:split_id])
        max_id = np.sum(dataset_split[0:split_id + 1])

        # Add images
        info = pd.read_csv(train_dir)
        for i, (name, group) in enumerate(info.groupby("ImageId")):
            id = i % sum
            if id < min_id or id >= max_id:
                continue
            group = group.values.tolist()
            self.add_image("fashion", image_id=name, path=os.path.join(root_dir, "train", name),
                           group=group)

    def load_image(self, image_id):  #
        # info = self.image_info[image_id]
        # image = info
        image = super(FashionDataset, self).load_image(image_id)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fashion":
            return info["fashion"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        group = info["group"]
        # make class list
        class_id = []
        for line in group:
            id = line[4].split("_")[0]
            class_id.append(int(id) + 1)
        class_id = np.array(class_id)
        # make mask
        count = len(group)
        mask_list = []
        for line in group:
            mask = decoding_mask(line[2], line[3], line[1])
            # img = plt.imshow(mask)
            # plt.axis("off")
            # plt.show()
            mask_list.append(mask)
        #???? 为什么在这里转置了  
        #可以简单的认为，网络中读取mask_list的一列为一个mask
        mask_list = np.array(mask_list).reshape([count, group[0][2], group[0][3]]).transpose(1, 2, 0)
        
        # 避免同一个像素被识别为多个实体
        occlusion = np.logical_not(mask_list[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask_list[:, :, i] = mask_list[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask_list[:, :, i]))
        return mask_list.astype(np.bool), class_id.astype(np.int32)

# dataset_train.load_mask(0)
# img = plt.imshow(dataset_train.load_image(0))
# plt.axis("off")
# plt.show()
# print("")
