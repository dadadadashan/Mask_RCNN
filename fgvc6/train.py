import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from sys import path
import random
from fashion import FashionDataset, FashionConfig
import model as modellib
import os
import matplotlib as plt
import skimage
import cv2
import numpy as np
import pandas as pd

base_path = "F:\githubProject\Mask_RCNN\"
path.append("F:\githubProject\Mask_RCNN\fgvc6\")
path.append("F:\githubProject\Mask_RCNN\mrcnn\")
#base_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(base_path)

import visualize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

print(os.getcwd())

data_root_dir = "F:\jupyter_project\"
COCO_MODEL_PATH = os.path.join(data_root_dir, "mask_rcnn_coco.h5")
MODEL_DIR = os.path.join(data_root_dir, "logs") #保存权重到此路径

dataset_split = [9, 1]

dataset_train = FashionDataset()
dataset_train.load_fashion(data_root_dir, dataset_split, 0)
dataset_train.prepare()

dataset_val = FashionDataset()
dataset_val.load_fashion(data_root_dir, dataset_split, 1)
dataset_val.prepare()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def train():
    config = FashionConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=60,
                layers='all')

# load last model
#model.load_weights(model.find_last(), by_name=True)

WIDTH = 512
HEIGHT = 512
test_dir = "/home/ph/taotao/test/"


def test_imgIDs():
    img_ids = []
    pathDir = os.listdir(test_dir)
    for file in pathDir:
        #print(file)
        img_ids.append(file)
    return img_ids

def get_img(img_id):
    img = cv2.imread(test_dir+img_id)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    if img.ndim != 3:
        image = skimage.color.gray2rgb(img)
    # If has an alpha channel, remove it for consistency
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img

def encode(mask):
    print(mask.shape)
    #mask = np.array(mask)
    mask = mask.transpose((1, 0))
    mask = mask.flatten()  #按行展开
    mask = np.squeeze(mask) #删除维度为1的维度，比如（1,10）维度的执行后为（10，）
    #print("shape:", mask.shape)
    #print("===interval================")
    #print("mask:", mask)
    i = 0
    ans = ""
    while i < len(mask):
        if mask[i] == True:
            #print("-------------------")
            cnt = 0
            ans += " " + str(i)
            while mask[i] == 1 and i < len(mask):
                i += 1
                cnt += 1
            ans += " " + str(cnt)
        else:
            i += 1
   
    #print("ans:", ans)
    #print("====")
    return ans

if __name__ == '__main__':
    # 训练，应该是加载coco的预训练参数
    #tf.reset_default_graph()
    train()
    print("train finish...")

    # 运行模型标记测试图片，应该是加载最后一次的模型参数
    class InferenceConfig(FashionConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()
    #config.display()
    model = modellib.MaskRCNN(mode="inference", config=inference_config,
                              model_dir=MODEL_DIR)
    model.load_weights(model.find_last(), by_name=True)

    img_ids = test_imgIDs()
    results = []

    for img_id in img_ids:
        #print(img_id)
        #print("==============================")
        img = get_img(img_id)
        result = model.detect([img], verbose=1)
        results.extend(result)

    print(results[0]["masks"].shape)
    ans = []
    for i in range(len(results)):
        ImageId = img_ids[i]
        num_mask = results[i]["masks"].shape[2]
        #如果没有识别出来mask，则手动添加一个
        if num_mask == 0:
            ans.append({
                "ImageId":ImageId,
                "EncodedPixels": "1 1",
                "ClassId": 1,
            })
        for j in range(num_mask):
            EncodedPixels = encode(results[i]["masks"][:,:,j])
            ClassId = results[i]["class_ids"][j]
            ans.append({
                "ImageId":ImageId,
                "EncodedPixels": EncodedPixels,
                "ClassId": ClassId,
            })

    df = pd.DataFrame(ans)
    print(df)
    df.to_csv("submission.csv")


    # image_id = random.choice(dataset_val.image_ids)
    #
    # original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #     modellib.load_image_gt(dataset_val, inference_config,image_id, use_mini_mask=False)
    #
    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset_train.class_names, figsize=(8, 8))
    #
    # results = model.detect([original_image], verbose=1)
    # print(results.head)
    # r = results[0]
    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                             dataset_val.class_names, r['scores'], ax=get_ax())




# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=2,
#             layers="all")



