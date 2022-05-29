from PIL import Image
from matplotlib.cbook import ls_mapper
import torch
from torch.utils.data import Dataset
import random
import os

class Five_Flowers_Load(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path 
        self.transform = transform

        random.seed(0)  # 保证随机结果可复现
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(os.path.join(data_path))] 
        self.num_class = len(flower_class)
        # 排序，保证顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引  {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
        class_indices = dict((cla, idx) for idx, cla in enumerate(flower_class)) 

        self.images_path = []  # 存储训练集的所有图片路径
        self.images_label = []  # 存储训练集图片对应索引信息 
        self.images_num = []  # 存储每个类别的样本总数
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
        # 遍历每个文件夹下的文件
        for cla in flower_class:
            cla_path = os.path.join(data_path, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(data_path, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            self.images_num.append(len(images)) 
            # 写入列表
            for img_path in images: 
                self.images_path.append(img_path)
                self.images_label.append(image_class)

        print("{} images were found in the dataset.".format(sum(self.images_num))) 

 

    def __len__(self):
        return sum(self.images_num)
    
    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
        if self.transform is not None:
            img = self.transform(img)
        else:
            raise ValueError('Image is not preprocessed')
        return img, label
    
    # 非必须实现，torch里有默认实现；该函数的作用是: 决定一个batch的数据以什么形式来返回数据和标签
    # 官方实现的default_collate可以参考
    # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0) 
        labels = torch.as_tensor(labels)  
        return images, labels

Five_Flowers_Load('/data/haowen_yu/code/dataset/flowers/train')