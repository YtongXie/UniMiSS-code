import numpy as np
import random
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


################# Dataset for Cls
class MyDataSet_cls(data.Dataset):
    def __init__(self, root_path, list_path, max_iters=None):
        self.root_path = root_path
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        print("Start preprocessing....")
        print('{} raw train images are loaded!'.format(len(self.img_ids)))

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.img_ids = self.img_ids[0:max_iters]
        self.files = []
        for data_file in self.img_ids:
            self.files.append({
                "img": data_file[0:-2],
                "label": data_file[-1],
            })

        print('{} training images are loaded!'.format(len(self.img_ids)))

        self.augmentation = build_transform_classification(normalize="chestx-ray", mode="train")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"]).convert('RGB')
        image = self.augmentation(image)
        label = np.array(np.int(datafiles["label"]))

        return image, label-1


class MyTestDataSet_cls(data.Dataset):
    def __init__(self, root_path, list_path):
        self.root_path = root_path
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        print("Start preprocessing....")
        print('{} raw vaild images are loaded!'.format(len(self.img_ids)))

        self.files = []
        for data_file in self.img_ids:
            self.files.append({
                "img": data_file[0:-2],
                "label": data_file[-1],
            })

        print('{} vaild images are loaded!'.format(len(self.img_ids)))

        self.augmentation = build_transform_classification(normalize="chestx-ray", mode="test")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"]).convert('RGB')
        image = self.augmentation(image)
        label = np.array(np.int(datafiles["label"]))

        return image, label-1




def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomVerticalFlip())
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence