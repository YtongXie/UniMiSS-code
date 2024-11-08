import numpy as np
import torchvision.transforms.functional as tf
import random
from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

################# Dataset for Seg
class MyDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, crop_size=(224, 224), max_iters=None):
        self.root_path = root_path + "Images/"
        self.root_path_mask1 = root_path + "Masks/clavicle/"
        self.root_path_mask2 = root_path + "Masks/heart/"
        self.root_path_mask3 = root_path + "Masks/lung/"

        self.list_path = root_path + list_path
        self.crop_w, self.crop_h = crop_size

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        print("Start preprocessing....")
        print('{} raw train images are loaded!'.format(len(self.img_ids)))

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = name
            self.files.append({
                "img": img_file,
                "name": name
            })

        print('{} training images are loaded!'.format(len(self.img_ids)))

        self.train_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_w)
             ])

        self.train_gt_augmentation = transforms.Compose(
            [transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1), shear=5.729578),
             transforms.RandomVerticalFlip(p=0.5),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             transforms.ToPILImage(),
             transforms.Resize(self.crop_h)
             ])


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        num_size = 256
        image = Image.open(self.root_path + datafiles["img"] + ".png").resize((num_size, num_size), Image.BICUBIC)
        label1 = Image.open(self.root_path_mask1 + datafiles["img"] + ".gif").resize((num_size, num_size), Image.NEAREST)
        label2 = Image.open(self.root_path_mask2 + datafiles["img"] + ".gif").resize((num_size, num_size), Image.NEAREST)
        label3 = Image.open(self.root_path_mask3 + datafiles["img"] + ".gif").resize((num_size, num_size), Image.NEAREST)

        is_crop = [0,1]
        random.shuffle(is_crop)

        if is_crop[0] == 0:
            [WW, HH] = image.size
            crop_num = np.array(range(0, WW-self.crop_w))

            random.shuffle(crop_num)
            crop_pw = crop_num[0]
            crop_ph = crop_num[1]

            rectangle = (crop_pw, crop_ph, self.crop_w + crop_pw, self.crop_w + crop_ph)
            image = image.crop(rectangle)
            label1 = label1.crop(rectangle)
            label2 = label2.crop(rectangle)
            label3 = label3.crop(rectangle)

        else:
            image = image.resize((self.crop_w, self.crop_h), Image.BICUBIC)
            label1 = label1.resize((self.crop_w, self.crop_h), Image.NEAREST)
            label2 = label2.resize((self.crop_w, self.crop_h), Image.NEAREST)
            label3 = label3.resize((self.crop_w, self.crop_h), Image.NEAREST)

        seed = np.random.randint(2147483647)
        set_seed(seed)
        image = self.train_augmentation(image)

        set_seed(seed)
        label1 = self.train_gt_augmentation(label1)

        set_seed(seed)
        label2 = self.train_gt_augmentation(label2)

        set_seed(seed)
        label3 = self.train_gt_augmentation(label3)

        image = np.array(image) / 255.
        image = np.concatenate([np.expand_dims(image, axis=0),np.expand_dims(image, axis=0),np.expand_dims(image, axis=0)],axis=0)
        image = image.astype(np.float32)

        label1 = np.array(label1)
        label1 = np.float32(label1 > 0)

        label2 = np.array(label2)
        label2 = np.float32(label2 > 0)

        label3 = np.array(label3)
        label3 = np.float32(label3 > 0)

        label = np.stack([label1, label2, label3], 0)

        name = datafiles["img"]

        return image.copy(), label.copy(), name


class MyValDataSet_seg(data.Dataset):
    def __init__(self, root_path, list_path, crop_size=(224, 224)):
        self.root_path = root_path + "Images/"
        self.root_path_mask1 = root_path + "Masks/clavicle/"
        self.root_path_mask2 = root_path + "Masks/heart/"
        self.root_path_mask3 = root_path + "Masks/lung/"

        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file = name
            self.files.append({
                "img": img_file,
                "name": name
            })
            
        print('{} val images are loaded!'.format(len(self.img_ids)))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(self.root_path + datafiles["img"] + ".png").resize((self.crop_h, self.crop_w), Image.BICUBIC)
        label1 = Image.open(self.root_path_mask1 + datafiles["img"] + ".gif").resize((self.crop_h, self.crop_w), Image.NEAREST)
        label2 = Image.open(self.root_path_mask2 + datafiles["img"] + ".gif").resize((self.crop_h, self.crop_w), Image.NEAREST)
        label3 = Image.open(self.root_path_mask3 + datafiles["img"] + ".gif").resize((self.crop_h, self.crop_w), Image.NEAREST)
        
        image = np.array(image) / 255.
        image = np.concatenate([np.expand_dims(image, axis=0),np.expand_dims(image, axis=0),np.expand_dims(image, axis=0)],axis=0)
        image = image.astype(np.float32)

        label1 = np.array(label1)
        label1 = np.float32(label1 > 0)

        label2 = np.array(label2)
        label2 = np.float32(label2 > 0)

        label3 = np.array(label3)
        label3 = np.float32(label3 > 0)

        label = np.stack([label1, label2, label3], 0)

        name = datafiles["img"]

        return image.copy(), label.copy(), name
