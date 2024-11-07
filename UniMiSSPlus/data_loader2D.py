import cv2
import os
from torch.utils.data import Dataset
from torchvision import transforms

class Dataset2D(Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size_2D=(256, 256), data_type='2D_Modal', local_crops_number=2):

        self.root = root
        self.list_path = root+list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        if not max_iters == None:
            self.img_ids = self.img_ids * int(max_iters)

        self.local_crops_number = local_crops_number

        self.files = []
        for item in self.img_ids:
            # print(item)
            image_path = item
            name = image_path[0]
            img_file = image_path[0]
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop2D = crop_size_2D
        self.tr_transforms2D_global0 = get_train_transform2D_global0(self.crop2D)
        self.tr_transforms2D_global1 = get_train_transform2D_global1(self.crop2D)

        self.tr_transforms2D3_local = get_train_transform2D_local(96)

        self.data_type = data_type
        print(self.data_type)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_idx = self.files[index]
        img = []

        image2D = cv2.imread(os.path.join(self.root, img_idx["img"]))

        image2D = image2D[:, :, ::-1]
        img.append(self.tr_transforms2D_global0(image2D))
        img.append(self.tr_transforms2D_global1(image2D))

        for _ in range(self.local_crops_number):
            img.append(self.tr_transforms2D3_local(image2D))

        return img


def get_train_transform2D_global0(crop_size):

    tr_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.14, 1), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
         transforms.RandomGrayscale(p=0.8),
         transforms.RandomApply([transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=23)], p=0.5),
         transforms.ToTensor(),
         ])

    return tr_transforms

def get_train_transform2D_global1(crop_size):

    tr_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.14, 1), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
         transforms.RandomGrayscale(p=0.4),
         transforms.RandomApply([transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=23)], p=1.0),
         transforms.ToTensor(),
         ])

    return tr_transforms

def get_train_transform2D_local(crop_size):

    tr_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.05, 0.14), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
         transforms.RandomGrayscale(p=0.5),
         transforms.RandomApply([transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=23)], p=1.0),
         transforms.ToTensor(),
         ])

    return tr_transforms

