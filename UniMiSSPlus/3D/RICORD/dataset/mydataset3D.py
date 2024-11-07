import numpy as np
from torch.utils import data
import nibabel as nib
import math
from batchgenerators.transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
    BrightnessTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from skimage.transform import resize


class Dataset3D(data.Dataset):
    def __init__(self, root, list_path, crop_size_3D=(64, 64, 64), max_iters=None):

        self.root = root
        self.list_path = root + list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for item in self.img_ids:
            # print(item)
            image_gt_path = item
            name = image_gt_path[0]
            img_file = image_gt_path[0]
            gt = image_gt_path[1]
            self.files.append({
                "img": img_file,
                "gt": gt,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

        self.crop3D_d, self.crop3D_h, self.crop3D_w = crop_size_3D
        self.tr_transforms3D = get_train_transform3D(patch_size=crop_size_3D)


    def __len__(self):
        return len(self.files)

    def truncate(self, CT):

        min_HU = -1024
        max_HU = 325
        subtract = 158.58
        divide = 324.70

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def pad_image(self, img):

        """Pad an image up to the target size."""
        rows_missing = math.ceil(self.crop3D_w - img.shape[0])
        cols_missing = math.ceil(self.crop3D_h - img.shape[1])
        dept_missing = math.ceil(self.crop3D_d - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (dept_missing//2, dept_missing-dept_missing//2)), 'constant')
        return padded_img

    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])
        image = imageNII.get_fdata()
        image = self.pad_image(image)
        image = self.truncate(image)
        image = image[np.newaxis, :][np.newaxis, :]
        image = image.transpose((0, 1, 4, 2, 3))

        label = int(datafiles["gt"])

        name = datafiles["name"]

        data_dict = {'image': image.astype(np.float32).copy(), 'label': None, 'name': name}
        data_dict = self.tr_transforms3D(**data_dict)

        img = data_dict['image'][0]

        return img.copy(), label


class ValDataSet3D(data.Dataset):
    def __init__(self, root, list_path, crop_size_3D=(64, 64, 64)):

        self.root = root
        self.list_path = root + list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.crop_size = crop_size_3D

        self.files = []
        for item in self.img_ids:
            # print(item)
            image_gt_path = item
            name = image_gt_path[0]
            img_file = image_gt_path[0]
            gt = image_gt_path[1]
            self.files.append({
                "img": img_file,
                "gt": gt,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT):

        min_HU = -1024
        max_HU = 325
        subtract = 158.58
        divide = 324.70

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT


    def __getitem__(self, index):
        # t1=timeit.timeit()
        datafiles = self.files[index]
        # read nii file
        imageNII = nib.load(self.root + datafiles["img"])
        image = imageNII.get_fdata()
        image = image.transpose((2, 0, 1))
        image = resize(image.astype(float), self.crop_size, order=3, mode="constant", anti_aliasing=False)
        image = image[np.newaxis, :]
        image = self.truncate(image)

        label = int(datafiles["gt"])

        return image.astype(np.float32).copy(), label


def get_train_transform3D(patch_size):
    tr_transforms = []

    tr_transforms.append(
        SpatialTransform(
            patch_size, patch_center_dist_from_border=[i // 2 for i in patch_size],
            do_elastic_deform=True, alpha=(0., 900.), sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True, scale=(0.85, 1.25),
            border_mode_data='constant', border_cval_data=0,
            order_data=3, border_mode_seg="constant", border_cval_seg=-1,
            order_seg=1,
            random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False,
            data_key="image")
    )
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2), data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms



