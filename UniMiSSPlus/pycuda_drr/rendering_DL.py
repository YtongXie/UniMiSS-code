import pydrr
import pydrr.autoinit
import SimpleITK as sitk
import numpy as np
import os
from pydrr import utils
import imageio
import cv2

def truncate(CT):
    # truncate
    min_HU = -1000
    max_HU = 1000
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    return CT

def load_image(filename):
    itkimage = sitk.ReadImage(filename)
    volume  = sitk.GetArrayFromImage(itkimage)
    spacing = itkimage.GetSpacing()
    spacing = spacing[::-1]
    return volume, spacing

def save_image(filename, image, spacing):
    image = np.transpose(image, (2,0,1))
    spacing = (*spacing[::-1], 1)
    itkimage = sitk.GetImageFromArray(image)
    itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, filename)

def main(path, filename, save_path):
    # if len(sys.argv) < 2:
    #     print('rendering.py <volume path>')
    #     return

    # Load materials
    volume, spacing = load_image(os.path.join(path, filename))
    print(pydrr.get_supported_kernels())
    print(pydrr.get_current_kernel())
    pydrr.set_current_kernel('render_with_cubic_interp')
    print(pydrr.get_current_kernel())

    # volume = truncate(volume)
    # volume = pydrr.utils.HU2Myu(volume - 1000, 0.2683)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

    pm_Nx3x4, image_size, image_spacing = load_test_projection_matrix()
    T_Nx4x4 = load_test_transform_matrix()

    # Construct objects
    volume_context = pydrr.VolumeContext(volume.astype(np.float32), spacing)
    geometry_context = pydrr.GeometryContext()
    geometry_context.projection_matrix = pm_Nx3x4

    n_channels = T_Nx4x4.shape[0] * pm_Nx3x4.shape[0]
    detector = pydrr.Detector(pydrr.Detector.make_detector_size(image_size, n_channels), image_spacing)
    # detector = pydrr.Detector.from_geometry(geometry_context, T_Nx4x4) # You can use from_geometry if you set pixel_size and image_size.
    projector = pydrr.Projector(detector, 1.0).to_gpu()

    # Host memory -> (Device memory) -> Texture memory
    t_volume_context = volume_context.to_texture()

    d_image = projector.project(t_volume_context, geometry_context, T_Nx4x4)

    # # Device memory -> Host memory
    # image = d_image.get()
    # print('Result image shape:', image.shape)
    #
    # # image = np.flip(image[:,:,2].transpose(1,0), axis=0)
    # # image = cv2.resize(np.flip(image[:, :, 2].transpose(1,0), axis=0), [512, 512], interpolation=cv2.INTER_LINEAR)
    # image = cv2.resize(image[:, :, 2].transpose(1,0), [512, 512], interpolation=cv2.INTER_LINEAR)
    # # image = (image-np.min(image))/(np.max(image)-np.min(image))
    # # image = image.astype(np.uint8)
    #
    # imageio.imwrite(os.path.join(save_path, filename[0:-7] + ".png"), image)

    # Device memory -> Host memory
    image = d_image.get()
    print('Result image shape:', image.shape)
    # imageio.imwrite(os.path.join(save_path, filename[0:-7] + ".png"), image[:,:,1])

    image = np.flip(image[:,:,1].transpose(1,0), axis=0)
    # image = image[:, :, 2].transpose(1,0)
    image = image[image.sum(axis=1) != 0, :]
    image = image[:, image.sum(axis=0) != 0]

    scale_percent = 224/np.min(image.shape)
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)

    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    imageio.imwrite(os.path.join(save_path, filename[0:-7] + ".png"), image)

    # plt.figure(figsize=(16,9))
    # n_show_channels = 3
    # for i in range(min(image.shape[2], n_show_channels)):
    #     ax = plt.subplot(1, min(image.shape[2], n_show_channels), i+1)
    #     divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    #     cax = divider.append_axes('right', '5%', pad='3%')
    #     im = ax.imshow(image[:, :, i], interpolation='none', cmap='gray')
    #     plt.colorbar(im, cax=cax)
    # plt.show()
    #
    # save_image('drr.mhd', image, image_spacing)


def load_test_projection_matrix(SDD=2000, SOD=1800, image_size=[1280, 1280], spacing=[0.287, 0.287] ):

    if isinstance(image_size, list):
        image_size = np.array(image_size)

    if isinstance(spacing, list):
        spacing = np.array(spacing)

    extrinsic_R = utils.convertTransRotTo4x4([[0,0,0,90,0,0],
                                              [0,0,0,0,90,0],
                                              [0,0,0,0,0,90]])

    print('extrinsic_R:', extrinsic_R)
    print('extrinsic_R.shape:', extrinsic_R.shape)

    extrinsic_T = utils.convertTransRotTo4x4([0,0,-SOD,0,0,0])

    print('extrinsic_T:', extrinsic_T)
    print('extrinsic_T.shape:', extrinsic_T.shape)


    extrinsic = utils.concatenate4x4(extrinsic_T, extrinsic_R)

    print('extrinsic:', extrinsic)
    print('extrinsic.shape:', extrinsic.shape)


    intrinsic = np.array([[-SDD/spacing[0], 0, image_size[0]/2.0], # unit: [pixel]
                          [0, -SDD/spacing[1], image_size[1]/2.0],
                          [0,                0,               1]])

    print('intrinsic:', intrinsic)
    print('intrinsic.shape:', intrinsic.shape)


    pm_Nx3x4 = utils.constructProjectionMatrix(intrinsic, extrinsic)
    #pm_Nx3x4 = np.repeat(pm_Nx3x4, 400, axis=0)

    print('pm_Nx3x4:', pm_Nx3x4)
    print('pm_Nx3x4.shape:', pm_Nx3x4.shape)

    return pm_Nx3x4, image_size, spacing

def load_test_transform_matrix(n_channels=1):
    T_Nx6 = np.array([0,0,0,90,0,0])
    T_Nx6 = np.expand_dims(T_Nx6, axis=0)
    T_Nx6 = np.repeat(T_Nx6, n_channels, axis=0)
    T_Nx4x4 = utils.convertTransRotTo4x4(T_Nx6)

    print('T_Nx4x4:', T_Nx4x4)
    print('T_Nx4x4.shape:', T_Nx4x4.shape)

    return T_Nx4x4

if __name__ == '__main__':

    path = '../data/3D_subvolumes'
    save_path = '../data/3D_subvolumes'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for root, dirs, files in os.walk(path):
        for i_files in sorted(files):
            if (i_files[0] == '.') or ('png' in i_files):
                continue
            print(i_files)
            main(path, i_files, save_path)
