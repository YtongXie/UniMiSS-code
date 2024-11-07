import pydrr
import numpy as np
import pycuda.autoinit
import pydrr.autoinit

def generate_drr(volume, spacing, **kwargs):
    """Generate DRR(Digitaly Reconstruct Radiography) image from volume.
    
    Args:
        volume (numpy.array): three-dimensional array. This volume is aligned by z, y, x
        spacing (tuple[float]): volume spacing. This argument is aligned by z, y, x. 
    
    Keyword Args:
        is_hu_volume: Input volume is HU volume so it will be converted myu volume
        SOD (float): Source to object distance (Default: 1800).
        SDD (float): Source to detector distance (Default: 2000).
        view (tuple[float]): View direction by x, y, z angles. (Default: [90, 0, 0]).
        pixel_size (tuple[float]): Pixel size of the detector.
        image_size (tuple[float]): Image size of the detector equal to result image size.
        extrinsic (numpy.array): Extrinsic matrix Nx4x4. If None, this value will compute automatically from view, SOD and SDD.
        intrinsic (numpy.array): Intrinsic matrix Nx3x3. If None, this vlaue will compute automatically from pixel_size and image_size.
        pose (array-like):  Translation and rotation of the volume. (Default:)
        projector (pydrr.Projector): Projector object. If None, Projector object is automatically construct.
                                     When you invoke this function many times, you might set this argument by performance reason.
    Returns:
        numpy.array: DRR image
    """

    args = {
        'SOD': 1800,
        'SDD': 2000,
        'view': [90, 0, 0],
        'extrinsic': None,
        'intrinsic': None,
        'projection': None,
        'pixel_size': [0.417, 0.417],
        'image_size': [1024, 1024],
        'pose': [0, 0, 0, 0, 0, 0],
        'projector': None,
        'is_hu_volume': False
    }

    args.update(kwargs)
    
    if args['is_hu_volume']:
        volume = pydrr.utils.HU2Myu(volume, 0.02)
    
    if args['extrinsic'] is None:
        args['extrinsic'] = pydrr.utils.convertTransRotTo4x4(
            np.asarray((0, 0, -args['SOD']) + tuple(args['view']))
            )
    if args['intrinsic'] is None:
        args['intrinsic'] = np.array(
            [[-args['SDD']/args['pixel_size'][0], 0, args['image_size'][0]/2.0], # unit: [pixel]
             [0, -args['SDD']/args['pixel_size'][1], args['image_size'][1]/2.0],
             [0,                0,               1]]
             )
    
    pm_Nx3x4 = pydrr.utils.constructProjectionMatrix(args['intrinsic'], args['extrinsic'])
    if pm_Nx3x4.ndim == 2:
        pm_Nx3x4 = pm_Nx3x4[np.newaxis, :, :]

    T_Nx4x4 = pydrr.utils.convertTransRotTo4x4(np.asarray(args['pose']))
    if T_Nx4x4.ndim == 2:
        T_Nx4x4 = T_Nx4x4[np.newaxis, :, :]

    # Define contexts.
    volume_context = pydrr.VolumeContext(volume.astype(np.float32), spacing)
    geometry_context = pydrr.GeometryContext()
    geometry_context.projection_matrix = pm_Nx3x4

    n_channels = T_Nx4x4.shape[0] * pm_Nx3x4.shape[0]

    if args['projector'] is None:
        detector = pydrr.Detector(
            pydrr.Detector.make_detector_size(args['image_size'], n_channels), 
            args['pixel_size']
            )

        projector = pydrr.Projector(detector, 1.0).to_gpu()

    # Host memory -> (Device memory) -> Texture memory
    t_volume_context = volume_context.to_texture()
    
    d_image = projector.project(t_volume_context, geometry_context, T_Nx4x4)

    # Device memory -> Host memory
    return d_image.get()
