from pycuda import gpuarray
from functools import wraps
from . import KernelManager
from . import utils
import numpy as np

class Detector:
    def __init__(self, image_size, pixel_spacing, image=None, *, cpu=None):
        self.cpu = cpu
        if self.is_cpu() and len(image_size) == 2:
            image_size = Detector.make_detector_size(image_size, 1)
        #self.image = self.alloc(image_size) if image is None else image
        self.image = np.ascontiguousarray(np.zeros(image_size, dtype=np.float32)) if image is None else image
        self.image_size = image_size
        self.pixel_spacing = pixel_spacing

    def to_cpu(self):
        #assert self.cpu is not None
        if self.is_cpu():
            return self
        return self.cpu

    def to_gpu(self):
        #assert self.cpu is None
        if self.is_gpu():
            return self
        image_size = KernelManager.Module.get_global(
            'd_image_size', 
            np.array(self.image_size, dtype=np.float32)
            )
        return Detector(image_size, self.pixel_spacing, gpuarray.to_gpu(self.image), cpu=self)

    def is_cpu(self):
        return self.cpu is None

    def is_gpu(self):
        return self.cpu is not None

    def alloc(self, image_size):
        return np.transpose(np.zeros((image_size[2],image_size[1],image_size[0]), dtype=np.float32))


    @staticmethod
    def from_geometry(geometry_context, T_Nx4x4):
        n_proj = geometry_context.projection_matrix.shape[0] if geometry_context.projection_matrix.ndim == 3 else 1
        n_T = T_Nx4x4.shape[0] if T_Nx4x4.ndim == 3 else 1

        image_size = Detector.make_detector_size(geometry_context.image_size, n_proj * n_T)
        return Detector(image_size, geometry_context.pixel_spacing)

    @staticmethod
    def make_detector_size(image_size, n_channels):
        return np.array((image_size[0], image_size[1], n_channels), dtype=np.int32)
