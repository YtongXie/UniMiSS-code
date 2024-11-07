import numpy as np
from pycuda import driver
from . import utils
from . import KernelManager

class VolumeContext:
    def __init__(self, volume, spacing = (1,1,1), *, cpu=None, gpu=None):
        self.cpu = cpu
        self.gpu = gpu
        if volume is None:
            return
        
        if not volume.flags['C_CONTIGUOUS']: # The array must be contiguous array for gpu copy.
            volume = np.ascontiguousarray(volume, dtype=np.float32)
        
        self.volume = volume
        volume_size = np.asarray(self.volume.shape, dtype=np.uint32)
        self.spacing = np.asarray(spacing, dtype=np.float32)
        self.volume_corner_mm = np.array(volume_size * self.spacing / 2.0, dtype=np.float32)

    def to_cpu(self):
        assert self.cpu is not None
        return self.cpu

    def to_gpu(self):
        assert self.volume is not None

        if self.is_gpu():
            return self
        elif self.is_texture():
            return self.gpu

        obj = VolumeContext(None, cpu=self)
        obj.volume = driver.np_to_array(self.volume, order='C')

        obj.spacing = KernelManager.Module.get_global('d_volume_spacing', self.spacing)
        obj.volume_corner_mm = KernelManager.Module.get_global('d_volume_corner_mm', self.volume_corner_mm)
        return obj

    def to_texture(self, interpolation = 'linear'):
        cpu = None
        gpu = None
        if self.is_cpu():
            cpu = self
            gpu = self.to_gpu()
        elif self.is_gpu():
            cpu = self.cpu
            gpu = self
        else:
            return self
        obj = VolumeContext(None, cpu=cpu, gpu=gpu)

        obj.volume = KernelManager.Module.get_texture('t_volume', gpu.volume, interpolation)

        obj.spacing = gpu.spacing
        obj.volume_corner_mm = gpu.volume_corner_mm
        return obj

    def is_cpu(self):
        return self.cpu is None and self.gpu is None

    def is_gpu(self):
        return self.cpu is not None and self.gpu is None

    def is_texture(self):
        return self.cpu is not None and self.gpu is not None
