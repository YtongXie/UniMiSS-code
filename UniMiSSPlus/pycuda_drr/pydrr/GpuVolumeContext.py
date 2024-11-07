from .VolumeContext import VolumeContext
import numpy as np

class GpuVolumeContext(VolumeContext):
    def __init__(self, gpu_object, spacing, bridge):
        self.volume_object = gpu_object
        self.bridge = bridge

        self.volume = self.bridge.get_pointer(gpu_object)
        self.spacing = spacing
        self.volume_shape = self.bridge.get_shape(gpu_object)

        volume_size = np.asarray(self.volume_shape, dtype=np.uint32)
        self.spacing = np.asarray(spacing, dtype=np.float32)
        self.volume_corner_mm = np.array(volume_size * self.spacing / 2.0, dtype=np.float32)
        self.cpu = None
        self.texture = None

    def to_cpu(self):
        if self.cpu is None:
            self.cpu = VolumeContext(self.bridge.get_cpu_array(self.volume_object), self.spacing)
        return self.cpu

    def to_gpu(self):
        assert self.volume is not None
        return self

    def is_cpu(self):
        cpu = self.to_cpu()
        return cpu is not None

    def is_gpu(self):
        return True # Always true

    def is_texture(self):
        return False
