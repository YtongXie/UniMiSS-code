from functools import wraps
import numpy as np
from . import KernelManager
from . import utils
from pycuda import driver

class Projector:
    
    # set block and grid size
    block = (32, 32, 1)
    grid  = None

    def __init__(self, target_detector, step_size_mm = 1, *, cpu=None):
        self.target_detector = target_detector
        self.step_size_mm = step_size_mm
        self.cpu = cpu

    def project(self, volume_context, geometry_context, T_Nx4x4):
        assert self.cpu is not None, 'CPU ray casting is not supported.'

        image_size = self.target_detector.to_cpu().image_size
        pm_Nx3x4 = geometry_context.projection_matrix

        p_Nx12 = utils.constructProjectionParameter(pm_Nx3x4, np.array(image_size[:2]), T_Nx4x4)
        
        assert self.target_detector.cpu.image_size[2] == p_Nx12.shape[0], 'Unmatched detector channel and pose parameter channel.(Actual: {} != {})'.format(self.target_detector.cpu.image_size[2], p_Nx12.shape[0])

        h_p_Nx12 = p_Nx12.astype(np.float32)
        d_p_Nx12 = driver.np_to_array(h_p_Nx12, order='C')
        t_p_Nx12 = KernelManager.Module.get_texture('t_proj_param_Nx12', d_p_Nx12)

        grid = (16, 16, 1)
        if Projector.grid is None:
            grid = tuple(np.uint32(np.ceil(image_size / Projector.block)).tolist())
        
        KernelManager.Kernel.invoke(
            self.target_detector.image.gpudata,
            texrefs=[volume_context.volume, t_p_Nx12],
            block=Projector.block, grid=grid
        )
        # Display debug info
        # print_kernel = KernelManager.Module.get_kernel('print_device_params')
        # print_kernel.invoke(texrefs=[t_p_Nx12])

        return self.target_detector.image

    def to_gpu(self):
        assert self.cpu is None

        step_size_mm = KernelManager.Module.get_global('d_step_size_mm', np.float32(self.step_size_mm))
        return Projector(self.target_detector.to_gpu(), step_size_mm, cpu=self)
