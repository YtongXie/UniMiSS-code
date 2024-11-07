from pycuda import driver, compiler, gpuarray, tools

from pydrr import kernels
from .KernelModule import KernelModule

class KernelManager:
    Kernel = None
    Module = None
    Modules = []

    def __init__(self, default_kernel, *kernel_codes):
        for kernel_code, kernel_info in kernel_codes:
            KernelManager.Modules.append(KernelModule(kernel_code, kernel_info))

        # TODO: (ad-hoc) only one kernel code is supported, currently
        KernelManager.Module = KernelManager.Modules[0]
        KernelManager.Kernel = KernelManager.Module.get_kernel(default_kernel)

    def set_current_kernel(self, name):
        KernelManager.Kernel = KernelManager.Module.get_kernel(name)

_manager = None

def initialize():
    global _manager

    _default_kernel = 'render_with_linear_interp'

    _manager = KernelManager(_default_kernel,
        (
            kernels.render_kernel,
            {
                'render_with_linear_interp':
                {
                    'global': [
                        'd_step_size_mm',
                        'd_image_size',
                        'd_volume_spacing',
                        'd_volume_corner_mm',
                    ],
                    'texture' : [
                        't_volume',
                        't_proj_param_Nx12',
                    ],
                },
                'render_with_cubic_interp':
                {
                    'global': [
                        'd_step_size_mm',
                        'd_image_size',
                        'd_volume_spacing',
                        'd_volume_corner_mm',
                    ],
                    'texture' : [
                        't_volume',
                        't_proj_param_Nx12',
                    ],
                },
                'print_device_params': { 'global':[], 'texture':['t_proj_param_Nx12'] }
            }
        ),
    )



def get_supported_kernels():
    return list(KernelManager.Module.kernels.keys())


def set_current_kernel(name):
    global _manager
    _manager.set_current_kernel(name)


def get_current_kernel():
    kernel = KernelManager.Kernel
    return kernel.func_name
