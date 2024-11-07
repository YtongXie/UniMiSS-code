from pycuda import driver, compiler, gpuarray, tools 
from .Kernel import Kernel
from . import KernelManager

class KernelModule:
    Interpolations = {
        'linear': driver.filter_mode.LINEAR
    }
    def __init__(self, source, info):
        self.module = compiler.SourceModule(source, options=['--ptxas-options=-v'], cache_dir=None, keep=False)
        self.kernels = dict()
        self.attributes = dict()
        self.texture_attributes = dict()
        self.setCurrentModule()
        for func_name, attrs in info.items():
            self.attributes.update(dict(zip(attrs['global'], [ False for _ in attrs['global'] ])))
            self.texture_attributes.update(dict(zip(attrs['texture'], [ False for _ in attrs['texture'] ])))
            self.kernels[func_name] = Kernel(self, func_name, attrs)

    def get_function(self, name):
        kernel_func = self.module.get_function(name)
        return kernel_func

    def get_kernel(self, name):
        return self.kernels[name]

    def verify_attributes(self, attrs):
        if not attrs: # Attributes are empty
            return True, []

        founds = [ (self.attributes[name], name) for name in attrs ]
        founds, names = map(list, zip(*founds))
        return all(founds), [ name for name, found in zip(names, founds) if not found ]

    def verify_texture_attributes(self, attrs):
        if not attrs: # Attributes are empty
            return True, []

        founds = [ (self.texture_attributes[name], name) for name in attrs ]
        founds, names = map(list, zip(*founds))
        return all(founds), [ name for name, found in zip(names, founds) if not found ]

    def get_global(self, name, host_obj):
        assert name in self.attributes, 'Unknown global atrribute: {}'.format(name)
        self.attributes[name] = True

        device_obj = self.module.get_global(name)[0]
        driver.memcpy_htod(device_obj, host_obj)
        return device_obj

    def get_texture(self, name, device_obj, interpolation=None):
        assert name in self.texture_attributes, 'Unknown texture atrribute: {}'.format(name)
        self.texture_attributes[name] = True

        texture_obj = self.module.get_texref(name)
        if interpolation is not None and interpolation in KernelModule.Interpolations:
            texture_obj.set_filter_mode(KernelModule.Interpolations[interpolation])
        texture_obj.set_array(device_obj)
        return texture_obj

    def setCurrentModule(self):
        KernelManager.Module = self
