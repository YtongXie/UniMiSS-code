from pycuda import driver, compiler, gpuarray, tools
from . import KernelManager

class Kernel:
    def __init__(self, module, func_name, attrs):
        self.parent_module = module
        self.func_name = func_name
        self.kernel = module.get_function(func_name)
        self.attributes = attrs
        self.setCurrent()

    def invoke(self, *args, **kwargs):
        flag, not_founds = self.parent_module.verify_attributes(self.attributes['global'])
        assert flag, 'Following global attributes are not initialized: {}'.format(not_founds)
        flag, not_founds = self.parent_module.verify_texture_attributes(self.attributes['texture'])
        assert flag, 'Following texture attributes are not initialized: {}'.format(not_founds)

        if 'grid' not in kwargs:
            kwargs['grid'] = (1, 1, 1)
        if 'block' not in kwargs:
            kwargs['block'] = (1, 1, 1)

        return self.kernel(*args, **kwargs)

    def setCurrent(self):
        self.parent_module.setCurrentModule()
        KernelManager.Kernel = self
