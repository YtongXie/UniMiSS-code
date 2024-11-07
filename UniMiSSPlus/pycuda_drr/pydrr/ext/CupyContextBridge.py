import cupy

class CupyContextBridge:
    @classmethod
    def get_pointer(cls, array):
        return array.data.ptr

    @classmethod
    def get_shape(cls, array):
        return array.shape

    @classmethod
    def get_cpu_array(cls, array):
        return cupy.asnumpy(array)