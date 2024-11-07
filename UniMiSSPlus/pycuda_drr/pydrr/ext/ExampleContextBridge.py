
class ExampleContextBridge:
    """Example bridge object for GpuVolumeContext
    
    A bridge object works get attribute from user type to construct VolumeContext.
    """

    @classmethod
    def get_pointer(cls, array):
        """Get device pointer from user type.
        """

        raise NotImplementedError

    @classmethod
    def get_shape(cls, array):
        """Get array shape from user type.
        """
        raise NotImplementedError

    @classmethod
    def get_cpu_array(cls, array):
        """Construct numpy array from user type.
        """
        raise NotImplementedError

