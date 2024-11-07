import numpy as np
from . import utils

class GeometryContext:
    def __init__(self):
        self.SOD_ = 0.0
        self.SDD_ = 0.0
        self.pixel_spacing_ = (1.0, 1.0)
        self.image_size_ = (1024, 1024)
        self.view_matrix_ = np.eye(4,4, dtype=np.float32)
        
        self.intrinsic_ = None
        self.extrinsic_ = None

        self.projection_matrix_ = None

    @property
    def SOD(self):
        return self.SOD_

    @SOD.setter
    def SOD(self, value):
        self.intrinsic = None
        self.SOD_ = value

    @property
    def SDD(self):
        return self.SDD_

    @SDD.setter
    def SDD(self, value):
        self.intrinsic = None
        self.extrinsic = None
        self.SDD_ = value

    @property
    def pixel_spacing(self):
        return self.pixel_spacing_

    @pixel_spacing.setter
    def pixel_spacing(self, value):
        self.intrinsic = None
        self.pixel_spacing_ = value

    @property
    def image_size(self):
        return self.image_size_

    @image_size.setter
    def image_size(self, value):
        self.intrinsic = None
        self.image_size_ = value
        
    @property
    def view_matrix(self):
        return self.view_matrix_

    @view_matrix.setter
    def view_matrix(self, value):
        self.extrinsic = None
        self.view_matrix_ = value

    @property
    def intrinsic(self):
        if self.intrinsic_ is None:
            self.intrinsic_ = np.array([
                [ self.SOD / self.pixel_spacing[0], 0, self.image_size[0] / 2 ], 
                [ 0, self.SOD / self.pixel_spacing[1], self.image_size[1] / 2 ],
                [0, 0, 1]
            ])
        return self.intrinsic_
    
    @intrinsic.setter
    def intrinsic(self, new_intrinsic):
        self.projection_matrix = None
        self.intrinsic_ = new_intrinsic
    
    @property
    def extrinsic(self):
        if self.extrinsic_ is None:
            extrinsic_T = utils.convertTransRotTo4x4([0, 0, -self.SOD, 0, 0, 0])
            self.extrinsic_ = utils.concatenate4x4(extrinsic_T, self.view_matrix)
        return self.extrinsic_

    @extrinsic.setter
    def extrinsic(self, new_extrinsic):
        self.projection_matrix = None
        self.extrinsic_ = new_extrinsic
    
    @property
    def projection_matrix(self):
        if self.projection_matrix_ is None:
            self.projection_matrix_ = utils.constructProjectionMatrix(self.intrinsic, self.extrinsic)
        return self.projection_matrix_

    @projection_matrix.setter
    def projection_matrix(self, value):
        self.projection_matrix_ = value

    