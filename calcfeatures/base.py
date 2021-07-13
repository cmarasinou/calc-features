import numpy as np


class BaseFeatures:
    """
    Defines initial and basic functionality for feature extraction

    Args:
        mask (ndarray,2d,bool)
        image (ndarray,2d,uint8)
    """
    def __init__(self,mask,image,**kwargs) -> None:
        self.check_mask(mask)
        self.check_image(image)
        assert(mask.shape == image.shape),"Mask and Image shapes do not match"
        self.mask = mask
        self.image = image
    def check_mask(self,mask):
        if type(mask)==np.ndarray:
            if (mask.dtype == np.dtype('bool')) and (mask.ndim==2):
                return True
        raise TypeError("Mask does not have the correct type \
            should be 2d array with dtype bool")
    def check_image(self,image):
        if type(image)==np.ndarray:
            if (image.dtype == np.dtype('uint8')) and (image.ndim==2):
                return True
        raise TypeError("Image does not have the correct type,\
            should be 2d array with dtype uint8")

