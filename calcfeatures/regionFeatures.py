from calcfeatures.base import BaseFeatures
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
from skimage.feature import greycomatrix,greycoprops
from scipy import ndimage

__all_feature_classes__ = ['size', 'shape']


class RegionFeatures(BaseFeatures):
    '''
    Given a region, extracts features based on individual segments (objects in segmentation mask)
    Args:
        mask (ndarray,2d,bool)
        image (ndarray,2d,uint8)
    '''
    def __init__(self,mask,image,**kwargs) -> None:
        super(RegionFeatures,self).__init__(mask,image,**kwargs)
        self.settings = kwargs
        self.features = dict()
    def get_shape_features(self):
        self.__load_regionprop__()
        self.features['eccentricity']=self.region_prop.eccentricity
        self.features['inertia_1'] = self.region_prop.inertia_tensor_eigvals[0]
        self.features['inertia_2'] = self.region_prop.inertia_tensor_eigvals[1]
        self.features['orientation'] = self.region_prop.orientation
        convexhull = convex_hull_image(self.mask.astype(int)).astype(int)
        convex_hull_rprop = regionprops(convexhull)[0]
        self.features['solidity']=convex_hull_rprop.solidity
        for i, val in enumerate(self.region_prop.moments_hu):
            self.features[f'moment_hu_{i}']=val
    def get_size_features(self):
        self.__load_regionprop__()
        self.features['area']=self.region_prop.area
        self.features['convex_area']=self.region_prop.convex_area
        self.features['major_axis_length']=self.region_prop.major_axis_length
        self.features['minor_axis_length']=self.region_prop.minor_axis_length
        _, n_objects = ndimage.label(self.mask.astype(int))
        self.features['n_objects'] = n_objects
    def get_greylevel_features(self):
        distances = [5]
        angles = [0]
        glcm=greycomatrix(self.image, distances, angles, 2**8, symmetric=True, normed=True)
        grey_level_features = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
        for f in grey_level_features:
            values = greycoprops(glcm,f).ravel()
            for i,v in enumerate(values):
                self.features[f+str(i)]=v
    def __load_regionprop__(self):
        if not hasattr(self,'region_prop'):
            self.region_prop = regionprops(self.mask.astype(int),self.image)[0]
    def execute(self):
        pass