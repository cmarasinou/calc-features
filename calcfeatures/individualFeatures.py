from skimage.measure import regionprops
import pandas as pd
import scipy

from calcfeatures.base import BaseFeatures

class IndividualFeatures(BaseFeatures):
    '''
    Given a region, extracts features based on individual segments (objects in segmentation mask)
    Args:
        mask (ndarray,2d,bool)
        image (ndarray,2d,uint8)
    '''
    def __init__(self,mask,image,**kwargs) -> None:
        super(IndividualFeatures,self).__init__(mask,image,**kwargs)
    def get_segments(self):
        self.segments=list()
        label, _ = scipy.ndimage.label(self.mask)
        for prop in regionprops(label):
            y1,x1,y2,x2 = prop.bbox
            seg_mask = label[y1:y2,x1:x2]==prop.label
            seg_img = None
            if self.image is not None:
                seg_img = self.image[y1:y2,x1:x2]
            self.segments.append(IndividualSegment(seg_mask,seg_img))
    def get_size_features(self):
        for segment in self.segments:
            segment.get_size_features()
    def get_shape_features(self):
        for segment in self.segments:
            segment.get_shape_features()
    def aggregate_statistically(self):
        feature_list = list()
        for segment in self.segments:
            feature_list.append(segment.features)
        df_features = pd.DataFrame(feature_list)
        self.aggregate_features = dict()
        for ftr in df_features.columns:
            self.aggregate_features[f'{ftr}_min']=df_features[ftr].min()
            self.aggregate_features[f'{ftr}_max']=df_features[ftr].max()
            self.aggregate_features[f'{ftr}_mean']=df_features[ftr].mean()
            self.aggregate_features[f'{ftr}_median']=df_features[ftr].median()
            self.aggregate_features[f'{ftr}_std']=df_features[ftr].std()
            self.aggregate_features[f'{ftr}_skew']=df_features[ftr].skew()
            self.aggregate_features[f'{ftr}_kurtosis']=df_features[ftr].kurtosis()



class IndividualSegment:
    '''
    To extract features from an individual segment (object)
    Args:
        mask (ndarray,2d,bool): mask of the object
        image (ndarray,2d,uint8): image patch for the same region as mask
    '''
    def __init__(self,mask,image=None):
        self.mask = mask
        self.image = image
        self.features=dict()
        self.__load_regionprop__()
    def get_size_features(self):
        self.features['area']=self.region_prop.area
        self.features['convex_area']=self.region_prop.convex_area
        self.features['major_axis_length']=self.region_prop.major_axis_length
        self.features['minor_axis_length']=self.region_prop.minor_axis_length
    def get_shape_features(self):
        self.features['eccentricity']=self.region_prop.eccentricity
        self.features['solidity']=self.region_prop.solidity
        # Hu moments
        for i, val in enumerate(self.region_prop.moments_hu):
            self.features[f'moment_hu_{i}']=val

    def __load_regionprop__(self):
        if not hasattr(self,'region_prop'):
            self.region_prop = regionprops(self.mask.astype(int),self.image)[0]