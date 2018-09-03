import random
from functools import wraps

import cv2
import imgaug as ia
import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
from imgaug import augmenters as iaa
from scipy.ndimage.filters import gaussian_filter


class Compose:
    def __init__(self,transforms,shuffle=True):
        self.transforms = [t for t in transforms if t is not None]
        
        self.shuffle = shuffle
    def __call__(self,**data):
        if self.shuffle:
            random.shuffle(self.transforms)
        for t in self.transforms:
            
                data = t(**data)
        return data
        
class OneOf:
    def __init__(self,transforms):
        self.transforms = [t for t in transforms if t is not None]
    def __call__(self,**data):
        t = random.choice(self.transforms)
        return t(**data)

class BaseTransform:
    def __init__(self,value_range=(0,0),prob=0.5):
        '''
        args:
            value_range: tranforms range
            prob:tranforms prob
        '''
        self.value_range = value_range
        self.prob = prob
    def __call__(self,image,masks=None,bboxs=None):
        '''
        image: (height,width,channels)
        masks: (height,width) or (height,width,channels)
        bboxs: (num,4),each row is (start_y,start_x,end_y,end_x)
        '''
        if random.random() < self.prob:
            #print('aug')
            return self.call(image,masks,bboxs)
        if masks is None and bboxs is None:
            return image
        elif masks is not None and bboxs is None:
            return image,masks
        elif bboxs is not None and masks is None:
            return image,bboxs
        else:
            return image,masks,bboxs
    def call(self,image,masks,bboxs):
        min = self.value_range[0]
        max = self.value_range[1]
        self.value = random.random()*(max-min) + min
        
        #print(value)
        image = self.process_image(image,self.value)
        if masks is None and bboxs is None:
            return image
        elif masks is not None and bboxs is None:
            return image,self.process_masks(masks,self.value)
        elif bboxs is not None and masks is None:
            return image,self.process_bboxs(bboxs,self.value)
        else:
            return image,self.process_masks(masks,self.value),self.process_bboxs(bboxs,self.value)
    def process_image(self,image,value):
        return image
    def process_masks(self,masks,value):
        return masks
    def process_bboxs(self,bboxs,value):
        return bboxs


###########################################AFFINE TRANSFORM ##############################################
def affine_bboxs(matrix,bboxs):
    t_bboxs = bboxs.copy()
    n_bboxs = np.hstack((t_bboxs,t_bboxs[:,[0,3,2,1]]))
    #print(n_bboxs.shape)
    n_bboxs = n_bboxs.reshape(-1,2)
    n_bboxs = np.hstack((n_bboxs,np.ones((len(n_bboxs),1)))).T
    #print(n_bboxs.shape)
    n_bboxs = n_bboxs[[1,0,2]]
    n_bboxs = np.dot(matrix,n_bboxs)[[1,0]]
    n_bboxs = n_bboxs.T.reshape(-1,8)     
    t_bboxs[:,0] = np.min(n_bboxs[:,[0,2,4,6]],axis=1)
    t_bboxs[:,1] = np.min(n_bboxs[:,[1,3,5,7]],axis=1)
    t_bboxs[:,2] = np.max(n_bboxs[:,[0,2,4,6]],axis=1)
    t_bboxs[:,3] = np.max(n_bboxs[:,[1,3,5,7]],axis=1)  
    return t_bboxs

def clip_bboxs(bboxs,height,width):
    bboxs[:,[0,2]] = np.clip(bboxs[:,[0,2]],0,height)
    bboxs[:,[1,3]] = np.clip(bboxs[:,[1,3]],0,width)
    return bboxs

class ShearX(BaseTransform):
    def __init__(self, shearX_range=(-0.2,0.2), prob=0.5):
        super(ShearX, self).__init__(shearX_range, prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        self.matrix = np.array([[1, value, 0],
                                [0, 1,     0]], dtype=np.float32)
        image = cv2.warpAffine(image, self.matrix, (self.width, self.height))
        return image
    def process_masks(self,masks,value):
        masks = cv2.warpAffine(masks, self.matrix, (self.width, self.height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs
    
class ShearY(BaseTransform):
    def __init__(self, shearY_range=(-0.2,0.2),prob=0.5):
        super(ShearY, self).__init__(shearY_range, prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        self.matrix = np.array([[1, 0, 0],
                                [value, 1,     0]], dtype=np.float32)
        image = cv2.warpAffine(image, self.matrix, (self.width, self.height))
        return image
    def process_masks(self,masks,value):    
        masks = cv2.warpAffine(masks, self.matrix, (self.width, self.height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs
    
class ShiftX(BaseTransform):
    def __init__(self, ShiftX_range=(-20,20), prob=0.5):
        super(ShiftX, self).__init__(ShiftX_range, prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        self.matrix = np.array([[1, 0, value],
                                [0, 1,     0]], dtype=np.float32)
        image = cv2.warpAffine(image, self.matrix, (self.width, self.height))
        return image
    def process_masks(self,masks,value):
        masks = cv2.warpAffine(masks, self.matrix, (self.width, self.height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs
    
class ShiftY(BaseTransform):
    def __init__(self, ShiftY_range=(-20,20), prob=0.5):
        super(ShiftY, self).__init__(ShiftY_range, prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        self.matrix = np.array([[1, 0, 0],
                                [0, 1, value]], dtype=np.float32)
        image = cv2.warpAffine(image, self.matrix, (self.width, self.height))
        return image
    def process_masks(self,masks,value):
        masks = cv2.warpAffine(masks, self.matrix, (self.width, self.height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs
    
class Rotation(BaseTransform):
    def __init__(self, rotation_range=(-10,10), prob=0.5):
        super(Rotation, self).__init__(rotation_range, prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        self.matrix = cv2.getRotationMatrix2D((self.height//2, self.width//2),value,scale=1)
        image = cv2.warpAffine(image, self.matrix, (self.width, self.height))
        return image
    def process_masks(self,masks,value):
        masks = cv2.warpAffine(masks, self.matrix, (self.width, self.height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs

class RotationAffine(BaseTransform):
    def __init__(self, scale=(-10, 10), prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.Affine(rotate=scale)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
    def process_masks(self,masks,value):
        return self.deterministic_processor.augment_image(masks)
    def process_bboxs(self,bboxs,value):
        BoundingBox = [ia.BoundingBox(*bbox[[1,0,3,2]]) for bbox in bboxs]
        bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=self.shape)
        bbs_aug = self.deterministic_processor.augment_bounding_boxes([bbs])[0]
        bboxs = []
        for bbox in bbs_aug.bounding_boxes:
            bboxs.append([bbox.y1,bbox.x1,bbox.y2,bbox.x2])
        return np.array(bboxs)

class Rotation90(BaseTransform):
    def __init__(self, prob=0.5):
        super(Rotation90, self).__init__(prob=prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        self.value = random.randint(1,3)*90
        self.matrix = cv2.getRotationMatrix2D((self.height//2, self.width//2),self.value,scale=1)
        image = cv2.warpAffine(image, self.matrix, (self.width, self.height))
        return image
    def process_masks(self,masks,value):
        masks = cv2.warpAffine(masks, self.matrix, (self.width, self.height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs

class Zoom(BaseTransform):
    def __init__(self, zoom_range=(0.8,1.2), prob=0.5):
        super(Zoom, self).__init__(zoom_range, prob)
        
    def process_image(self,image,value):
        height, width = image.shape[0:2]
        self.t_height,self.t_width = int(height*value),int(width*value)
        #print(self.t_height,self.t_width )
        image = cv2.resize(image,(self.t_width,self.t_height))
        return image
    def process_masks(self,masks,value):
        masks = cv2.resize(masks,(self.t_width,self.t_height))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs[:,[0,2]] = value*bboxs[:,[0,2]]
        bboxs[:,[1,3]] = value*bboxs[:,[1,3]]
        return bboxs
    
class Resize(BaseTransform):
    def __init__(self,target_size=None,prob=0.5):
        super(Resize, self).__init__(prob=prob)
        self.target_size = target_size
    def process_image(self,image,value):
        height, width = image.shape[0:2]
        self.scale_h = self.target_size[0]/height
        self.scale_w = self.target_size[1]/width
        image = cv2.resize(image,(self.target_size[1],self.target_size[0]))
        return image
    def process_masks(self,masks,value):
        masks = cv2.resize(masks,(self.target_size[1],self.target_size[0]))
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs[:,[0,2]] = self.scale_h*bboxs[:,[0,2]]
        bboxs[:,[1,3]] = self.scale_w*bboxs[:,[1,3]]
        return bboxs    



class Horizon_Flip(BaseTransform):
    def __init__(self, prob=0.5):
        super(Horizon_Flip, self).__init__(prob=prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        image = image[:,::-1]
        return image
    def process_masks(self,masks,value):
        masks = masks[:,::-1]
        return masks
    def process_bboxs(self,bboxs,value):
        bboxs[:,[1,3]] = self.width - bboxs[:,[1,3]]
        bboxs[:,0],bboxs[:,2] = np.min(bboxs[:,[0,2]],axis=1),np.max(bboxs[:,[0,2]],axis=1)
        bboxs[:,1],bboxs[:,3] = np.min(bboxs[:,[1,3]],axis=1),np.max(bboxs[:,[1,3]],axis=1)
        return bboxs
    
class Vertical_Flip(BaseTransform):
    def __init__(self, prob=0.5):
        super(Vertical_Flip, self).__init__(prob=prob)
    def process_image(self,image,value):
        self.height, self.width = image.shape[0:2]
        image = image[::-1]
        return image
    def process_masks(self,masks,value):
        masks = masks[::-1]
        return masks
    def process_bboxs(self,bboxs,value):
        #bboxs[:,[0,2]] = self.height - bboxs[:,[0,2]]
        bboxs[:,[0,2]] = self.height - bboxs[:,[0,2]]
        bboxs[:,0],bboxs[:,2] = np.min(bboxs[:,[0,2]],axis=1),np.max(bboxs[:,[0,2]],axis=1)
        bboxs[:,1],bboxs[:,3] = np.min(bboxs[:,[1,3]],axis=1),np.max(bboxs[:,[1,3]],axis=1)
        return bboxs
    
class RandomCrop(BaseTransform):
    def __init__(self,crop_size=None, value_range=(0,1), prob=0.5):
        super(RandomCrop, self).__init__(value_range, prob)
        self.crop_size = crop_size
    def process_image(self,image,value):
        height, width = image.shape[0:2]
        t_height, t_width = self.crop_size
        self.start_y, self.start_x = int(value*(height-t_height)),int(value*(width-t_width))
        if self.start_y == 0 and self.start_x ==0:
            return image
        image = image[self.start_y:self.start_y+t_height,self.start_x:self.start_x+t_width]
        return image
    def process_masks(self,masks,value):
        t_height, t_width = self.crop_size
        if self.start_y == 0 and self.start_x ==0:
            return masks
        masks = masks[self.start_y:self.start_y+t_height,self.start_x:self.start_x+t_width]
        return masks
    def process_bboxs(self,bboxs,value):
        t_height, t_width = self.crop_size
        if self.start_y == 0 and self.start_x ==0:
            return n_bboxs
        bboxs[:,[0,2]] = np.clip(bboxs[:,[0,2]] - self.start_y,0,t_height)
        bboxs[:,[1,3]] = np.clip(bboxs[:,[1,3]] - self.start_x,0,t_width)
        return bboxs
    
class CenterCrop(BaseTransform):
    def __init__(self,crop_size=None, prob=0.5):
        super(CenterCrop, self).__init__(prob=prob)
        self.crop_size = crop_size
    def process_image(self,image,value):
        height, width = image.shape[0:2]
        t_height, t_width = self.crop_size
        self.start_y = (height-t_height)//2
        self.start_x = (width-t_width)//2
        if self.start_y == 0 and self.start_x ==0:
            return image
        image = image[self.start_y:self.start_y+t_height,self.start_x:self.start_x+t_width]
        return image
    def process_masks(self,masks,value):
        t_height, t_width = self.crop_size
        if self.start_y == 0 and self.start_x ==0:
            return masks
        masks = masks[self.start_y:self.start_y+t_height,self.start_x:self.start_x+t_width]
        return masks
    def process_bboxs(self,bboxs,value):
        n_bboxs = bboxs.copy()
        t_height, t_width = self.crop_size
        if self.start_y == 0 and self.start_x ==0:
            return n_bboxs
        n_bboxs[:,[0,2]] = np.clip(bboxs[:,[0,2]] - self.start_y,0,t_height)
        n_bboxs[:,[1,3]] = np.clip(n_bboxs[:,[1,3]] - self.start_x,0,t_width)
        return n_bboxs    
    
class EvenPad(BaseTransform):
    def __init__(self,factor=32, prob=0.5):
        super(EvenPad, self).__init__(prob=prob)
        self.factor = factor
    def process_image(self,image,value):
        height, width = image.shape[0:2]
        t_height, t_width = np.ceil(height/32)*self.factor,np.ceil(width/32)*self.factor
        
        self.pad_y, self.pad_x = int(t_height-height),int(t_width-width)
        try:
            image = np.pad(image,((0,self.pad_y),(0,self.pad_x),(0,0)),mode='reflect')
        except:
            image = np.pad(image,((0,self.pad_y),(0,self.pad_x)),mode='reflect')
        return image
    def process_masks(self,masks,value):
        try:
            masks = np.pad(masks,((0,self.pad_y),(0,self.pad_x)),mode='reflect')
        except:
            masks = np.pad(masks,((0,self.pad_y),(0,self.pad_x),(0,0)),mode='reflect')
        return masks
    
class Transpose(BaseTransform):
    def __init__(self, prob=0.5):
        super(Transpose, self).__init__(prob=prob)
    def process_image(self,image,value):
        return image.transpose(1, 0, 2) if len(image.shape) > 2 else image.transpose(1, 0)
    def process_masks(self,masks,value):
        return masks.transpose(1, 0, 2) if len(masks.shape) > 2 else masks.transpose(1, 0)
    def process_bboxs(self,bboxs,value):
        n_bboxs = bboxs.copy()
        n_bboxs[:,[0,2]],n_bboxs[:,[1,3]] = n_bboxs[:,[1,3]],n_bboxs[:,[0,2]]
        return n_bboxs


class Clahe(BaseTransform):
    def __init__(self,value_range=(1,4),prob=0.5):
        super(Clahe, self).__init__(value_range,prob)
    def process_image(self,image,value):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(8,8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        image = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
        return image
    
class Blur(BaseTransform):
    def __init__(self,value_range=(3,7),prob=0.5):
        super(Blur, self).__init__(value_range,prob)
    def process_image(self,image,value):
        value = int(value)
        image = cv2.blur(image, (value, value))
        return image
    
class MotionBlur(BaseTransform):
    def __init__(self,value_range=(6,9),prob=0.5):
        super(MotionBlur, self).__init__(value_range,prob)
    def process_image(self,image,value):
        value = int(2*(value//2)+1)
        image = cv2.medianBlur(image, value)
        return image
    
class MedianBlur(BaseTransform):
    def __init__(self,value_range=(2,4),prob=0.5):
        super(MedianBlur, self).__init__(value_range,prob)
    def process_image(self,image,value):
        value = int(value)
        kernel = np.zeros((value, value))
        xs, ys = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
        xe, ye = np.random.randint(0, kernel.shape[1]), np.random.randint(0, kernel.shape[0])
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)
        return cv2.filter2D(image, -1, kernel / np.sum(kernel))
    
class Polosa(BaseTransform):
    def __init__(self,prob=0.5):
        super(Polosa, self).__init__(prob=prob)
    def process_image(self,image,value):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) < 100:
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            xs, ys = np.random.randint(0, empty.shape[1]), np.random.randint(0, empty.shape[0])
            xe, ye = np.random.randint(0, empty.shape[1]), np.random.randint(0, empty.shape[0])
            factor = np.random.randint(1, 10) / 3.
            cv2.line(empty, (xs, ys), (xe, ye), np.max(gray) / factor, thickness=np.random.randint(10, 100))
            empty = cv2.blur(empty, (5, 5))
            empty = empty | gray
            return cv2.cvtColor(empty, cv2.COLOR_GRAY2RGB)
        return image


class ElasticTransform(BaseTransform):
    def __init__(self, alpha=1, sigma=50, alpha_affine=50, prob=0.5):
        super().__init__(prob=prob)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma

    def process_image(self,image,value):
        random_state = np.random.RandomState(1234)
        shape = image.shape
        shape_size = shape[:2]
        self.height,self.width = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        alpha = float(self.alpha)
        sigma = float(self.sigma)
        alpha_affine = float(self.alpha_affine)

        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        self.Matrix = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, self.Matrix, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        dx = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)
        return cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    def process_masks(self,masks,value):
        random_state = np.random.RandomState(1234)
        shape = masks.shape
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        alpha = float(self.alpha)
        sigma = float(self.sigma)
        alpha_affine = float(self.alpha_affine)

        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        masks = cv2.warpAffine(masks, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        dx = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(*shape_size) * 2 - 1), sigma) * alpha)
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)
        return cv2.remap(masks, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    def process_bboxs(self,bboxs,value):
        bboxs = affine_bboxs(self.Matrix,bboxs)
        bboxs = clip_bboxs(bboxs,self.height,self.width)
        return bboxs

class ElasticTransformAffine(BaseTransform):
    def __init__(self, alpha=(0, 5.0), sigma=0.25, prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
    def process_masks(self,masks,value):
        return self.deterministic_processor.augment_image(masks)
    def process_bboxs(self,bboxs,value):
        BoundingBox = [ia.BoundingBox(*bbox[[1,0,3,2]]) for bbox in bboxs]
        bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=self.shape)
        bbs_aug = self.deterministic_processor.augment_bounding_boxes([bbs])[0]
        bboxs = []
        for bbox in bbs_aug.bounding_boxes:
            bboxs.append([bbox.y1,bbox.x1,bbox.y2,bbox.x2])
        return np.array(bboxs)

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)
    return wrapped_function

class HSV(BaseTransform):
    def __init__(self,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, prob=0.5):
        super(HSV, self).__init__(prob=prob)
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
    def process_image(self,image,value):
        hue_shift = np.random.uniform(-self.hue_shift_limit, self.hue_shift_limit)
        sat_shift = np.random.uniform(-self.sat_shift_limit, self.sat_shift_limit)
        val_shift = np.random.uniform(-self.val_shift_limit, self.val_shift_limit)
        # self.value = (hue_shift,sat_shift,val_shift)
        dtype = image.dtype
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
        h, s, v = cv2.split(image)
        h = cv2.add(h, hue_shift)
        h = np.where(h < 0, 255 - h, h)
        h = np.where(h > 255, h - 255, h)
        h = h.astype(dtype)
        s = clip(cv2.add(s, sat_shift), dtype, 255 if dtype == np.uint8 else 1.)
        v = clip(cv2.add(v, val_shift), dtype, 255 if dtype == np.uint8 else 1.)
        image = cv2.merge((h, s, v)).astype(dtype)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

class RGB(BaseTransform):
    def __init__(self,r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, prob=0.5):
        super(RGB, self).__init__(prob=prob)
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit

    def process_image(self,image,value):
        r_shift = np.random.uniform(-self.r_shift_limit, self.r_shift_limit)
        g_shift = np.random.uniform(-self.g_shift_limit, self.g_shift_limit)
        b_shift = np.random.uniform(-self.b_shift_limit, self.b_shift_limit)
        dtype = image.dtype
        image[...,0] = image[...,0] + r_shift
        image[...,1] = image[...,1] + g_shift
        image[...,2] = image[...,2] + b_shift
        image = clip(image, dtype, 255 if dtype == np.uint8 else 1.)
        return image

class Brightness(BaseTransform):
    def __init__(self,value_range=(0.8,1.2), prob=0.5):
        super(Brightness, self).__init__(value_range,prob)
      
    def process_image(self,image,value):
        dtype = image.dtype
        image = value * image
        image = clip(image, dtype, 255 if dtype == np.uint8 else 1.)
        return image

class Invert(BaseTransform):
    def __init__(self,prob=0.5):
        super(Invert, self).__init__(prob=prob)
    def process_image(self,image,value):
        return 255 - image

class InvertAffine(BaseTransform):
    def __init__(self, p=0.25, per_channel=0.5, prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.Invert(p,per_channel=per_channel)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)

class ChannelShuffle(BaseTransform):
    def __init__(self,prob=0.5):
        super(ChannelShuffle, self).__init__(prob=prob)
    def process_image(self,image,value):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        image = image[..., ch_arr]
        return image  

class GaussNoise(BaseTransform):
    def __init__(self,value_range=(10, 50), prob=0.5):
        super(GaussNoise, self).__init__(value_range,prob)
    def process_image(self,image,value):
        row, col, ch = image.shape
        dtype = image.dtype
        mean = int(value)
        # var = 30
        sigma = value**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        gauss = (gauss - np.min(gauss)).astype(np.uint8)
        image = image.astype(np.int32) + gauss
        image = clip(image, dtype, 255 if dtype == np.uint8 else 1.)
        return image

class SaltPepperNoise(BaseTransform):
    def __init__(self,range=(0.2,0.8),prob=0.5):
        super(SaltPepperNoise, self).__init__(range,prob=prob)
    def process_image(self,image,value):
    
        amount = 0.004
        noisy = image
        # Salt mode
        num_salt = np.ceil(amount * image.size * value)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - value))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0
        return noisy  

class PoissonNoise(BaseTransform):
    def __init__(self,prob=0.5):
        super(PoissonNoise, self).__init__(prob=prob)
    def process_image(self,image,value):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy  

class SpeckleNoise(BaseTransform):
    def __init__(self,prob=0.5):
        super(SpeckleNoise, self).__init__(prob=prob)
    def process_image(self,image,value):
        row, col, ch = image.shape
        dtype = image.dtype

        image = clip(image, dtype, 255 if dtype == np.uint8 else 1.)
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss/2
        noisy = clip(noisy, dtype, 255 if dtype == np.uint8 else 1.)
        return noisy 

class Contrast(BaseTransform):
    def __init__(self,value_range=(0.8, 1.2), prob=0.5):
        super(Contrast, self).__init__(value_range,prob)
    def process_image(self,image,value):
        dtype = image.dtype
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = (3.0 * (1.0 - value) / gray.size) * np.sum(gray)
        image = value * image + gray
        image = clip(image, dtype, 255 if dtype == np.uint8 else 1.)
        return image

class ThreeChannelGray(BaseTransform):
    def __init__(self,prob=0.5):
        super(ThreeChannelGray, self).__init__(prob=prob)
    def process_image(self,image,value):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        invgray = 255 - gray
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        if np.mean(invgray) < np.mean(gray):
            invgray, gray = gray, invgray
        res = [invgray, gray, clahe.apply(invgray)]
        return cv2.merge(res)

class Gray(BaseTransform):
    def __init__(self,prob=0.5):
        super(Gray, self).__init__(prob=prob)
    def process_image(self,image,value):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if np.mean(gray) > 127:
            gray = 255 - gray
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class Emboss(BaseTransform):
    def __init__(self, alpha=(0.2, 0.5), strength=(0.2, 0.7), prob=0.5):
        super().__init__(prob=prob)
        self.processor = iaa.Emboss(alpha, strength)
        
    def process_image(self,image,value):
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)

class Superpixels(BaseTransform):
    def __init__(self, p_replace=0.1, n_segments=100, prob=0.5):
        super().__init__(prob=prob)
        self.processor = iaa.Superpixels(p_replace=p_replace, n_segments=n_segments)
        
    def process_image(self,image,value):
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)

class Sharpen(BaseTransform):
    def __init__(self, alpha=(0.2, 0.5), lightness=(0.5, 1.), prob=0.5):
        super().__init__(prob=prob)
        self.processor = iaa.Sharpen(alpha, lightness)
    def process_image(self,image,value):
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)

class AdditiveGaussianNoise(BaseTransform):
    def __init__(self, loc=0, scale=(0.01*255, 0.05*255), prob=0.5):
        super().__init__(prob=prob)
        self.processor = iaa.AdditiveGaussianNoise(loc, scale)
    def process_image(self,image,value):
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)

class PiecewiseAffine(BaseTransform):
    def __init__(self, scale=(0.01, 0.05), nb_rows=4, nb_cols=4, prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.PiecewiseAffine(scale, nb_rows, nb_cols)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
    # def process_bboxs(self,bboxs,value):
    #     BoundingBox = [ia.BoundingBox(*bbox[[1,0,3,2]]) for bbox in bboxs]
    #     bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=self.shape)
    #     bbs_aug = self.deterministic_processor.augment_bounding_boxes([bbs])[0]
    #     bboxs = []
    #     for bbox in bbs_aug.bounding_boxes:
    #         bboxs.append([bbox.y1,bbox.x1,bbox.y2,bbox.x2])
    #     return np.array(bboxs)

class Perspective(BaseTransform):
    def __init__(self, scale=(0.05, 0.1), prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.PerspectiveTransform(scale)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
    def process_masks(self,masks,value):
        return self.deterministic_processor.augment_image(masks)
    def process_bboxs(self,bboxs,value):
        BoundingBox = [ia.BoundingBox(*bbox[[1,0,3,2]]) for bbox in bboxs]
        bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=self.shape)
        bbs_aug = self.deterministic_processor.augment_bounding_boxes([bbs])[0]
        bboxs = []
        for bbox in bbs_aug.bounding_boxes:
            bboxs.append([bbox.y1,bbox.x1,bbox.y2,bbox.x2])
        return np.array(bboxs)

class Equalize(BaseTransform):

    def __init__(self,prob=0.5):
        super(Equalize, self).__init__(prob=prob)
    def process_image(self,image,value):
        Image = PIL.Image.fromarray(image)
        Image = PIL.ImageOps.equalize(Image)
        return np.array(Image)

class Solarize(BaseTransform):
    def __init__(self,prob=0.5):
        super(Solarize, self).__init__(prob=prob)
    def process_image(self,image,value):
        Image = PIL.Image.fromarray(image)
        Image = PIL.ImageOps.solarize(Image)
        return np.array(Image)

class Posterize(BaseTransform):
    def __init__(self,value_range=(3, 8),prob=0.5):
        super(Posterize, self).__init__(value_range,prob=prob)
    def process_image(self,image,value):
        value = int(value)
        Image = PIL.Image.fromarray(image)
        Image = PIL.ImageOps.posterize(Image,value)
        return np.array(Image)

class Cutout(BaseTransform):
    def __init__(self,value_range=(0.1, 0.2),prob=0.5):
        super(Cutout, self).__init__(value_range,prob)
    def process_image(self,image,value):
        Image = PIL.Image.fromarray(image)
        w, h = Image.size
        v = value*Image.size[0]
        x0 = np.random.uniform(0,w-v)
        y0 = np.random.uniform(0,h-v)
        self.xy = (x0, y0, x0+v, y0+v)
        color = (127, 127, 127)
        #Image = Image.copy()
        PIL.ImageDraw.Draw(Image).rectangle(self.xy, color)
        return np.array(Image)
    def process_masks(self,masks,value):
        Masks = PIL.Image.fromarray(masks)
        try:
            color = (0, 0, 0)
            PIL.ImageDraw.Draw(Masks).rectangle(self.xy, color)
        except:
            color = 0
            PIL.ImageDraw.Draw(Masks).rectangle(self.xy, color)
        return np.array(masks)

class Dropout(BaseTransform):
    def __init__(self, p=(0, 0.2), per_channel=0.5, prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.Dropout(p=p,per_channel=per_channel)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
    # def process_masks(self,masks,value):
    #     return self.deterministic_processor.augment_image(masks)
    # def process_bboxs(self,bboxs,value):
    #     BoundingBox = [ia.BoundingBox(*bbox[[1,0,3,2]]) for bbox in bboxs]
    #     bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=self.shape)
    #     bbs_aug = self.deterministic_processor.augment_bounding_boxes([bbs])[0]
    #     bboxs = []
    #     for bbox in bbs_aug.bounding_boxes:
    #         bboxs.append([bbox.y1,bbox.x1,bbox.y2,bbox.x2])
    #     return np.array(bboxs)

class CoarseDropout(BaseTransform):
    def __init__(self, p=(0.0, 0.05), size_percent=(0.02, 0.25), prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.CoarseDropout(p,size_percent=size_percent)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
    # def process_masks(self,masks,value):
    #     return self.deterministic_processor.augment_image(masks)
    # def process_bboxs(self,bboxs,value):
    #     BoundingBox = [ia.BoundingBox(*bbox[[1,0,3,2]]) for bbox in bboxs]
    #     bbs = ia.BoundingBoxesOnImage(BoundingBox, shape=self.shape)
    #     bbs_aug = self.deterministic_processor.augment_bounding_boxes([bbs])[0]
    #     bboxs = []
    #     for bbox in bbs_aug.bounding_boxes:
    #         bboxs.append([bbox.y1,bbox.x1,bbox.y2,bbox.x2])
    #     return np.array(bboxs)
    
class ContrastNormalization(BaseTransform):
    def __init__(self, p=(0.5, 1.5), per_channel=0.5, prob=.5):
        super().__init__(prob=prob)
        self.processor = iaa.ContrastNormalization(p,per_channel=per_channel)
    def process_image(self,image,value):
        self.shape = image.shape
        self.deterministic_processor = self.processor.to_deterministic()
        return self.deterministic_processor.augment_image(image)
