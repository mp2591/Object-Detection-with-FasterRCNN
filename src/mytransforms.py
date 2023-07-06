import numpy as np
import torch
import torchvision


class Resized(object):
    
    """Resize the image and given target element in a sample to a given size.

    Args:
    
        img_output_sz: output size of for transformed image. (width,height)
        
    """
    
    def __init__(self,img_input_sz):
        self.img_input_sz = img_input_sz

    #@torch.compile    
    def __call__(self,image,target):
        boxes = target['boxes']
        height_ratio = self.img_input_sz[0]/image.size[1] #pil image (width,height) 
        width_ratio = self.img_input_sz[1]/image.size[0] #pil and img tensor have opposite format
        
        image = torchvision.transforms.functional.resize(image,self.img_input_sz)
        image = torchvision.transforms.functional.to_tensor(image)
        
        scaled_boxes = []
        for box in boxes:
            x1 = np.round(width_ratio*box[0])
            y1 = np.round(height_ratio*box[1])
            x2 = np.round(width_ratio*box[2])
            y2 = np.round(height_ratio*box[3])
            scaled_boxes.append([x1,y1,x2,y2])
        target["boxes"] = torch.as_tensor(scaled_boxes) #device="cuda"
        
        return image,target
    
 
class Normalized(object): #no need to use it since detection models have internal normalization
    
    '''This class will normalize image that has gone through ToTensor or to_tensor tranform.
    
    Args:
        image,target
    
    '''
    
    def __init__(self):
        pass

    #@torch.compile    
    def __call__(self,image,target):
        mean, std = image.mean([1,2]), image.std([1,2])
        image = torchvision.transforms.functional.normalize(image,mean,std)
        
        return image,target
        

class TCompose(object):
    
    '''This class is a replacement for transforms.Compose class.
    It runs into problems when resizing image and target at the same time.
    
    Args: 
        transforms: list of transforms
    
    '''
    
    def __init__(self, transforms):
        self.transforms = transforms

    #@torch.compile
    def __call__(self, img, tar):
        for t in self.transforms:
            img, tar = t(img, tar)
        return img, tar