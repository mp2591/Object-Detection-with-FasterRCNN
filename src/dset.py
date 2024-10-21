
import torch
from pycocotools.coco import COCO
import PIL
import os
import torchvision
import numpy as np


#@torch.compile
def _has_an_empty_bbox(ann):
     # will remove the sample from training if any box has close to zero area or negative width or height (initial coco bbox format is xywh so we can just check for zero or negative values)
    return any(any(o <= 1 for o in obj["bbox"][2:]) for obj in ann) # change outer function to all() if you want to keep a sample even if it has one or more zero area bboxes.

#@torch.compile
def has_valid_annotation(ann):
    # if it's empty or has area close to zero, there is no annotation
    if len(ann) == 0 or _has_an_empty_bbox(ann):
        return False
    return True


class COCODataset(torch.utils.data.Dataset): 
    
    def __init__(self, root, annFile, transforms=None, remove_images_without_annotations=True, categories=None):
        
        
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            annFile: coco annotation file path for annotatoion JSON.
            remove_images_without_annotations: removes files without annotations.
            transform: image transformer.
            bbox_transforms: bounding box transformer.
            
        """
        super(COCODataset,self).__init__()
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys())) #must be sorted for reproducibility
        self.categories = categories

        if self.categories is not None:
            self.ids = self._filter_ids()
        
        #this will remove images without annotation from ids
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id) 
            # below list contains file ids with faulty bboxes found during EDA even after applying restrictions on annotation
            a = [8226,16985,32813,42393,45854,48854,50973,63543,68796,75743,81166,83730,85730,106324,108944,114381,115754,2486,3041]
            ids_invalid_removed = [i for i in ids if i not in a]
            self.ids = ids_invalid_removed
        
        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(list(set(self.coco.getCatIds())))} # returns a mapping from json category id to contiguous id
        self.contiguous_id_to_json_category_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()} # returns a reverse mapping
        self.cat_list = self.coco.loadCats(self.coco.getCatIds()) #returns category list with json ids
        self.contiguous_id_to_category_name = {k:i["name"] for k,v in self.contiguous_id_to_json_category_id.items() for i in self.cat_list if i["id"] == v} #returns a mapping from contiguous id to category name
        self.contiguous_id_to_supercategory_name = {k:i["supercategory"] for k,v in self.contiguous_id_to_json_category_id.items() for i in self.cat_list if i["id"] == v} #returns a mapping from contiguous id to supercategory name
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        
    def __getitem__(self,idx):
        
        #load image
        image_id = self.ids[idx]
        image_path = self.coco.loadImgs(image_id)[0]['file_name']
        img = PIL.Image.open(os.path.join(self.root, image_path))
        img = torchvision.transforms.functional.to_tensor(img)
        
        #load annotation
        annotation_id = self.coco.getAnnIds(imgIds=image_id)
        annotation = self.coco.loadAnns(annotation_id)

        if self.categories is not None:
            annotation = self._filter_annotations(annotation)
        
        #number of objects in annotation file
        obj_count = len(annotation)
        
        #can filter out uncrowded by using -[obj for obj in annotation if iscrowd==0]-
        
        #convert image id to tensor after using it in annotation dont do it before or it wont call annotation file

        ''' In the coco dataset the format for each object in a given image is as followiung:
        [{segmentation:[[]], area:,iscrowd:,image_id:,bbox:[x,y,width,height],category_id:},
        {segmentation:[[]], area:,iscrowd:,image_id:,bbox:[x,y,width,height],category_id:},obj3,obj4,...]
        '''
        
        #classes
        classes = [obj["category_id"] for obj in annotation]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]


        labels = torch.tensor(classes, dtype=torch.int64) #,device="cuda"
        
        '''#area - not needed since this is only detection task and not segmentation task
        areas = [annotation[i]['area'] for i in range(obj_count)]
        areas = torch.as_tensor(areas, dtype=torch.float32)'''
        
        # Iscrowd
        iscrowd = [obj["iscrowd"] for obj in annotation]
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64) #,device="cuda" --> dont send to cuda in the dataset class if you want to use multiprocessing
        
        #image id
        image_id = torch.as_tensor([image_id]) #,device="cuda"
        
        
        #bounding boxes 
        
        '''
        bouding boxes need to be converted from xywh to xyxy format
        '''
        
        boxes = [
            [annotation[i]['bbox'][0],
             annotation[i]['bbox'][1],
             annotation[i]['bbox'][0] + annotation[i]['bbox'][2],
             annotation[i]['bbox'][1] + annotation[i]['bbox'][3]] for i in range(obj_count)
        ]
        
        boxes = torch.as_tensor(boxes,dtype=torch.float16) #,device="cuda"
        
        '''
        we can also do similar conversions for segmentation mask and other inputs
        but since this is only object detection task we dont need that.
        '''
        
        #target
        target = {"boxes":boxes,"labels":labels,"image_id":image_id,"iscrowd":iscrowd}
        
        if self.transforms is not None: 
            img,target = self.transforms(img,target) 
            
        return img,target
    
    def __len__(self):
        return len(self.ids)
    
    def get_img_info(self, index):
        """This function will return image info for an image.

        Args:
            index: index of image to get info for.
        """
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    def _filter_ids(self, num_imgs_per_cat=1000):
        """This function will filter out image ids for categories that are not in self.categories.
        
        Args:
            num_imgs_per_cat: number of images per category.
        
        """
        img_id_lst = []
        for cat in self.categories:
            cat_img_ids = self.coco.getImgIds(catIds=[cat])
            np.random.shuffle(cat_img_ids)
            if len(cat_img_ids) < num_imgs_per_cat:
                img_id_lst.extend(cat_img_ids)
            else:
                img_id_lst.extend(cat_img_ids[:num_imgs_per_cat])
        img_id_lst = list(set(img_id_lst))
        return img_id_lst
    
    def _filter_annotations(self,anno):
        """This function will filter out annotations for categories that are not in self.categories.
        
        Args:
            anno: annotation file for an image.
        
        """
        return [o for o in anno if self.coco.loadCats[o["name"]] in self.categories] #classes are o["labels"] for lmdb class. so different implementation for lmdb dataset.
    
    def show_annotated_image(self,idx):
        """This function will show low rez test image.
        

        Args:
            idx: index of image to show.
            
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        img,target = self.__getitem__(idx)
        img = img.permute(1,2,0)
        img = 255*img #denormalize
        img = np.array(img,dtype=np.uint8)
        #img = Image.fromarray(img)
        plt.imshow(img,interpolation="nearest")
        ax = plt.gca()
        for i in range(len(target["labels"])):
            x1,y1,x2,y2 = target["boxes"][i]
            width = x2-x1
            height = y2-y1
            rect = Rectangle((x1,y1),width,height,fill=False,color="red")
            ax.add_patch(rect)
            ax.text(x1,y1,s=self.contiguous_id_to_category_name[target["labels"][i].item()],color="red")
        plt.show()
        plt.close()
            
    
@torch.compile
def c_fn(batch):
    
    '''This function resolves the issue of batch creation where tensor would not be created from diffent
    dimensions for different number of bounding boxes in targets.
    
    Args:batch from dataloader
    '''
    return tuple(zip(*batch))
