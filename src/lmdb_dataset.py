import os
import pickle
import time
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as F
from dset import has_valid_annotation
from pycocotools.coco import COCO


class LMDBDataset(torch.utils.data.Dataset): 

    def __init__(self, lmdb_path, annFile_path,transforms=None,remove_images_without_annotations:bool=True,categories:list=None,num_imgs_per_cat:int=1000):
        
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            annFile: coco annotation file path for annotatoion JSON.
            remove_images_without_annotations: removes files without annotations.
            transforms: image and annotation transformation.
        """
        super(LMDBDataset, self).__init__()
        self.root = lmdb_path
        self.transforms = transforms
        self.coco = COCO(annFile_path) #annotation reader object
        self.ids = list(sorted(self.coco.imgs.keys())) #must be sorted for reproducibility
        self.lmdb_env = None #don't open lmdb until dataset is initialized when using forkserver since lmdb environment cant be pickled. with spawn it needs to be opened in every iteration. 

        self.coco_cat_list = self.coco.loadCats(self.coco.getCatIds()) #returns category list with json file category ids
        for i in categories:
            assert i in [j["name"] for j in self.coco_cat_list], f"{i} not in COCO categories"
        self.categories = categories if categories is not None else [i["name"] for i in self.coco_cat_list]
        self.cat_list = [i for i in self.coco_cat_list if i["name"] in self.categories]
        print(f"cat_list: {self.cat_list}")
        self.supercategories = [i["supercategory"] for i in self.cat_list]  
        self.num_imgs_per_cat = num_imgs_per_cat

        #ids according to given categories
        if self.categories is not None:
            self.ids = self._filter_ids()
            print(f"number of images = {len(self.ids)}")
        
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
            
        self.json_category_id_to_contiguous_id = {v: i + 1 for i, v in enumerate(sorted(list(set(self.coco.getCatIds(catNms=self.categories)))))} # returns a mapping from json category id to contiguous id for given categories. need to sort for reproducibility
        self.contiguous_id_to_json_category_id = {v: k for k, v in self.json_category_id_to_contiguous_id.items()} # returns a reverse mapping
        self.contiguous_id_to_category_name = {k:i["name"] for k,v in self.contiguous_id_to_json_category_id.items() for i in self.cat_list if i["id"] == v} #returns a mapping from contiguous id to category name
        self.contiguous_id_to_supercategory_name = {k:i["supercategory"] for k,v in self.contiguous_id_to_json_category_id.items() for i in self.cat_list if i["id"] == v} #returns a mapping from contiguous id to supercategory name
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
    
    
    def __getitem__(self,idx):

        #open lmdb
        if self.lmdb_env is None:
            self._init_db()
        
        
        '''lmdb.open(self.root,readonly=True,
                             readahead=False, meminit=False)'''
        
        #begin readings
        with self.lmdb_env.begin(write=False) as txn:
            key = self.ids[idx]
            value = txn.get(str(key).encode('ascii'))

        
        data = self.deserialize(value)
        
        #image
        img = data['image']
        img = F.to_tensor(img).permute(1,2,0) #convert to tensor and permute to (C,H,W). it is already (C,H,W) but when use to_tensor it becomes (W,H,C)
        
        #annotations
        annotations = self._filter_annotations(data['target']) if self.categories is not None else data['target']


        labels = torch.from_numpy(annotations['labels']).to(torch.int64)
        iscrowd = torch.from_numpy(annotations['iscrowd']).to(torch.int64)
        image_id = torch.from_numpy(np.array(annotations['image_id'])).to(torch.int64)
        boxes = torch.from_numpy(annotations['boxes']).to(torch.float16)

        target = {"boxes":boxes,"labels":labels,"image_id":image_id,"iscrowd":iscrowd}

        if self.transforms is not None: 
            img,target = self.transforms(img,target)
    
        if idx == self.ids[-1]:
            self.lmdb_env.close()
            self.lmdb_env = None
        
        return img,target
    
    def __len__(self):
        return len(self.ids)
    
    def deserialize(self,data): 
        return pickle.loads(data)
    
    def _init_db(self):
        self.lmdb_env = lmdb.open(self.root,
                             readonly=True,
                             readahead=False, #can be true if enough memory
                             meminit=False,
                             lock=False)
        
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    def _filter_ids(self):
        img_id_lst = []
        for cat in self.categories:
            cat_img_ids = self.coco.getImgIds(catIds=self.coco.getCatIds(catNms=cat))
            np.random.shuffle(cat_img_ids)
            if len(cat_img_ids) < self.num_imgs_per_cat:
                img_id_lst.extend(cat_img_ids)
            else:
                img_id_lst.extend(cat_img_ids[:self.num_imgs_per_cat])
        img_id_lst = list(set(img_id_lst))
        return img_id_lst

    def _filter_annotations(self,anno):
        new_anno = {"labels":np.array([],dtype=np.int64) ,"boxes":[],"iscrowd":np.array([],dtype=np.int64),"image_id":anno["image_id"]}
        anno = self._adjust_contiguous_id(anno) #remove this line if labels saved in LMDB are not already converted to contigious ids.
        for i,j in enumerate(anno["labels"]):
            if j != "Nan" and j in self.contiguous_id_to_category_name.keys():
                if self.contiguous_id_to_category_name[j] in self.categories:
                    new_anno["labels"] = np.append(new_anno["labels"],j)
                    new_anno["boxes"].append(anno["boxes"][i])
                    new_anno["iscrowd"]= np.append(new_anno["iscrowd"],anno["iscrowd"][i])
        new_anno["boxes"] = np.array(new_anno["boxes"],dtype=np.float16)
        assert new_anno["boxes"].shape==(len(new_anno["labels"]),4), f"boxes shape not correct. current box shape = {new_anno['boxes'].shape}. \n original_annotation = {anno}. \n new_annotation = {new_anno}"
        return new_anno
    

    #following function can be removed if labels saved in LMDB are not already converted to contigious ids.
    def _adjust_contiguous_id(self,anno):
        #we cant use the instance attributes for mappings because they take in given categories and not all categories
        json_2_contigious = {v:i+1 for i,v in enumerate(self.coco.getCatIds())} #json id to contigious id same mapping as the original one in creatlmdb.py
        contigious_2_json = {v:k for k,v in json_2_contigious.items()} #reverse mapping
        adjustment_map = {k:i for k,v in contigious_2_json.items() for i,j in self.contiguous_id_to_json_category_id.items() if v==j}
        anno["labels"] = [adjustment_map[i] if i in adjustment_map.keys() else "Nan" for i in anno["labels"]]   # # if only one condition then use  [adjustment_map[i] for i in anno["labels"] if i in adjustment_map.keys()]
        return anno
    
    def show_annotated_image(self,idx):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        img,target = self.__getitem__(idx)
        img = img.permute(1,2,0)
        img = 255*img 
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


    
    '''def _filter_annotations(self,anno):
        new_anno = {}
        for k,v in anno.items():
            new_anno[k] = [val for i,val in enumerate(v) if self.contiguous_id_to_category_name[anno["labels"][i]] in self.categories]
        return new_anno'''
        
        