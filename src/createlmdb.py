import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import pickle
import lmdb
from dset import has_valid_annotation
import tqdm
from argparse import ArgumentParser


def remove_images_without_annotations(ids,coco_obj):

    ids_new = []
    for img_id in ids:
        ann_ids = coco_obj.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco_obj.loadAnns(ann_ids)
        if has_valid_annotation(anno):
            ids_new.append(img_id)
    # below list contains file ids with faulty bboxes found during EDA even after applying restrictions on annotation
    a = [8226,16985,32813,42393,45854,48854,50973,63543,68796,75743,81166,83730,85730,106324,108944,114381,115754,2486,3041]
    ids_invalid_removed = [i for i in ids if i not in a]
    ids_new = ids_invalid_removed
    return ids_new

#TODO: remove this and simplify adjust_annotations function in dset.py        
def json_to_contiguous_id(coco_obj):
    return {v: i + 1 for i, v in enumerate(coco_obj.getCatIds())}

# A function to serialize the data using pickle
def serialize(data):
    return pickle.dumps(data)

# A function to deserialize the data using pickle
def deserialize(data):
    return pickle.loads(data)


def iterate(coco_obj,ids,image_folder_path):

    for img_id in ids:
        #image
        img_load = coco_obj.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder_path, img_load['file_name'])
        img = Image.open(img_path)
        img = np.array(img,dtype=np.uint8)

        #annotation
        annotation_id = coco_obj.getAnnIds(imgIds=img_id)
        annotation = coco_obj.loadAnns(annotation_id)

        #number of objects in annotation file
        obj_count = len(annotation)

        #classes
        classes = [obj["category_id"] for obj in annotation]
        #classes = [json_to_contiguous_id(coco_obj)[c] for c in classes]
        labels = np.array(classes,dtype=np.uint8)

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

        boxes = np.array(boxes,dtype=np.float16)

        # is crowd
        iscrowd = [obj["iscrowd"] for obj in annotation]
        iscrowd = np.array(iscrowd,dtype=np.uint8)

        #target
        target = {"boxes":boxes,"labels":labels,"image_id":img_id,"iscrowd":iscrowd}
        #add area and segmentation mask if doing semantic segmentation. Make sure to convert segmentation mask from RLE format to normal for serialization

        data = {"image":img,"target":target}

        yield data


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lmdb_path",type=str,help="path to where lmdb dataset will be created")
    parser.add_argument("--coco_path",type=str,help="path to where coco dataset is stored")
    args = parser.parse_args()

    image_dir = os.path.join(args.coco_path,"/coco2017/train2017/")
    annFile = os.path.join(args.coco_path,"/coco2017/annotations/instances_train2017.json")

    coco = COCO(annFile)
    ids = list(sorted(coco.imgs.keys()))
    ids = remove_images_without_annotations(ids,coco)

    # create the lmdb file
    env = lmdb.open(args.lmdb_path, map_size=int(150e9))

    # start a new write transaction
    with env.begin(write=True) as txn:
        itr = iterate(coco,ids,image_dir)
        progress_bar = tqdm(itr,total=len(itr))
        for data in progress_bar:
            # serialize the data
            data = serialize(data)
            txn.put(key=str(data["target"]["image_id"]).encode(), value=data)