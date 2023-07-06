import torch
import torchvision.transforms.functional as F
import cv2 
import numpy as np
import os
from dset import COCODataset
from training_utils import get_model
from argparse import ArgumentParser



def init_dset(coco_root,annFile):
    dataset = COCODataset(coco_root,annFile)
    return dataset

def init_model(model_path,device,num_classes=15):
    #from FasterRCNNLIghtningModule import FasterRCNNLightningModule #uncomment if model saved as lightning module. Model path will become checkpoint path.
    #model = FasterRCNNLightningModule.load_from_checkpoint(model_path,num_classes=15,training_set=dataset_train) 
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval().to(device)
    return model

def get_image(img_path):
    img = cv2.imread(img_path)
    img = np.array(img)
    img = F.to_tensor(img)
    img = img.unsqueeze(0).to(device)
    return img

def detect(model_path,img_path,device,detection_threshold=0.50):
    model = init_model(model_path,device)
    img = get_image(img_path)
    with torch.no_grad():
        detections = model(img)
    boxes,classes,scores = detections[0]["boxes"].to(torch.int64).tolist(),detections[0]["labels"].tolist(),detections[0]["scores"].tolist()
    boxes,classes = [boxes[i] for i in range(len(boxes)) if scores[i] > detection_threshold],[classes[i] for i in range(len(classes)) if scores[i] > detection_threshold]
    return boxes,classes

def draw_boxes(img_path,boxes,categories,coco_root,annFile,classes):
    image = get_image(img_path)
    dataset = init_dset(coco_root,annFile)
    categories = [dataset.contiguous_id_to_category_name[i] for i in classes]
    if len(boxes) >1:
        for i in range(len(boxes)):
            print(len(boxes))
            x1,y1,x2,y2 = boxes[i]
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(image,str(categories[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    elif len(boxes) == 1:
        x1,y1,x2,y2 = boxes[0]
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image,str(categories[0]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    else:
        print("No objects detected")
    cv2.imshow(img_path,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__=="__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = ArgumentParser()
    parser.add_argument("--model_path",type=str,help="Path to the model")
    parser.add_argument("--img_path",type=str,help="Path to the image")
    parser.add_argument("--coco_root",type=str,help="Path to COCO dataset root folder")
    parser.add_argument("--detection_threshold",type=float,default=0.50,help="Threshold for detection")
    parser.add_argument("--num_classes",type=int,default=15,help="Number of classes. Same number as in training")
    args = parser.parse_args()

    annFile_path = os.path.join(args.coco_root,"/coco2017/annotations/instances_train2017.json")
    boxes,classes = detect(args.model_path,args.img_path,device,args.detection_threshold)
    draw_boxes(args.img_path,boxes,classes,args.coco_root,annFile_path,classes)
