import torchvision
from typing import Tuple, List, Dict
from collections import OrderedDict
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet34
from torchvision.utils import draw_bounding_boxes
from torch.cuda.amp import GradScaler, autocast
from PIL.Image import Image
import PIL
import math
from tqdm import tqdm
import sys
import time
import torch
import numpy as np
import os

def get_model(num_classes):
    
    #uncomment following code for transfer learning for custom backbone
    
    '''from torchvision.models import mobilenet_v2
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
    model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)'''
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None) #, weights_backbone='MobileNet_V3_Large_Weights.IMAGENET1K_V2' weights= "DEFAULT",box_nms_thresh=0.75
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #model.share_memory()
    #set (model.transform = None) if transforms preapplid or set it to your own transforms.
    #model = torch.compile(model) this is not supported on windows
    #print(model)
    
    return model

def eval_forward(model, images, targets):
    
    '''# type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]'''
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "val_loss_objectness": loss_objectness,
        "val_loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {
        "val_loss_classifier": loss_classifier, 
        "val_loss_box_reg": loss_box_reg
        }
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections,
                                            images.image_sizes, 
                                            original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections

def avg(lst):
    return np.mean(lst)



def train_one_epoch(model,optimizer,training_loader,device,epoch,num_accumulation_steps=1,gradient_scaler=None,lr_scheduler=None,logger=None,amp=True):

    #set_num_accumulation steps to 1 if you dont want to accumulate gradients
    
    print("Training")
    
    model.train()
    
    '''if epoch == 0:
        warmup_factor = 1./1000
        warmup_itr = min(1000,len(training_loader)-1)
        
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                   start_factor=warmpu_factor,
                                                   total_iters=warmup_itr)'''
    
    if amp and gradient_scaler is None:
        gradient_scaler = GradScaler() 

    train_losses=[]
    train_losses_dict=[]
    diff_lst = []
    progress_bar = tqdm(training_loader,total=len(training_loader))
    for idx,(images,targets) in enumerate(progress_bar):
        t1 = time.time()
        images = [image.to(device,non_blocking=True) for image in images]
        targets = [{k:v.to(device,non_blocking=True) for k,v in t.items()} for t in targets]
        t2 = time.time()-t1
        
        if amp:
            with torch.autocast(device_type=device,dtype=torch.bfloat16):
                loss_dict = model(images,targets) 
                #model will compute loss automatically passes input AND labels
                #this may only work for detection models since we are not caluclating losses through loss function
            
                losses = sum(loss for loss in loss_dict.values())/num_accumulation_steps
                loss_dict_append = {k:v.item() for k,v in loss_dict.items()}     
                loss_value = losses.item()
                
                train_losses.append(loss_value)
                train_losses_dict.append(loss_dict_append)

            gradient_scaler.scale(losses).backward() #here scaled losses are backpropagated and accumulated so we need to normalize them beforehand
            if (idx+1)%num_accumulation_steps==0 or (idx+1)==len(training_loader):
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                optimizer.zero_grad()
        else:
            loss_dict = model(images,targets) 
            losses = sum(loss for loss in loss_dict.values())/num_accumulation_steps
            loss_dict_append = {k:v.item() for k,v in loss_dict.items()}     
            loss_value = losses.item()
            train_losses.append(loss_value)
            train_losses_dict.append(loss_dict_append)

            if (idx+1)%num_accumulation_steps==0 or (idx+1)==len(training_loader):
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
        
        if not math.isfinite(loss_value): #in case of loss becomes infinite
            print(f"loss = {loss_value}, stop training")
            print([o["image_id"] for o in targets]) #print the image ids of the images that caused the error
            #sys.exit(1)
            os.system("shutdown.exe /s /t 10")
        
        print(optimizer.param_groups[0]['lr'])

        del loss_dict,losses,images,targets #to optimize vram usage
        
        if lr_scheduler is not None and isinstance(lr_scheduler,torch.optim.lr_scheduler.OneCycleLR) and (idx+1)%num_accumulation_steps==0: #for one cycle lr schedule policy. It changes LR after each batch
            lr_scheduler.step()
            if logger is not None:
                logging_index = ((epoch*len(training_loader)/num_accumulation_steps)+((idx+1)/num_accumulation_steps)) #this is some mathfoo to get the current iteration number
                logger.add_scalar('lr',np.array(lr_scheduler.get_last_lr()),logging_index)
        
        progress_bar.set_description(desc=f"Running Training Loss: {loss_value:.4f}")
    
    if lr_scheduler is not None and not isinstance(lr_scheduler,torch.optim.lr_scheduler.OneCycleLR):
        lr_scheduler.step()

    diff = time.time()-t1
    diff_lst.append([diff,t2])

    print(f'processing time = {avg([o[0] for o in diff_lst])}, copying time = {avg([o[1] for o in diff_lst])}')
    del diff_lst,train_losses_dict

    return train_losses, np.array(lr_scheduler.get_last_lr())

def validate(model,val_loader,device,metric=None):
    print('Validating')
    
    model.eval()
    
    diff_lst = []
    val_losses=[]
    val_losses_dict=[]
    prog_bar = tqdm(val_loader,total=len(val_loader))
    for images, targets in prog_bar:
        
        t1 = time.time()
        images = [image.to(device,non_blocking=True) for image in images]
        targets = [{k: v.to(device,non_blocking=True) for k, v in t.items()} for t in targets]
        
        t2 = time.time()-t1

        with torch.no_grad():
            loss_dict_val,detections = eval_forward(model,images,targets)  #other output is detections
        losses_val = sum(loss for loss in loss_dict_val.values())
        loss_dict_append_val = {k:v.item() for k,v in loss_dict_val.items()} 
        loss_value_val = losses_val.item()
        val_losses.append(loss_value_val)
        val_losses_dict.append(loss_dict_append_val)
        if metric is not None:
            metric.update(detections,targets)
        
        # delete these variables
        del loss_dict_val,detections,losses_val,images,targets
        
        
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Running Validation Loss: {loss_value_val:.4f}")

        diff = time.time()-t1
        diff_lst.append([diff,t2])

    print(f'processing time = {avg([o[0] for o in diff_lst])}, copying time = {avg([o[1] for o in diff_lst])}')
    del diff_lst, val_losses_dict

    return val_losses


