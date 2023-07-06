import os
import sys
from typing import Any, Tuple, List, Dict
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision 
from collections import OrderedDict
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers
from torchvision.models.detection import FasterRCNN

import torch
import numpy as np


try:
    from torchmetrics import MeanAveragePrecision
except:
    from torchmetrics import MAP


class FasterRCNNLightningModule(pl.LightningModule):

    def __init__(self,num_classes:int,
                 training_set,lr:float=0.0001,
                 sgd_momentum:float=0.9,
                 sgd_wd:float=0.0005,
                 batch_size:int=16,
                 transforms=None,
                 model_architecture:str="mobilenet_v3",
                 pretrained:bool=True,
                 pretrained_backbone_only:bool=False,
                 optim_type:str="SGD",
                 scheduler_type:str="stepLR"):
        
        super(FasterRCNNLightningModule,self).__init__()
        self.map = MAP()
        self.model = self.get_model(num_classes,model_architecture,pretrained,pretrained_backbone_only)
        self.train_transforms = transforms 
        self.lr = lr
        self.training_set = training_set
        self.batch_size = batch_size
        self.optim_type = optim_type
        self.scheduler_type = scheduler_type
        #self.training_epoch_output = []
        #self.validation_epoch_output = []
        self.save_hyperparameters("lr","sgd_momentum","sgd_wd")
    
    def forward(self,input):
        self.model.eval()
        return self.model(input)
    
    def training_step(self,batch,batch_idx):
        images,targets = batch
        loss_dict = self.model(images,targets) 
        loss = sum(loss for loss in loss_dict.values())
        self.log('running_train_loss',loss,prog_bar=True,on_step=True,on_epoch=True,logger=True,batch_size=self.batch_size)
        self.log_dict(loss_dict,on_step=True,on_epoch=True,logger=True,batch_size=self.batch_size)
        #print(self.trainer.optimizers[0].param_groups[0]['lr'])
        if not torch.isfinite(loss): #in case of loss becomes infinite
            print(f"loss = {loss}, stop training")
            sys.exit("Training loss is infinite. Training stopped.")
            #os.system("shutdown.exe /s /t 10")
        #self.training_epoch_output.append(loss)
        return loss
        
    
    '''def on_train_epoch_end(self) -> None:
        epoch_train_loss = torch.stack(self.training_epoch_output).mean()
        self.log('epoch_train_loss',epoch_train_loss,prog_bar=True)
        self.training_epoch_output = []'''
    
    def validation_step(self,batch,batch_idx):
        images,targets = batch
        val_loss_dict, detections = self.validate_with_losses(images,targets)
        val_loss = sum(loss for loss in val_loss_dict.values())
        self.map.update(detections,targets)
        self.log('val_loss',val_loss,prog_bar=True,on_step=True,on_epoch=True,logger=True,batch_size=self.batch_size)
        self.log_dict(val_loss_dict,on_step=True,on_epoch=True,logger=True,batch_size=self.batch_size)
        #self.validation_epoch_output.append(val_loss)

    def on_validation_epoch_end(self) -> None:
        #epoch_val_loss = torch.stack(self.validation_epoch_output).mean()
        #self.log('epoch_val_loss',epoch_val_loss,prog_bar=True,logger=True,)
        #self.validation_epoch_output = []
        mAP_state_compute = self.map.compute()
        mAP = {k:mAP_state_compute[k] for k in mAP_state_compute.keys() if k != "map_per_class" and k != "mar_100_per_class"} #{k:v.item() for k,v in mAP_state_compute.items()} 
        mAP_class = {self.training_set.contiguous_id_to_category_name[i+1]:mAP_state_compute["map_per_class"][i] for i,_ in enumerate(mAP_state_compute["map_per_class"])}
        mAR_class = {self.training_set.contiguous_id_to_category_name[i+1]:mAP_state_compute["mar_100_per_class"][i] for i,_ in enumerate(mAP_state_compute["mar_100_per_class"])}
        self.log('Mean Average Precision',mAP["map"],prog_bar=True,logger=True)
        self.log_dict(mAP,logger=True)
        self.log_dict(mAP_class,logger=True)
        self.log_dict(mAR_class,logger=True)
        self.map.reset()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        #configering optimezer
        if self.optim_type == "SGD":
            optimizer =  torch.optim.SGD(params,
                                         lr=self.hparams.lr,
                                         momentum=self.hparams.sgd_momentum,
                                         weight_decay=self.hparams.sgd_wd) 
        elif self.optim_type == "Adam":
            optimizer =  torch.optim.Adam(params,
                                          self.hparams.lr/10,
                                          weight_decay=self.hparams.sgd_wd) 
        #configering scheduler
        if self.scheduler_type == "stepLR":
            stepsize = 5*self.trainer.estimated_stepping_batches/self.trainer.max_epochs
            scheduler =  torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                           base_lr=self.hparams.lr/10,
                                                           max_lr=self.hparams.lr*10,
                                                           step_size_up=stepsize,
                                                           step_size_down=stepsize,
                                                           base_momentum=0.85,
                                                           max_momentum=0.95,
                                                           cycle_momentum=True)
            interval = "step"
        elif self.scheduler_type == "cosine":
            stepsize = 5*self.trainer.estimated_stepping_batches/self.trainer.max_epochs
            scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,int(stepsize))  #, eta_min=self.hparams.lr/10
            interval = "step"
        elif self.scheduler_type == "onecycle":
            scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                             max_lr=self.hparams.lr*5,
                                                             total_steps=self.trainer.estimated_stepping_batches,
                                                             pct_start=0.50,
                                                             div_factor=5,
                                                             anneal_strategy="linear")
            interval = "step"
        elif self.scheduler_type == "plateau":
            scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode="max",
                                                                    factor=0.75,
                                                                    patience=5,
                                                                    min_lr=0)
            interval = "epoch"

        return {
            'optimizer':optimizer,
            'lr_scheduler':{"scheduler":scheduler,
                            "monitor":"Mean Average Precision",
                            "interval":interval,
                            "frequency":1}, 
        }
    
    @staticmethod
    def get_model(num_classes,model_architecture,pretrained,pretrained_backbone_only):
    
        #uncomment following code for transfer learning with custom backbone.
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
        
        if model_architecture == "mobilenet_v3":
            if pretrained:
                model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
                pretrained_backbone_only = False
                '''model.box_fg_iou_thresh = 0.7 box_nms_thresh=0.70,
                model.box_bg_iou_thresh = 0.7'''
            elif pretrained_backbone_only:
                model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V2")
            else:
                model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        elif model_architecture == "resnet50":
            if pretrained:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
                pretrained_backbone_only = False
                '''model.box_fg_iou_thresh = 0.7 box_nms_thresh=0.70,
                model.box_bg_iou_thresh = 0.7'''        
            elif pretrained_backbone_only:
                model = model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights_backbone="ResNet50_Weights.IMAGENET1K_V2")
            else:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
        model.share_memory() #if ddp is used for training
        #set (model.transform = None) if transforms preapplid or set it to your own transforms
        return model
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    def validate_with_losses(self, images, targets):
    
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

        images, targets = self.model.transform(images, targets)

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

        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        self.model.rpn.training=True
        #model.roi_heads.training=True


        #####proposals, proposal_losses = model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = self.model.rpn.head(features_rpn)
        anchors = self.model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        
        proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = self.model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        proposal_losses = {}
        assert targets is not None
        labels, matched_gt_boxes = self.model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = self.model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "val_loss_objectness": loss_objectness,
            "val_loss_rpn_box_reg": loss_rpn_box_reg,
        }

        #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = self.model.roi_heads.select_training_samples(proposals, targets)
        box_features = self.model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

        
        detector_losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {
            "val_loss_classifier": loss_classifier, 
            "val_loss_box_reg": loss_box_reg
            }
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        result: List[Dict[str, torch.Tensor]] = [{
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                } for i in range(num_images)]
        
        detections = result
        detections = self.model.transform.postprocess(detections,
                                                images.image_sizes, 
                                                original_image_sizes)  # type: ignore[operator]
        self.model.rpn.training=False
        self.model.roi_heads.training=False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections