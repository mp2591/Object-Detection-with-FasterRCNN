# Depedencies
#nvidia-smi
#%env PYTORCH_CUDA_ALLOC_CONF = max_split_size_mb:50 #to avoid fragmentation in gpu memory
#%env
#!pip install pycocotools
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import sys
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import datetime
from dset import COCODataset, c_fn
from mytransforms import TCompose, Resized
from lmdb_dataset import LMDBDataset
from training_utils import *
from argparse import ArgumentParser
from EarlyStopping import EarlyStopper
from torch.utils.tensorboard import SummaryWriter
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP as MeanAveragePrecision   #torchmetrics.detection.mean_ap for torchmetrics newest version

import torchmetrics 
print(torchmetrics.__version__)

def system_shutdown():
    os.system("shutdown.exe /s /t 10")

if __name__=="__main__":
    
    SEED = 137
    np.random.seed(SEED)

    if sys.platform == 'linux':
        torch.multiprocessing.set_start_method('forkserver',force=True)
    elif sys.platform == ('win32' or 'Cywin' or 'darwin'):
        torch.multiprocessing.set_start_method('spawn',force=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


    # data preperations

    root = "COCO/coco2017"  #"/kaggle/input/coco-2017-dataset/coco2017"

    imgdir_train = os.path.join(root, 'train2017') 

    annFile_train = os.path.join(root, 'annotations/instances_train2017.json') 

    imgdir_val = os.path.join(root, 'val2017') 

    annFile_val = os.path.join(root, 'annotations/instances_val2017.json')

    if sys.platform == 'linux':
        lmdb_train = r"/mnt/g/LMDB/COCO_train"
        lmdb_val = r"/mnt/g/LMDB/COCO_val"
    elif sys.platform == ('win32' or 'Cywin'):
        lmdb_train = os.path.abspath("G:\LMDB\COCO_train")
        lmdb_val = os.path.abspath("G:\LMDB\COCO_val")
    '''elif sys.platform == "darwin":
            enter mac os path for lmdb database'''
    

    dataset_categories = ["truck","person","train","boat","airplane","car","motorcycle","bicycle","bus","traffic light","fire hydrant","stop sign","parking meter","bench"] #,"car","motorcycle","bicycle"

    dataset_train = LMDBDataset(lmdb_train,annFile_train,categories=dataset_categories,num_imgs_per_cat=1000) # COCODataset(imgdir_train,annFile_train) if you want to use original coco dataset without memory mapping 
    dataset_valid = LMDBDataset(lmdb_val,annFile_val,categories=dataset_categories,num_imgs_per_cat=200)  # COCODataset(imgdir_val,annFile_val)  
    num_classes = len(dataset_train.categories)+1 if dataset_train.categories is not None else len(dataset_train.cat_list)+1 #+1 for background class

    #computing subset indices. seed already set so it will be reproducible
    #subset_indices_train = np.random.randint(0,len(dataset_train)+1,size=8192)
    #subset_indices_valid = np.random.randint(0,len(dataset_valid)+1,size=5000) #

    #dataset_train = torch.utils.data.Subset(dataset_train,indices=subset_indices_train)
    #dataset_valid = torch.utils.data.Subset(dataset_valid,indices=subset_indices_valid)
    



    '''def show(sample):
        

        image, target = sample
        if isinstance(image,Image):
            image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image, torch.uint8)
        annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

        fig, ax = plt.subplots()
        ax.imshow(annotated_image.permute(1, 2, 0).numpy())
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        fig.tight_layout()

        fig.show()
        

    show(sample)'''


    # data loading
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            collate_fn=c_fn,
                                            pin_memory=True,
                                            num_workers=7,
                                            persistent_workers=True,
                                            prefetch_factor=3)                   #num_workers=0,persistent_workers=True,prefetch_factor=2

    val_loader = torch.utils.data.DataLoader(dataset_valid,
                                            batch_size=batch_size,
                                            collate_fn=c_fn,
                                            pin_memory=True,
                                            num_workers=7,
                                            persistent_workers=True,
                                            prefetch_factor=3)


    #show image
    #dataset_train.show_annotated_image(9562) #dont use it with multi processing dataloader. it will cause deadlock

    #testing code
    '''images,targets = next(iter(val_loader))


    images = [image.to(device) for image in images]
    targets = [{k:v.to(device) for k,v in tar.items()} for tar in targets]

    model = get_model()
    model.to(device)
    model.eval()
    with torch.no_grad():
        loss_dict_val,detections = eval_forward(model,images,targets)
    loss_dict_val
    '''
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)

    # get the model using our helper function
    model = get_model(num_classes)

    # model fine tuning, training, testing
    model.to("cuda")

    num_epochs = 80
    num_accumulation_steps = 1
    total_steps = num_epochs*math.floor((len(train_loader)/num_accumulation_steps))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, nesterov=True, weight_decay=1e-4) #Adam(params,0.001,weight_decay=1e-4) 
    lr_steps = math.floor(len(train_loader)/num_accumulation_steps)
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.01,total_steps=total_steps,pct_start=0.5,div_factor=1e3) #StepLR(optimizer,step_size=3,gamma=0.1) ,verbose=True
    scaler = GradScaler()
    early_stopper = EarlyStopper(patience=15,threshold=0.04,improvement_req=0.001,threshold_map=0.005,improvement_req_map=0.005,model=model)
    logger = SummaryWriter()
    metric = MeanAveragePrecision(class_metrics=True,dist_sync_on_step=True) #iou_type='bbox',iou_thresholds=[0.5],rec_thresholds=[0.5],max_detection_thresholds=[10]
    t_1 = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    for epoch in range(num_epochs):
        # train for one epoch
        train_loss,lr = train_one_epoch(model, optimizer, train_loader, "cuda",epoch,num_accumulation_steps,scaler,lr_scheduler,logger=logger,amp=False) #During training, running loss should be focused since model learns through back propagation on each batch iteration.
        train_loss_last = train_loss[-1]
        avg_train_loss = np.mean(train_loss)
        print("Epoch: {} Epoch Train Loss: {}".format(epoch,avg_train_loss))

        val_loss = validate(model, val_loader, "cuda",metric=metric)
        avg_val_loss = np.mean(val_loss) #avearge loss for the epoch. For validation loss this should be focused since model dosent learn anything through back propagation.
        print("Epoch: {} Epoch Validation Loss: {}".format(epoch,avg_val_loss))

        #mean average precision
        mAP_state_compute = metric.compute() #TO DO: this takes a lot of time between epochs so if you can do a custom implementation, it would be better.
        metric.reset()
        mAP = {k:mAP_state_compute[k].item() for k in mAP_state_compute.keys() if k != "map_per_class" and k != "mar_100_per_class"} #{k:v.item() for k,v in mAP_state_compute.items()} 
        mAP_class = {dataset_train.contiguous_id_to_category_name[i+1]:mAP_state_compute["map_per_class"][i].item() for i,_ in enumerate(mAP_state_compute["map_per_class"])}
        mAR_class = {dataset_train.contiguous_id_to_category_name[i+1]:mAP_state_compute["mar_100_per_class"][i].item() for i,_ in enumerate(mAP_state_compute["mar_100_per_class"])}
        print("Epoch: {} Epoch mAP: {}".format(epoch,mAP["map"]))

        #logging
        logger.add_scalars("run_"+str(t_1),{'Running Loss/Train': train_loss[-1], 'AVG Loss/train': avg_train_loss, 'AVG Loss/val': avg_val_loss}, epoch)
        logger.add_scalars("run_"+str(t_1)+"_map",mAP,epoch)
        logger.add_scalars("run_"+str(t_1)+"_map_",mAP_class,epoch)
        logger.add_scalars("run_"+str(t_1)+"_mar100_",mAR_class,epoch)

        '''for i,v in enumerate(mAP_state_compute["map_per_class"]):
            logger.add_scalar("mAP_"+str(dataset_train.contiguous_id_to_category_name[i+1]),v.item(),epoch)
        for i,v in enumerate(mAP_state_compute["mar_100_per_class"]):
            logger.add_scalar("mar_100_"+str(dataset_train.contiguous_id_to_category_name[i+1]),v.item(),epoch)'''

        #calling early stopper with its __call__ method
        early_stopper(mAP["map"],avg_val_loss,epoch) 
        print(early_stopper.early_stop)
        print(avg_val_loss-early_stopper.min_val_loss)
        print(mAP["map"]-early_stopper.max_val_map)

        if early_stopper.early_stop:
            model.load_state_dict(early_stopper.best_state_dict)
            model.eval()
            torch.save(early_stopper.best_state_dict,'model_state_dict_epoch_'+str(early_stopper.epoch)+'.pth')
            break

    logger.close()

    #saving model
    torch.save(model,f'model_no_early_stop_run ={t_1}.pt')
    #saving model state dict    
    torch.save(model.state_dict(),f'model_state_dict_no_early_stop_run ={t_1}.pth')
    torch.save(early_stopper.best_state_dict,'best_model_state_dict_end of_run.epoch_'+str(early_stopper.epoch)+'.pth')


    gc.collect()
    torch.cuda.empty_cache()
    
    #model = torch.jit.script(model) # Export to TorchScript
    #torch.save(model,'model_scripted_lmdb'+str(time.time())+'.pt')

    system_shutdown()
        
 
'''a = [8226,16985,32813,42393,45854,48854,50973,63543,68796,75743,81166,83730,85730,106324,108944,114381,115754,2486,3041]
for i in a:
    _,target = dataset_train[i]
    print(target["boxes"])'''



#if __name__=="__main__":


    