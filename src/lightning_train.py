import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from dset import COCODataset, c_fn
from mytransforms import TCompose, Resized
from lmdb_dataset import LMDBDataset
from training_utils import *
from argparse import ArgumentParser
from FasterRCNNLIghtningModule import FasterRCNNLightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, StochasticWeightAveraging


import torchmetrics 
print(torchmetrics.__version__)

def show(sample):
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

if __name__=="__main__":

    #arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset",type=str,default="lmdb_COCO",help="dataset to use for training. COCO for original COCO dataset, lmdb_COCO for lmdb dataset") 
    parser.add_argument("--show_sample",type=bool,default=False,help="whether to show a sample from the dataset or not")
    parser.add_argument("--seed",type=int,default=137,help="seed to use for reproducibility")
    parser.add_argument("--batch_size",type=int,default=16,help="batch size to use for training")
    parser.add_argument("--lr",type=float,default=5e-3,help="learning rate")
    parser.add_argument("--num_workers",type=int,default=6,help="number of workers to use for data loading")
    parser.add_argument("--prefetch_factor",type=int,default=2,help="number of samples loaded in advance by each worker")
    parser.add_argument("--accumulate_grad_batches",type=int,default=2,help="number of batches to accumulate gradients for")
    parser.add_argument("--gradient_clip_val",type=float,default=0.5,help="value to clip gradients" )
    parser.add_argument("--gradient_clip_algorithm",type=str,default="norm",help="gradient clipping algorithm to use. norm for norm clipping, value for value clipping")
    parser.add_argument("--precision",type=str,default="16-mixed",help="precision to use for training. 16-mixed for mixed precision,16 for half precision, 32 for single precision, bf16 for bfloat16 precision")
    parser.add_argument("--accelerator",type=str,default="gpu",help="type of accelerator to use")
    parser.add_argument("--min_epochs",type=int,default=100,help="minimum number of epochs to train for")
    parser.add_argument("--max_epochs",type=int,default=150,help="maximum number of epochs to train for")
    parser.add_argument("--patience",type=int,default=15,help="number of epochs to wait before early stopping")
    parser.add_argument("--mode",type=str,default="max",help="max or min for early stopping")
    parser.add_argument("--dirpath",type=str,default="checkpoints",help="directory path to save checkpoints")
    parser.add_argument("--swa_epoch_start",type=float,default=0.5,help="epoch to start stochastic weight averaging")
    parser.add_argument("--monitor",type=str,default="Mean Average Precision",help="metric to monitor for early stopping and checkpointing")
    parser.add_argument("--save_weights_only",type=bool,default=True,help="whether to save only weights or whole model")
    parser.add_argument("--lr_finder",type=bool,default=False,help="whether to tune lr or not")
    parser.add_argument("--num_training",type=int,default=2500,help="number of training steps to perform during lr tuning")
    parser.add_argument("--lmdb_train_path",type=str,default=r"/mnt/g/LMDB/COCO_train",help="path to lmdb database for training set. If not created then execute lmdb_dataset.py to create one")
    parser.add_argument("--lmdb_val_path",type=str,default=r"/mnt/g/LMDB/COCO_val",help="path to lmdb database for validation set. If not created then execute lmdb_dataset.py to create one")
    parser.add_argument("--root_dir",type=str,default="COCO/coco2017",help="root directory of COCO dataset")
    parser.add_argument("--dataset_categories",type=list,default=["truck","person","train",
                                                                  "boat","airplane","car","motorcycle",
                                                                  "bicycle","bus","traffic light",
                                                                  "fire hydrant","stop sign",
                                                                  "parking meter","bench"],help="categories to be used from COCO dataset. set it to None if you want to use all 80 categories.")
    parser.add_argument("--num_imgs_per_cat",type=int,default=1000,help="number of images per category to be used from COCO dataset")
    parser.add_argument("--system_shutdown",type=bool,default=False,help="shutdown system after training is done")
    parser.add_argument("--model_architecture",type=str,default="resnet50",help="model architecture to use for training. mobilenet_v3 for mobilenetv3 large, resnet50 for resnet50")
    parser.add_argument("--pretrained",type=bool,default=True,help="whether to use pretrained weights or not")
    parser.add_argument("--pretrained_backbone_only",type=bool,default=False,help="whether to use pretrained backbone or not")
    parser.add_argument("--optim_type",type=str,default="SGD",help="optimizer to use for training.",choices=["SGD","Adam"])
    parser.add_argument("--scheduler_type",type=str,default="StepLR",help="lr scheduler to use for training.",choices=["stepLR","cosine","onecycle","plateau"])
    args = parser.parse_args()


    SEED = args.seed
    np.random.seed(SEED)
    #seed_everything(SEED)

    if sys.platform == 'linux':
        torch.multiprocessing.set_start_method('forkserver',force=True)
    elif sys.platform == ('win32' or 'cygwin' or 'darwin'):
        torch.multiprocessing.set_start_method('spawn',force=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True #turn this to false when setting deterministic to true
    #torch.backends.cudnn.deterministic = True #for reproducibility
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.allow_tf32 = True


    # data preperations
    root = args.root_dir  #"/kaggle/input/coco-2017-dataset/coco2017"

    imgdir_train = os.path.join(root, 'train2017') 

    annFile_train = os.path.join(root, 'annotations/instances_train2017.json') 

    imgdir_val = os.path.join(root, 'val2017') 

    annFile_val = os.path.join(root, 'annotations/instances_val2017.json')

    if sys.platform == 'linux' or 'darwin':
        lmdb_train = args.lmdb_train_path
        lmdb_val = args.lmdb_val_path 
    elif sys.platform == ('win32' or 'cygwin'):
        lmdb_train = os.path.abspath(args.lmdb_train_path) 
        lmdb_val = os.path.abspath(args.lmdb_val_path)

    dataset_categories = args.dataset_categories
    if args.dataset == "lmdb_COCO":
        dataset_train = LMDBDataset(lmdb_train,annFile_train,categories=dataset_categories,num_imgs_per_cat=args.num_imgs_per_cat)
        dataset_valid = LMDBDataset(lmdb_val,annFile_val,categories=dataset_categories,num_imgs_per_cat=args.num_images_per_cat/5)
    if args.dataset == "COCO":
        dataset_train = COCODataset(imgdir_train,annFile_train,categories=dataset_categories,num_imgs_per_cat=args.num_imgs_per_cat) #if you want to use original coco dataset without memory mapping
        dataset_valid = COCODataset(imgdir_val,annFile_val,categories=dataset_categories,num_imgs_per_cat=args.num_images_per_cat/5)
    num_classes = len(dataset_train.categories)+1 if dataset_train.categories is not None else len(dataset_train.cat_list)+1 #+1 for background class
        
    if args.show_sample:
        dataset_train.show_annotated_image(int(np.random.randint(0,len(dataset_train),1)))
        dataset_valid.show_annotated_image(int(np.random.randint(0,len(dataset_valid),1)))

    # data loading
    batch_size = args.batch_size

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            collate_fn=c_fn,
                                            pin_memory=True,
                                            num_workers=args.num_workers,
                                            persistent_workers=True,
                                            prefetch_factor=args.prefetch_factor)  

    val_loader = torch.utils.data.DataLoader(dataset_valid,
                                            batch_size=batch_size,
                                            collate_fn=c_fn,
                                            pin_memory=True,
                                            num_workers=args.num_workers,
                                            persistent_workers=True,
                                            prefetch_factor=args.prefetch_factor)

    #model
    model = FasterRCNNLightningModule(num_classes=num_classes,
                                      training_set=dataset_train,
                                      batch_size=batch_size,
                                      lr=args.lr,
                                      model_architecture=args.model_architecture,
                                      pretrained=args.pretrained,
                                      pretrained_backbone_only=args.pretrained_backbone_only,
                                      optim_type=args.optim_type,
                                      scheduler_type=args.scheduler_type)


    early_stopping = EarlyStopping(monitor=args.monitor,
                                   patience=args.patience,
                                   mode=args.mode)
    
    checkpointer = ModelCheckpoint(monitor=args.monitor,
                                   dirpath=args.dirpath,
                                   filename='FasterRCNN-{epoch:02d}-{Mean Average Precision:.4f}',
                                   save_top_k=3,
                                   mode=args.mode,
                                   save_weights_only=args.save_weights_only)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    stochastic_weight_avg = StochasticWeightAveraging(swa_lrs=model.hparams.lr,
                                                      swa_epoch_start=args.swa_epoch_start)

    #trainer
    trainer = Trainer(callbacks=[early_stopping,checkpointer,lr_monitor,stochastic_weight_avg], 
                      accelerator=args.accelerator, 
                      min_epochs=args.min_epochs,
                      max_epochs=args.max_epochs,
                      precision=args.precision,
                      accumulate_grad_batches=args.accumulate_grad_batches,
                      gradient_clip_algorithm=args.gradient_clip_algorithm,
                      gradient_clip_val=args.gradient_clip_val) 

    #lr tuning
    if args.lr_finder:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model,train_dataloaders=train_loader,val_dataloaders=val_loader,max_lr=1e-2,min_lr=1e-6,num_training=2500,mode="linear")
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        print(f"new lr = {new_lr}")
        model.hparams.lr = new_lr
        del tuner,lr_finder
    
    #training
    trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader)
    torch.save(model.model.state_dict(), f"FasterRCNN_lightning_{args.model_architecture}.pth")

    #turn off system if training on local system.
    if args.system_shutdown:
        os.system("shutdown.exe /s /t 10")
    



