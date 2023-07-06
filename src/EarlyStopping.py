import numpy as np

class EarlyStopper():
    
    def __init__(self,patience,threshold,improvement_req,threshold_map,improvement_req_map,model):
        self.patience = patience
        self.threshold = threshold #threshold that the validation loss must be more than to be considered an improvement. Smaller the value the more strict the early stopping.
        self.min_val_loss = np.inf
        self.max_val_map = np.NINF
        self._counter = 0
        self.best_state_dict = None
        self.model = model
        self.epoch = None
        self.improvement_thres = improvement_req #threshold that the validation loss must improve (decrease) by to be considered an improvement between epochs. Bigger the value, the more strict the early stopping.
        self._val_loss_last_epoch = None
        self.early_stop = False
        self.map_last_epoch = None
        self.improvement_thres_map = improvement_req_map #threshold that the validation map must improve (increase) by to be considered an improvement between epochs. Bigger the value, the more strict the early stopping.
        self.threshold_map = threshold_map #threshold that the validation map must be more than to be considered an improvement. Larger the value the more strict the early stopping.

    def __call__(self,meanap,val_loss,epoch):
        if  meanap > self.max_val_map: #val_loss < self.min_val_loss or
            self.min_val_loss = val_loss
            self.max_val_map = meanap
            self._counter = 0
            self.best_state_dict = self.model.state_dict()
            self.epoch = epoch
            self._val_loss_last_epoch = val_loss
            self._map_last_epoch = meanap
        elif val_loss > (self.min_val_loss+self.threshold) and meanap < (self.max_val_map - self.threshold_map):
            self._counter += 1
            self._val_loss_last_epoch = val_loss
            self._map_last_epoch = meanap
            if self._counter >= self.patience:
                print("Early Stopping -- validation loss increasing or Mean Average Precision decreasing")
                self.early_stop = True
        elif (self._val_loss_last_epoch - val_loss) < self.improvement_thres and meanap < (self._map_last_epoch-self.improvement_thres_map):
            self._counter += 1
            self._val_loss_last_epoch = val_loss
            self._map_last_epoch = meanap
            if self._counter >= self.patience:
                print("Early Stopping -- validation loss or Mean Average Precision not improving")
                self.early_stop = True 
        