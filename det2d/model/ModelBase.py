import pytorch_lightning as pl
import torch
import copy

class ModelBase(pl.LightningModule):
    def __init__(self, model_cfg):
        super(ModelBase, self).__init__()
        self.optim_config = model_cfg['schedule']['optimizer']
        self.scheduler_config = model_cfg['schedule']['lr_schedule']

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.optim_config)
        name = optimizer_cfg.pop('name')
        Optimizer = getattr(torch.optim, name)
        self.optimizer = Optimizer(params=self.model.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.scheduler_config)
        name = schedule_cfg.pop('name')
        Scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = Scheduler(optimizer=self.optimizer, **schedule_cfg)

        return  [self.optimizer], [
                {
                 'scheduler': self.lr_scheduler,
                 'interval': 'epoch',
                 'frequency': 1,
                }]



