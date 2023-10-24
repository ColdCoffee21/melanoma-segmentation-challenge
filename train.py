import argparse
import time
import torch
import torchvision
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.tuner import Tuner

from datasets import *
from models import *

import segmentation_models_pytorch as smp

unetplusplus = smp.UnetPlusPlus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  
    classes=1,          
    # activation='sigmoid',   
    decoder_use_batchnorm=True,            
)

manet = smp.MAnet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  
    classes=1,  
    # activation='sigmoid',   
    decoder_use_batchnorm=True,                 
)

models = {'unet': UNet, 'deeper_unet': DeeperUNet, 'code_unet': CodeUNet, 'residual_unet': ResidUNet, 'attention_unet': AttentionUNet, 'unetplusplus': unetplusplus, 'manet': manet}
optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'adamW': torch.optim.AdamW}
metrics = {'acc': torchmetrics.Accuracy(task='binary').to('cuda'),
            'dice': torchmetrics.Dice().to('cuda'),
            'f1': torchmetrics.F1Score(task='binary').to('cuda'),
            'precision': torchmetrics.Precision(task='binary').to('cuda'),
            'recall': torchmetrics.Recall(task='binary').to('cuda')}

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, no_cosine=False):
        self.warmup = warmup
        self.no_cosine = no_cosine
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if self.no_cosine:
            lr_factor = 1.0
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_num_iters))
        if step <= self.warmup:
            lr_factor *= step * 1.0 / self.warmup
        return lr_factor

class Segmenter(pl.LightningModule):
  def __init__(self, **kwargs):
    super().__init__()  
    self.save_hyperparameters()

    # defining model
    self.model_name = self.hparams['model_name']
    assert self.model_name in models, f'Model name "{self.model_name}" is not available. List of available names: {list(models.keys())}'
    if self.model_name == 'unetplusplus' or self.model_name == 'manet':
      self.model      = models[self.model_name].to('cuda')
      self.model.train()
    elif self.model_name == 'code_unet':
      self.model      = models[self.model_name](self.hparams['code_size']).to('cuda')
    else:
      self.model      = models[self.model_name]().to('cuda')

    # assigning optimizer values
    self.optimizer_name = self.hparams['optimizer_name']
    self.lr             = self.hparams['learning_rate']
    self.learning_rate = self.hparams['learning_rate']


    self.loss_pos_weight = self.hparams['loss_pos_weight']
    # self.Dice = smp.losses.DiceLoss(mode='binary', from_logits=False)
    self.Dice = DiceLoss()
  

  def step(self, batch, nn_set):
    X, y = batch['image'], batch['mask']
    X, y   = X.float().to('cuda'), y.to('cuda').float()
    y_hat  = self.model(X)
    y_prob = torch.sigmoid(y_hat)

    pos_weight = torch.tensor([self.loss_pos_weight]).float().to('cuda')
    # loss = F.binary_cross_entropy_with_logits(y, y_prob, pos_weight=pos_weight)
    if self.hparams['loss_type'] == 'bce':
      loss = F.binary_cross_entropy_with_logits(y_hat, y.float(), pos_weight=pos_weight)
    elif self.hparams['loss_type'] == 'dice':
      loss = self.Dice(y_hat, y.float())
    self.log(f"{nn_set}_loss", loss, on_step=False, on_epoch=True)
    del X, y_hat, batch

    for i, (metric_name, metric_fn) in enumerate(metrics.items()):
      score = metric_fn(y_prob, y.int())
      self.log(f'{nn_set}_{metric_name}', score, on_step=False, on_epoch=True)

    return loss

  def training_step(self, batch, batch_idx):
    # self.step(batch, 'train')
    return {"loss": self.step(batch, "train")}

  def validation_step(self, batch, batch_idx):
    # self.step(batch, 'val')
    return {"val_loss": self.step(batch, "val")}

  def test_step(self, batch, batch_idx):
    # self.step(batch, 'test')
    return {"test_loss": self.step(batch, "test")}

  def forward(self, X):
    return self.model(X)

  # def configure_optimizers(self):
  #   assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
  #   return optimizers[self.optimizer_name](self.parameters(), lr = self.lr)
  
  def configure_optimizers(self):
        assert self.optimizer_name in optimizers, f'Optimizer name "{self.optimizer_name}" is not available. List of available names: {list(models.keys())}'
        optimizer = optimizers[self.optimizer_name](self.parameters(), lr = self.lr)
        # return optimizer
        if self.hparams['warmup'] is not None:
            lr_scheduler = CosineWarmupScheduler(
                optimizer,
                warmup=self.hparams['warmup']
                if self.hparams['warmup_use_steps']
                else self.hparams['warmup'] * self.hparams.samples_per_epoch,
                max_iters=self.hparams['max_epochs'] * self.hparams.samples_per_epoch,
                no_cosine=self.hparams['no_cosine'],
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        else:
            return optimizer

def main(args):
    data_dir = 'data/data/segmentation'
    torch.set_float32_matmul_precision('medium')
    config_segm = {
        'train_data_dir': os.path.join(data_dir, 'train'),
        'val_data_dir': os.path.join(data_dir, 'val'),
        'test_data_dir': os.path.join(data_dir, 'test'),
        'batch_size': args.batch_size,
        'learning_rate': args.optimizer_lr,
        'loss_pos_weight': args.loss_pos_weight,
        'max_epochs': args.max_epochs,
        'model_name': args.model_name,
        'optimizer_name': args.optimizer_name,
        'bin': args.bin,
        'experiment_name': args.experiment_name,
        'warmup': args.warmup,
        'warmup_use_steps': args.warmup_use_steps,
        'no_cosine': args.no_cosine,
        'samples_per_epoch': 3328, # len of train dataset
        'loss_type': args.loss_type,
        'code_size': args.code_size,
    }
    print("The model configuration is: \n", config_segm)
    data = Scan_DataModule_Segm(config_segm)
    segmenter = Segmenter(**config_segm)
    logger = TensorBoardLogger(config_segm['bin'], name=config_segm['experiment_name'])
    checkpoint_callback = ModelCheckpoint(monitor='val_dice', mode='max')
    trainer = Trainer(devices=1, accelerator='gpu', max_epochs=config_segm['max_epochs'],
                      logger=logger, callbacks=[checkpoint_callback, LearningRateMonitor("step"), TQDMProgressBar(refresh_rate=100)],
                      default_root_dir=config_segm['bin'], deterministic=True,
                      log_every_n_steps=1)
    
    lr_finder = Tuner(trainer)
    lr_finder_result = lr_finder.lr_find(segmenter, data)
    # Results can be found in
    # print(f"suggested lr: {lr_finder_result.results}, used lr: {config_segm['learning_rate']}") 
    lr_finder.scale_batch_size(segmenter, data, mode="power")

    start = time.time()
    trainer.fit(segmenter, data)
    print(f"training time: {(time.time() - start) / 3600}h")
    print(f"best model: {trainer.checkpoint_callback.best_model_path}")
    # print the lr of the model
    print(f"lr: {segmenter.lr}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer_lr', type=float, default=0.1)
    parser.add_argument('--loss_pos_weight', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--model_name', type=str, default='unet')
    parser.add_argument('--optimizer_name', type=str, default='adam')
    parser.add_argument('--experiment_name', type=str, default='20_epoch_unet_adam')
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--warmup_use_steps", action="store_true")
    parser.add_argument("--no_cosine", action="store_true")
    parser.add_argument('--loss_type', type=str, default='bce')
    parser.add_argument('--code_size', type=int, default=2)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--bin', type=str, default='segm_models/')
    args = parser.parse_args()

    main(args)