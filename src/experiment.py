import os
import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
from barbar import Bar


class experiment:
   def __init__(self, model, lr_rate, amsgrad, weight_decay,lr_milestones,lr_gamma,
                 n_epochs,dataloader_val,dataloader_train, device,experiment_path,epsilon=1e-6,eps=1e-07):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr_rate, eps=eps, amsgrad=amsgrad, weight_decay=weight_decay)
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.n_epochs = n_epochs
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val   
        self.device = device
        self.experiment_path = experiment_path
                
   def train_step(self,x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self.dice_loss(yhat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()
    
   def  train_model(self):
        self.load_model()
        self.n_epochs -= self.start_epoch
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.lr_milestones, gamma=self.lr_gamma, last_epoch=-1 if self.start_epoch == 0 else self.start_epoch)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)
        for epoch in range(self.n_epochs):  
            batch_losses = []
    
            for batch in Bar(self.dataloader_train):
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=self.device, dtype=torch.float32)
                true_masks = true_masks.to(device=self.device, dtype=torch.float32)
                loss = self.train_step(imgs, true_masks)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.training_losses.append(training_loss)
    
            with torch.no_grad():
                val_losses = []
                for batch in Bar(self.dataloader_val):
                    imgs = batch['image'].to(device = self.device, dtype=torch.float32)
                    true_masks = batch['mask'].to(device = self.device, dtype=torch.float32)
    
                    self.model.eval()
                    yhat = self.model(imgs)
                    val_loss = self.dice_loss(yhat,true_masks).item()
        #             val_loss = dice_loss(yhat,true_masks, yhat, log=False).item()
                    val_losses.append(val_loss)
                validation_loss = np.mean(val_losses)
                self.validation_losses.append(validation_loss)
    
            if (validation_loss < self.min_val_loss):
                self.min_val_loss = validation_loss
    
                self.best_state = {'epoch': self.start_epoch + epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                              'validation_losses': self.validation_losses, 'training_losses': self.training_losses, 'min_val_loss': self.min_val_loss}
    
                torch.save(self.best_state, self.experiment_path + "state_best.pt")
    
                print(f"[{self.start_epoch + epoch + 1}] Learning Rate: {self.optimizer.param_groups[0]['lr']} Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}\t Saved!")
            else:
                print(f"[{self.start_epoch + epoch + 1}] Learning Rate: {self.optimizer.param_groups[0]['lr']} Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}")
    
            if (epoch%5 == 0):
                self.last_state = {'epoch': self.start_epoch + epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                          'validation_losses': self.validation_losses, 'training_losses': self.training_losses, 'min_val_loss': self.min_val_loss}
                torch.save(self.last_state, self.experiment_path + "state_last.pt")
    
            self.scheduler.step()
    
        if(self.n_epochs > 0):
            self.last_state = {'epoch': self.start_epoch + epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                          'validation_losses': self.validation_losses, 'training_losses': self.training_losses, 'min_val_loss': self.min_val_loss}
            torch.save(self.last_state, self.experiment_path + "state_last.pt")
    
            print('Training Time: {}s'.format(np.round(self.training_time, 2)))
            if os.path.isfile(self.experiment_path + "report.txt"):
                text_file = open(self.experiment_path + "report.txt", "a")
                report_text = "training time: {}s\n".format(np.round(self.training_time, 2))
                text_file.write(report_text)
                text_file.close()
        
        
   # ------------------------------------
   # Load checkpoint and define optimizer
   # ------------------------------------
   def  load_model(self):
        self.training_losses = []
        self.validation_losses = []
        self.min_val_loss = float('inf')

        self.start_epoch = 0
        self.training_time = 0
        state_filename = self.experiment_path + "state_last.pt"
        if os.path.isfile(state_filename):
            checkpoint = torch.load(state_filename,map_location=torch.device('cpu'))
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_losses = checkpoint['training_losses']
            self.validation_losses = checkpoint['validation_losses']
            self.min_val_loss = checkpoint['min_val_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                      .format(state_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(state_filename))

    
    
   # --------------------
   # dice loss calculator
   # --------------------  
   def  dice_loss(self, input, target, epsilon=1e-6, weight=None):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
        Args:
             input (torch.Tensor): NxCxSpatial input tensor
             target (torch.Tensor): NxCxSpatial target tensor
             epsilon (float): prevents division by zero
             weight (torch.Tensor): Cx1 tensor of weight per channel/class
        """

        # input and target shapes must match
        input = input*850
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = self.flatten(input)
        target = self.flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = (input * target).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
        per_channel_dice = 2 * (intersect / denominator.clamp(min=epsilon))
        return  1. - torch.mean(per_channel_dice)
    
   @staticmethod
   def flatten(tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        # number of channels
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)