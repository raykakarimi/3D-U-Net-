%load_ext autoreload
%autoreload 2

import os
import numpy as np
from pathlib import Path
from data_prep import data_prep
from make_dataset import make_dataset
import matplotlib.pyplot as plt
from utils.transforms import PairResize, ToTensor
import torchvision
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
from torchsummary import summary
from barbar import Bar
from data_handler import Data_Handler
from experiment import experiment
from model.unet3d import UNet3D


source_dataset_path = '../'
source_dataset_train_dir = 'ct_train'
source_dataset_test_dir = 'ct_test'
source_image_suffix = 'image'
source_label_suffix = 'label'

target_dataset_path = '../data/'
target_dataset_train_dir = 'train'
target_dataset_test_dir = 'test'
target_header_prefix = 'hd_p'
target_image_prefix = 'img_p'
target_mask_prefix = 'msk_p'

validation_set_ratio = 0.2
dataset_file_name = 'CT_dataset.np'

experiment_number = 1
experiment_path = '../data/models/experiment{}/'.format(experiment_number)
# experiment_path = '../data/models/experiment{}/'.format(experiment_number)

img_resized = [32, 96, 96]

n_epochs = 1000
batch_size = 1
lr_rate = 0.0001
lr_gamma = 1
lr_milestones = [1,1]
amsgrad = False
weight_decay = 0.01

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
# torch.device("cpu")
Path(experiment_path).mkdir(parents=True, exist_ok=True)

################
#data prepration
################

# Preparing train data
data_prep(source_dataset_path + source_dataset_train_dir, target_dataset_path + target_dataset_train_dir,
          source_image_suffix, source_label_suffix, target_image_prefix, target_mask_prefix, target_header_prefix)

# Preparing test data
data_prep(source_dataset_path + source_dataset_test_dir, target_dataset_path + target_dataset_test_dir,
          source_image_suffix, source_label_suffix, target_image_prefix, target_mask_prefix, target_header_prefix)

# Save data absolute paths in CT_dataset.np
make_dataset(target_dataset_path, target_dataset_train_dir, target_dataset_test_dir, target_header_prefix,
             target_image_prefix, target_mask_prefix, validation_set_ratio, dataset_file_name)

#data handler and data loader call

transforms = torchvision.transforms.Compose([PairResize(img_resized),
    
    ToTensor()
])

transforms1 = torchvision.transforms.Compose([PairResize(img_resized, mode = 'test'),
    
    ToTensor(mode = 'test')
])


dataset_train = Data_Handler(target_dataset_path + dataset_file_name, 'train', transforms)
dataset_val = Data_Handler(target_dataset_path + dataset_file_name, 'val', transforms)
dataset_test = Data_Handler(target_dataset_path + dataset_file_name, 'test', transforms1)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# instantiating model and experiment class
model = UNet3D(1,1).to(device)
experiment1 = experiment(model,lr_rate, amsgrad, weight_decay,lr_milestones,lr_gamma,
                 n_epochs,dataloader_val,dataloader_train, device,experiment_path) 
experiment1.train_model()

#sketching figure of validation loss and training loss across  #epochs
val_loss=list(np.subtract([1] * len(experiment1.validation_losses),(.experiment1validation_losses)))
train_loss=list(np.subtract([1] * len(experiment1.training_losses),(experiment1.training_losses)))
fig = plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
fig.savefig(experiment_path + 'train-history.png')