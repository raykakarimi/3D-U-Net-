### deploy
import os
from os.path import splitext
from os import listdir
from glob import glob
import pathlib
import numpy as np
from scipy.ndimage import zoom
from model.unet3d import UNet3D
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
from barbar import Bar
from data_handler import Data_Handler
import torchvision
from utils.transforms import PairResize, ToTensor

#saving  images
def save_image(image,header, target_test_mask_dir):
    
    #transforming images into initial size of images(before downsampling)
    shape = header['ITK_FileNotes'][0][-14:-1]
    file_name = header['ITK_FileNotes'][0][0:-27]
    image_real_size = [int(shape[0:3]),int(shape[5:8]),int(shape[10:13])]
    plt.imshow(image[1,1,100,:,:].squeeze())
    plt.show()
    resized_image = zoom(image, (1,1, image_real_size[0]/image.shape[2],
            image_real_size[1]/image.shape[3], image_real_size[2]/image.shape[4]))
    
# ######## masking with 7 values
#     myList = [0, 420, 550, 205, 850, 500, 820, 600]
#     resized_image = resized_image.squeeze()
#     data1 = resized_image.reshape((image_real_size[0]*image_real_size[1]*image_real_size[2]))
#     mask_func = lambda myList,myNumber: min(myList, key=lambda x:abs(x-myNumber))
#     data2 = np.array([mask_func(myList,i) for i in data1 ])
#     resized_image = data2.reshape(image_real_size)

#saving with dicom format in the output directory
    img = sitk.GetImageFromArray(resized_image)
    for key in header:
        img.SetMetaData(key,header[key][0])
    sitk.WriteImage(img, os.path.join(target_test_mask_dir, file_name +'mask.nii.gz'))
    print("{}' mask saved successfully!".format( file_name))

##################deploy
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

target_dataset_path = '../data/'
dataset_file_name = 'CT_dataset.np'
experiment_path = '../data/models/experiment{}/'.format(experiment_number)
target_test_mask_dir = target_dataset_path + 'deploy'

img_resized = [32, 96, 96]

transforms1 = torchvision.transforms.Compose([PairResize(img_resized, mode = 'test'),
    
    ToTensor(mode = 'test')
])
dataset_test = Data_Handler(target_dataset_path + dataset_file_name, 'test', transforms1)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = UNet3D(1,1).to(device)

#loading model
training_time = state_filename = experiment_path + "state_last.pt"
training_losses = []
validation_losses = []
min_val_loss = float('inf')
test_losses=[]
start_epoch=0
if os.path.isfile(state_filename):
    checkpoint = torch.load(state_filename,map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    min_val_loss = checkpoint['min_val_loss']
    print("=> loaded checkpoint '{}' (epoch {})"
              .format(state_filename, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(state_filename))
        
########################
### masking test set ###
########################

model.eval()
with torch.no_grad():
    for batch in Bar(dataloader_test):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        yhat = model(imgs)
        save_image(yhat,batch['header'],target_test_mask_dir)

    
