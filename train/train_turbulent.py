import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
import pickle
from scipy.io import loadmat 

from accelerate import Accelerator, DataLoaderConfiguration
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


train_secxy = '/home/wangpengkai/node/self_diffusion_learn/data/ch_2Dxysec.pickle'
snapshots = 10000
with open(train_secxy, 'rb') as f:
    cy_box = pickle.load(f).copy()[:snapshots,...]

res = 48
print(f'original size: {cy_box.shape}')
# clip matrix
resized_data = cy_box[:, :res, :res, :].reshape(-1, 1, res, res)
print(f'clip size: {cy_box.shape}')
# Visualize a sample
tensor_data = torch.as_tensor(resized_data, dtype=torch.float16)  # Convert to PyTorch tensor

print(f"Tensor shape: {tensor_data.shape}")

# Normalize to [0, 1]
min_val = tensor_data.min()
max_val = tensor_data.max()
standardized_data = (tensor_data - min_val) / (max_val - min_val)

print(f"Standardized tensor shape: {standardized_data.shape}")

# Confirm normalization by checking the mean and std
print(f"Standardized mean: {standardized_data.mean().item():.4f}")
print(f"Standardized std: {standardized_data.std().item():.4f}")

plt.imshow(standardized_data[0][0], cmap='jet')
plt.title("Resized Image")
plt.axis('off')
plt.savefig('turbulent.jpg')


from accelerate import Accelerator, DataLoaderConfiguration

# Rest of your setup

from  denoising_diffusion_pytorch import GaussianDiffusion, Unet
from denoising_diffusion_pytorch_modified import Trainer
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = res,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 250
)

trainer = Trainer(
    diffusion,
    standardized_data,
    train_batch_size = 350,
    results_folder = '/home/wangpengkai/node/self_diffusion_learn/models/turbulent',
    train_lr = 8e-5,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
)

trainer.train()