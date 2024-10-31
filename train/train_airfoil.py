import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

train_data_dir = '/home/wangpengkai/node/self_diffusion_learn/data/airfoil_pressure.npy'
# shape: (1000, 200, 200)
air_pressure_data = np.load(train_data_dir)

print(f'original size: {air_pressure_data.shape}')
# clip matrix
res = 200
resized_data = air_pressure_data[:, :res, :res,].reshape(-1, 1, res, res)
print(f'clip size: {resized_data}')
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

plt.imshow(standardized_data[0][0], cmap='viridis')
plt.title("Resized Image")
plt.axis('off')
plt.savefig('air_foil.jpg', dpi=500)


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
    train_batch_size = 20,
    results_folder = '/home/wangpengkai/node/self_diffusion_learn/models/airfoil',
    train_lr = 8e-5,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
)

trainer.train()