import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import h5py

from accelerate import Accelerator, DataLoaderConfiguration
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ns_mat = h5py.File('/home/wangpengkai/node/self_diffusion_learn/data/ns_V1e-3_N5000_T50.mat', 'r')
# a->shape(5000, 64, 64) u->shape(5000, 50, 64, 64)
a_mat = np.array(ns_mat['a']).transpose(2, 0, 1)
u_mat = np.array(ns_mat['u']).transpose(3, 0, 1, 2)[:, -1,...]   


# Check original shape
print(f"Original shape: {u_mat.shape}")

# Visualize a sample
plt.imshow(u_mat[0], cmap='viridis')
plt.title("Resized Image")
plt.axis('off')
plt.savefig('ns.jpg')

# Reshape to (1024, 1, 72, 72) and convert to PyTorch tensor
# Assuming 1024 is the number of samples; adjust if needed based on your dataset
resized_data = u_mat.reshape(-1, 1, 64, 64)  # Reshape to (n_samples, channels, height, width)
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
    image_size = 64,
    timesteps = 1000,    # number of steps
    sampling_timesteps = 250
)

trainer = Trainer(
    diffusion,
    standardized_data,
    train_batch_size = 200,
    results_folder = '/home/wangpengkai/node/self_diffusion_learn/models/ns',
    train_lr = 8e-5,
    train_num_steps = 10000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
)

trainer.train()