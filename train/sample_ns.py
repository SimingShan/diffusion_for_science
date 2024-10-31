import os
from  denoising_diffusion_pytorch import GaussianDiffusion, Unet
from denoising_diffusion_pytorch_modified import Trainer
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate import Accelerator, DataLoaderConfiguration
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import h5py
from scipy.io import loadmat
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
    sampling_timesteps = 1000
).to(device='cuda:1')


diffusion.load_state_dict(torch.load('/home/wangpengkai/node/self_diffusion_learn/models/ns/model-10.pt', map_location='cuda:1')['model'])

sampled_images = diffusion.sample(batch_size = 4)
sampled_images = sampled_images[..., 1:-1, 1:-1].detach().cpu().numpy()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Assuming sampled_images and resized_data are available as numpy arrays
# sampled_images: model predictions
# resized_data: ground truth data

# Create a figure with two rows and five columns
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust height to be smaller
print(sampled_images.shape)
# Plot the predicted images (sampled_images) in the top row
for i in range(4):
    ax = axes[0, i]
    ax.set_xlabel('x')  
    ax.set_ylabel('y') 
    im = ax.imshow(sampled_images[i, 0], cmap="twilight", aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    fig.colorbar(im, ax=ax)
    


ns_mat = h5py.File('/home/wangpengkai/node/self_diffusion_learn/data/ns_V1e-3_N5000_T50.mat', 'r')
# a->shape(5000, 64, 64) u->shape(5000, 50, 64, 64)
a_mat = np.array(ns_mat['a']).transpose(2, 0, 1)
u_mat = np.array(ns_mat['u']).transpose(3, 0, 1, 2)[:, -1,...]   

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

index = np.random.choice(1024, size=4, replace=False)
for i in range(4):
    j = index[i]
    ax = axes[1, i]
    ax.set_xlabel('x')  
    ax.set_ylabel('y') 
    im = ax.imshow(standardized_data[j, 0], cmap="twilight", aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    fig.colorbar(im, ax=ax)

# Add a line to separate the top and bottom rows
line = Line2D([0.05, 0.95], [0.5, 0.5], color='black', transform=fig.transFigure, linewidth=2)
fig.add_artist(line)

# Adjust layout and spacing
plt.tight_layout(pad=0.5)
plt.subplots_adjust(hspace=0.1)  # Adjust vertical spacing

# Save the combined plot
plt.savefig("ddpm_combined_ns_1000.png", bbox_inches='tight', dpi=400)