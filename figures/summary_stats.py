import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import argparse 
import json 
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib.ticker as mtick
import glob 
import re 


plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # Best available
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})


fig, axs = plt.subplots(7, 2, figsize=(5.5, 8), gridspec_kw={'hspace':0.0}, layout="constrained")

DATA_DIR = os.environ['DATA_DIR'] 

sample_files = glob.glob(os.path.join(DATA_DIR, "data_for_optimization/*.npy")) 
sample_files = [sf for sf in sample_files if "_2.npy" not in sf and "_512.npy" not in sf]

sample_files = sorted(
    sample_files,
    key=lambda x: int(re.search(r'samples_block_(\d+)\.npy', x).group(1))
)

for i, sf in enumerate(sample_files): 
    print(f"working on {sf}")
    block_num = re.search(r'samples_block_(\d+)\.npy', sf).group(1)
    samples = jnp.load(sf)[:, :10000]
    frequencies = jnp.mean(samples, axis=1) 
    axs[i, 0].plot(range(len(frequencies)), jnp.sort(frequencies))
    axs[i, 0].set_yscale("log")
    axs[i, 0].set_yticks([1e-1, 1e-2, 1e-3])
    axs[i, 0].minorticks_off() 
    axs[i, 0].grid() 
    cov = jnp.corrcoef(samples) 
    im = axs[i, 1].imshow(cov, interpolation="none", cmap="Greens")
    cbar = fig.colorbar(im, ax=axs[i, 1], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Sigma_{ij}$", rotation=0, labelpad=0)  
    axs[i, 0].set_title(f"blocks = {block_num}", loc='center', y=1.1,x=1.2)
    axs[i, 0].set_ylabel("frequency")
    if i == 6: 
        axs[i, 0].set_xlabel("odorants")

fig.savefig("summary_stats.png", dpi=600)
fig.savefig("summary_stats.pdf", dpi=600)
fig.savefig("summary_stats.svg")

