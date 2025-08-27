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


DATA_DIR = os.environ['DATA_DIR'] 

sample_files = glob.glob(os.path.join(DATA_DIR, "constant_frequencies/*.npy")) 
# sample_files = [sf for sf in sample_files if "_2.npy" not in sf and "_512.npy" not in sf]

sample_files = sorted(
    sample_files,
    key=lambda x: int(re.search(r'samples_block_(\d+)\.npy', x).group(1))
)

key = jax.random.key(0) 

for i, sf in enumerate(sample_files): 
    print(f"working on {sf}")
    block_num = re.search(r'samples_block_(\d+)\.npy', sf).group(1)
    samples = jnp.load(sf)[:, :]
    key, subkey = jax.random.split(key) 
    indices = jax.random.permutation(key, samples.shape[1])[:20000]
    subsamples = samples[:, indices]
    print(subsamples.shape)
    path = os.path.join("/n/holylabs/murthy_users/Lab/juancarlos/olfactory-ec-cached-data/constant_frequencies", f"samples_block_{block_num}.npy")
    jnp.save(path, subsamples) 
