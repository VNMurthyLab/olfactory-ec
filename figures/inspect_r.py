import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob 
import sys 
import os 
import re 
import json 

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

CLIP = 1e-9
RESULTS_DIR = os.environ['RESULTS_DIR'] 
DEFAULT_DIRECTORY = os.path.join(RESULTS_DIR, "environment_sweep/opt_E_given_shuffle_W/noncanonical_init") 

DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY

r_init_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "r_init_*.npy")),
    key=lambda x: int(re.search(r'r_init_(\d+)\.npy', x).group(1))
)

r_final_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "r_final_*.npy")),
    key=lambda x: int(re.search(r'r_final_(\d+)\.npy', x).group(1))
)

config_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "config_*.json")),
    key=lambda x: int(re.search(r'config_(\d+)\.json', x).group(1))
)

def extract_sigma_and_blocks(config): 
    sigma = config["hyperparams"]["sigma_c"]
    blocks = os.path.basename(config["data_path"]).strip("samples_block_").strip(".npy")
    return sigma, blocks 

def plot_hist(ax, config_path, r_final_file, r_init_file=None, transform=None):
    with open(config_path, "r") as c:
        config = json.load(c)
    r_final = jnp.load(r_final_file)
    if transform is not None:
        r_final = transform(r_final)
    if r_init_file is not None:
        r_init = jnp.load(r_init_file)
        if transform is not None:
            r_init = transform(r_init)
        ax.hist(r_init.flatten(), alpha=1.0, density=True)
    ax.hist(r_final.flatten(), alpha=1.0, density=True)
    sigma, blocks = extract_sigma_and_blocks(config)
    ax.set_title(fr"$\sigma_c = {sigma}$, blocks = {blocks}")
    return ax

def compute_PCs(r): 
    cov = jnp.cov(r)
    evalues, evectors = jnp.linalg.eigh(cov) 
    return evalues / jnp.sum(evalues), evectors, 

def make_title(config): 
    sigma, blocks = extract_sigma_and_blocks(config)
    title = fr"$\sigma_c ={sigma}$, blocks={blocks}"
    return title 

def get_frequencies(config_path):
    with open(config_path, "r") as c: 
        config = json.load(c) 
    odorant_path = config["data_path"]
    odorants = jnp.load(odorant_path) 
    frequencies = jnp.mean(odorants, axis=1) 
    return frequencies

def evec_barplot(ax, evec, jitter=0, label=""): 
    ax.bar(jnp.arange(len(evec)) + jitter, evec, label=label, zorder=3)
    return ax 

def r_heatmap(ax, r, config_path): 
    with open(config_path, "r") as c: 
        config = json.load(c) 
    im = ax.imshow(r, aspect="auto", interpolation="none", cmap="Blues")
    ax.set_title(make_title(config)) 
    return im 

key = jax.random.key(0) 

ts = [] 
ss = [] 
titles = [] 
true_top_evecs = [] 
shuffle_top_evecs = [] 
rs = [] 

# for i in range(len(r_init_files)): 
#     with open(config_files[i], "r") as c: 
#         config = json.load(c)
#     output_dir = config["logging"]["output_dir"]
#     key, subkey_sample, subkey_shuffle = jax.random.split(key, 3)
#     r_init = jnp.load(r_init_files[i])
#     r_final = jnp.load(r_final_files[i])
#     rs.append(r_final) 
#     # threshold = jnp.min(W_init) 
#     # pdf, mean, std = fit_normal(W_final.flatten(), threshold)
#     shuffle_indices = jax.random.permutation(subkey_shuffle, len(r_final.ravel())) 
#     r_shuffle = r_final.ravel()[shuffle_indices].reshape(r_final.shape)
#     true_spectrum, true_evecs = compute_PCs(r_final)
#     shuffle_spectrum, shuffle_evecs = compute_PCs(r_init)
#     ts.append(true_spectrum)
#     ss.append(shuffle_spectrum)
#     titles.append(make_title(config))
#     true_top_evecs.append(true_evecs[:, 0])
#     shuffle_top_evecs.append(shuffle_evecs[:, 0])

fig, axs = plt.subplots(7, 4,
    figsize=(7, 9), 
    layout="constrained")

true_spectra = jnp.array(ts) 
shuffle_spectra = jnp.array(ss)
for i, ax in enumerate(axs.flatten()):
    ax = plot_hist(ax, config_files[i], r_final_files[i]) 
    # ax = plot_hist(ax, config_files[i], r_final_files[i], transform=jnp.tanh)
    try: 
        ax.set_title(titles[i]) 
    except: 
        pass 

fig.savefig(f"r_histograms_given_W_shuffle.png", dpi=600) 
fig.savefig(f"r_histograms_given_W_shuffle.pdf", dpi=600) 
fig.savefig(f"r_histograms_given_W_shuffle.svg")

