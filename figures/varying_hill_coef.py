import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob 
import sys 
import os 
import re 
import json
from plot_utils import sort_rows_by_first_threshold


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

RESULTS_DIR = os.environ['RESULTS_DIR'] 
DEFAULT_DIRECTORY = os.path.join(RESULTS_DIR, "varying_hill_coef/opt_W") 
DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY
E_DIRECTORY = os.path.join(RESULTS_DIR, "varying_hill_coef/opt_E_given_shuffle_W") 

W_init_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "W_init_[0-9]*.npy")),
    key=lambda x: int(re.search(r'W_init_(\d+)\.npy', x).group(1))
)
W_final_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "W_final_[0-9]*.npy")),
    key=lambda x: int(re.search(r'W_final_(\d+)\.npy', x).group(1))
)
E_final_files = sorted(
    glob.glob(os.path.join(E_DIRECTORY, "E_final_*.npy")),
    key=lambda x: int(re.search(r'E_final_(\d+)\.npy', x).group(1))
)
config_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "config_*.json")),
    key=lambda x: int(re.search(r'config_(\d+)\.json', x).group(1))
)

def extract_sigma_and_blocks(config): 
    sigma = config["hyperparams"]["sigma_c"]
    blocks = os.path.basename(config["data_path"]).strip("samples_block_").strip(".npy")
    return sigma, blocks 

def plot_hist(W_init, W_final, ax, threshold=None): 
    W_init = jnp.log10(W_init) 
    if threshold: 
        W_final = W_final[W_final > threshold] 
    W_final = jnp.log10(W_final) 
    ax.hist(W_init, alpha=0.6, density=True) 
    ax.hist(W_final, alpha=0.6, density=True) 
    ax.set_yscale("log")
    return ax 

def fit_normal(W, threshold=None):
    if threshold: 
        W = W[W > threshold] 
    log_W = jnp.log10(W)
    log_vals = jnp.ravel(log_W)
    mean = jnp.mean(log_vals)
    std = jnp.std(log_vals)
    def normal_pdf(log_x):
        return (1 / (std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((log_x - mean) ** 2) / (2 * std ** 2))
    return normal_pdf

def extract_sigma_and_blocks(config): 
    sigma = config["hyperparams"]["sigma_c"]
    blocks = os.path.basename(config["data_path"]).strip("samples_block_").strip(".npy")
    return sigma, blocks 

def make_title(config): 
    sigma_c, blocks = extract_sigma_and_blocks(config) 
    title = fr"$\sigma_c ={sigma_c}$, blocks={blocks}"
    return title 

def get_frequencies(config_path):
    with open(config_path, "r") as c: 
        config = json.load(c) 
    odorant_path = config["data_path"]
    odorants = jnp.load(odorant_path) 
    frequencies = jnp.mean(odorants, axis=1) 
    return frequencies

def W_heatmap(ax, W, config_path, min=None, max=None): 
    log_W = jnp.log10(W) 
    min = jnp.log10(min)
    max = jnp.log10(max) 
    frequencies = get_frequencies(config_path) 
    order = jnp.argsort(frequencies)[::-1] 
    if min is not None: 
        im = ax.imshow(log_W[:, order], aspect="auto", interpolation="none", cmap="Blues", vmin=min, vmax=max)
    else: 
        im = ax.imshow(log_W[:, order], aspect="auto", interpolation="none", cmap="Blues")
    return im, log_W[:, order], order

def sort_neurons_by_thresholded_expression(expression_matrix):
    threshold = 0.1 * jnp.max(expression_matrix, axis=1)
    neuron_order = sort_rows_by_first_threshold(expression_matrix, threshold)
    return expression_matrix[neuron_order], neuron_order

    
fig, axs = plt.subplots(3, 4, figsize=(7, 4.5), layout="constrained", gridspec_kw={"wspace": 0.0, "hspace": 0.0, "height_ratios": [2, 1.3, 2]})

for i in range(4): 
    with open(config_files[i], "r") as c: 
        config = json.load(c)
    W_init = jnp.load(W_init_files[i]) 
    W_final = jnp.load(W_final_files[i]) 
    E_final = jnp.load(E_final_files[i]) 
    threshold = jnp.min(W_init) 
    W_heatmap(axs[0, i], W_final, config_files[i], min=threshold, max=1)
    axs[0, i].set_title(fr"$n={i+1}$"+"\noptimized W")
    axs[0, i].set_ylabel("receptors", labelpad=-12) 
    axs[1, i].set_ylabel("density") 
    axs[0, i].set_xticks([0, 999])
    axs[0, i].set_xticklabels(["1", "1000"])
    axs[0, i].set_yticks([0, 59])
    axs[0, i].set_yticklabels(["1", "60"])
    axs[0, i].set_xlabel("odorants", labelpad=-12)
    axs[1, i] = plot_hist(W_init.flatten(), W_final.flatten(), axs[1, i], threshold) 
    axs[1, i].set_title(r"$P(W_{ij})$")
    axs[1, i].set_xlabel(r"$\log_{10}(W_{ij})$")
    sorted_E, _ = sort_neurons_by_thresholded_expression(E_final)
    axs[2, i].imshow(sorted_E, aspect="auto", cmap="Blues", interpolation="none")
    axs[2, i].set_title("optimized E")
    axs[2, i].set_xticks([0, 59])
    axs[2, i].set_xticklabels(["1", "60"])
    axs[2, i].set_xlabel("receptors", labelpad=-12) 
    axs[2, i].set_ylabel("neurons", labelpad=-18) 
    axs[2, i].set_yticks([0, 1259])
    axs[2, i].set_yticklabels(["1", "1260"])


fig.savefig("varying_hill_coefs.png", dpi=600)
fig.savefig("varying_hill_coefs.pdf", dpi=600)
fig.savefig("varying_hill_coefs.svg")

