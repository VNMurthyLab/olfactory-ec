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

RESULTS_DIR = os.environ['RESULTS_DIR']
DEFAULT_DIRECTORY = os.path.join(RESULTS_DIR, "flat_frequencies/environment_sweep/opt_W")
DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY
W_init_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "W_init_[0-9]*.npy")),
    key=lambda x: int(re.search(r'W_init_(\d+)\.npy', x).group(1))
)
W_final_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "W_final_[0-9]*.npy")),
    key=lambda x: int(re.search(r'W_final_(\d+)\.npy', x).group(1))
)
E_final_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "E_final_*.npy")),
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

def plot_hist(W_init, W_final, ax, config, threshold=None): 
    W_init = jnp.log10(W_init) 
    if threshold: 
        W_final = W_final[W_final > threshold] 
    W_final = jnp.log10(W_final) 
    ax.hist(W_init, alpha=0.6, density=True) 
    ax.hist(W_final, alpha=0.6, density=True) 
    ax.set_yscale("log")
    sigma, blocks = extract_sigma_and_blocks(config) 
    ax.set_title(fr"$\sigma_c = {sigma}$, blocks = {blocks}")
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
    
fig, axs = plt.subplots(7, 4, figsize=(7.08, 8), layout="constrained")
for i, ax in enumerate(axs.flatten()): 
    with open(config_files[i], "r") as c: 
        config = json.load(c)
    W_init = jnp.load(W_init_files[i]).flatten() 
    W_final = jnp.load(W_final_files[i]).flatten() 
    threshold = jnp.mean(W_init) 
    ax = plot_hist(W_init, W_final, ax, config, threshold) 
    log_xs = jnp.linspace(jnp.log10(threshold), jnp.max(jnp.log10(W_final)))
    pdf = fit_normal(W_final, threshold)
    ax.plot(log_xs, pdf(log_xs), label="log normal fit")

fig.savefig("W_distributions_environment_sweep.png", dpi=600)
fig.savefig("W_distributions_environment_sweep.pdf", dpi=600)
fig.savefig("W_distributions_environment_sweep.svg")

def compute_coexpression_matrix(E_file, threshold=0.01, normalize=True): 
    E = jnp.load(E_file) 
    print(jnp.median(E)) 
    binary_matrix = (E >= threshold).astype(int)
    coexpression_matrix = jnp.dot(binary_matrix.T, binary_matrix)
    if normalize: 
        coexpression_matrix /= jnp.mean(jnp.diag(coexpression_matrix)) 
    return coexpression_matrix

def compute_receptor_similarity(W_file, logscale=True, normalize=True): 
    W = jnp.load(W_file)
    if logscale: 
        W = jnp.log10(W) 
    receptor_similarity = jnp.dot(W, W.T)
    if normalize: 
        receptor_similarity /= jnp.mean(jnp.diag(receptor_similarity))
    return receptor_similarity


# fig, axs = plt.subplots(4, 4, figsize=(7.08, 5), layout="constrained")
# for i, ax in enumerate(axs.flatten()): 
#     E = jnp.load(E_final_files[i])
#     ax.hist(jnp.ravel(E))


# for W_file, E_file in zip(W_final_files, E_final_files):
#     receptor_similarity_matrix = compute_receptor_similarity(W_file)
#     coexpression_matrix = compute_coexpression_matrix(E_file) 
#     print(jnp.mean(coexpression_matrix))
#     break 

# fig, ax = plt.subplots() 
# ax.scatter(receptor_similarity_matrix, coexpression_matrix) 

# fig.savefig("coexpression_vs_receptor_affinity.png", dpi=600)

