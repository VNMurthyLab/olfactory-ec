import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob 
import sys 
import os 
import re 
import json 
from matplotlib.ticker import ScalarFormatter

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
DEFAULT_DIRECTORY = os.path.join(RESULTS_DIR, "glomerular_convergence/read_in_E") 
CONFIG_ID = 18
MI_CLIP = -1

DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY

def build_file_dict(pattern, regex_pattern):
    """Build a dictionary of {config_id: file_path} for files matching the pattern."""
    files = glob.glob(os.path.join(DIRECTORY, pattern))
    file_dict = {}
    for file_path in files:
        match = re.search(regex_pattern, file_path)
        if match:
            config_id = int(match.group(1))
            file_dict[config_id] = file_path
    return file_dict

import os
import re
import glob

def build_file_dict(glob_pattern, regex_pattern):
    """Create {config_id: file_path} dictionary for a file type."""
    return {
        int(re.search(regex_pattern, f).group(1)): f
        for f in glob.glob(os.path.join(DIRECTORY, glob_pattern))
        if re.search(regex_pattern, f)
    }

# Individual dictionaries (same names as your original variables)
r_init_files = build_file_dict("r_init_*.npy", r'r_init_(\d+)\.npy')
r_final_files = build_file_dict("r_final_*.npy", r'r_final_(\d+)\.npy')
G_init_files = build_file_dict("G_init_*.npy", r'G_init_(\d+)\.npy')
G_final_files = build_file_dict("G_final_*.npy", r'G_final_(\d+)\.npy')
gain_files = build_file_dict("gain_final_*.npy", r'gain_final_(\d+)\.npy')
E_final_files = build_file_dict("E_final_*.npy", r'E_final_(\d+)\.npy')
config_files = build_file_dict("config_*.json", r'config_(\d+)\.json')
mutual_information_files = build_file_dict("mutual_information_*.npy", r'mutual_information_(\d+)\.npy')


# r_init_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "r_init_*.npy")),
#     key=lambda x: int(re.search(r'r_init_(\d+)\.npy', x).group(1))
# )

# r_final_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "r_final_*.npy")),
#     key=lambda x: int(re.search(r'r_final_(\d+)\.npy', x).group(1))
# )

# G_init_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "G_init_*.npy")),
#     key=lambda x: int(re.search(r'G_init_(\d+)\.npy', x).group(1))
# )

# G_final_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "G_final_*.npy")),
#     key=lambda x: int(re.search(r'G_final_(\d+)\.npy', x).group(1))
# )

# gain_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "gain_final_*.npy")),
#     key=lambda x: int(re.search(r'gain_final_(\d+)\.npy', x).group(1))
# )

# E_final_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "E_final_*.npy")),
#     key=lambda x: int(re.search(r'E_final_(\d+)\.npy', x).group(1))
# )

# config_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "config_*.json")),
#     key=lambda x: int(re.search(r'config_(\d+)\.json', x).group(1))
# )

# mutual_information_files = sorted(
#     glob.glob(os.path.join(DIRECTORY, "mutual_information_*.npy")),
#     key=lambda x: int(re.search(r'mutual_information_(\d+)\.npy', x).group(1))
# )

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
        ax.hist(r_init.flatten(), alpha=0.6, density=True)
    ax.hist(r_final.flatten(), alpha=0.6, density=True)
    sigma, blocks = extract_sigma_and_blocks(config)
    ax.set_title(fr"$\sigma_c = {sigma}$, blocks = {blocks}")
    return ax

def sort_neurons_by_max_expression(expression_matrix):
    primary_receptors = jnp.argmax(expression_matrix, axis=1)
    max_values = jnp.max(expression_matrix, axis=1)
    neuron_order = jnp.lexsort((-max_values, primary_receptors))
    return expression_matrix[neuron_order], neuron_order

def sort_glomerular_matrix(G, neuron_order):
    G_sorted = G[:, neuron_order]
    glomerular_order = jnp.argsort(jnp.argmax(G_sorted, axis=1))
    return G_sorted[glomerular_order, :], glomerular_order

def compute_PCs(r): 
    cov = jnp.cov(r)
    evalues, evectors = jnp.linalg.eigh(cov) 
    return evalues / jnp.sum(evalues), evectors, 

def make_title(config): 
    data_path = config["data_path"]
    blocks = os.path.basename(data_path).strip("samples_block_").strip(".npy") 
    sigma_c = config["hyperparams"]["sigma_c"]
    title = fr"$\sigma_c ={sigma_c}$, blocks={blocks}"
    return title 

def compute_glomerular_activity(key, gain, r):
    return jnp.tanh(gain * r) + 0.1 * jax.random.normal(key, r.shape)

key = jax.random.key(0) 

E_final = jnp.load(E_final_files[CONFIG_ID])
G_init = jnp.load(G_init_files[CONFIG_ID])
G_final = jnp.load(G_final_files[CONFIG_ID])
mi = jnp.load(mutual_information_files[CONFIG_ID]) 

mosaic = [["mutual_information", "mutual_information"], ["G_init", "G_final"]]
fig, axs = plt.subplot_mosaic(
    mosaic, 
    figsize=(3.3, 2),
    gridspec_kw={"wspace": 0.5, "hspace": 0.5, "height_ratios": [1.5, 4]}
)

# make panel a: expression and optimization trajectory 
axs["G_init"].set_xlabel("neurons", labelpad=-10)
axs["G_init"].set_ylabel("glomeruli", labelpad=-4)  
axs["G_final"].set_xlabel("neurons", labelpad=-10)
axs["G_final"].set_ylabel("glomeruli")

for ax in [axs["G_init"], axs["G_final"]]: 
    ax.set_xticks([0, G_init.shape[1] - 1]) 
    ax.set_yticks([0, G_init.shape[0] - 1])
    ax.set_xticklabels(["1", str(G_init.shape[1])])
    ax.set_yticklabels(["1", str(G_init.shape[0])])

# axs["G_final"].set_xticks([])
axs["G_final"].set_yticks([])

E, neuron_order = sort_neurons_by_max_expression(E_final) 
G, glomerular_order = sort_glomerular_matrix(G_final, neuron_order)
G_init_sorted = G_init[glomerular_order, :] 
G_init_sorted = G_init_sorted[:, neuron_order]

im1 = axs["G_init"].imshow(G_init_sorted, aspect="auto", cmap="Blues", interpolation="none")
im2 = axs["G_final"].imshow(G, aspect="auto", cmap="Blues", interpolation="none")

axs["G_init"].set_title("initial")
axs["G_final"].set_title("optimized")

cbar1 = fig.colorbar(im1, ax=axs["G_init"])
cbar2 = fig.colorbar(im2, ax=axs["G_final"])

for cbar, im in zip([cbar1, cbar2], [im1, im2]):
    # Use 2 ticks: min and max of the image's normalization
    vmin, vmax = im.norm.vmin, im.norm.vmax
    cbar.set_ticks([vmin, vmax])

cbar1.set_ticklabels(["7.9e-4", "8.0e-4"])
cbar2.set_ticklabels(["0.00", "0.06"])
# axs["G_init"].set_title("Initial expression")
# axs["G_final"].set_title("Optimized expression") 
axs["mutual_information"].plot(jnp.clip(mi, min=MI_CLIP), color="blue") 
# axs["mutual_information"].set_title(r"Mutual information")
axs["mutual_information"].set_xlabel("epoch", labelpad=-10) 
axs["mutual_information"].set_ylabel(r"$\widehat{MI}(r, c)$", labelpad=-2)
axs["mutual_information"].set_yticks([0, 0.5, 1, 1.5])
axs["mutual_information"].set_xticks([0, 0.5e6, 1e6])
axs["mutual_information"].set_xticklabels(["0", "", "1e6"])
axs["mutual_information"].set_yticklabels(["0", "", "", "1.5"])
axs["mutual_information"].grid()

fig.savefig("glomerular_convergence.png", dpi=600)
fig.savefig("glomerular_convergence.pdf", dpi=600)
fig.savefig("glomerular_convergence.svg")

# make panel d: glomerular activity vs neural activity 

neural_activity = jnp.load(r_final_files[CONFIG_ID])
gain = jnp.load(gain_files[CONFIG_ID])
glomerular_activity = compute_glomerular_activity(key, gain, neural_activity)

# Define subplot layout using list of lists
mosaic = [
    ["left",   "main"],
    [None,     "bottom"]
]

fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=(3.3, 2.7),
    gridspec_kw={"height_ratios": [5, 1], "width_ratios": [1, 5], "wspace": 0.01, "hspace": 0.01},
    constrained_layout=True
)

fig.delaxes(axs[None])

# Access axes
ax_main = axs["main"]
ax_left = axs["left"]
ax_bottom = axs["bottom"]


# Main scatter plot
x = jnp.linspace(jnp.min(neural_activity), jnp.max(neural_activity), 100)
ax_main.plot(x, jnp.tanh(gain * x), label=r"$\tanh(\alpha_{opt} x)$")
ax_main.plot(x, jnp.tanh(x), label=r"$\tanh(\alpha_{init} x)$", ls="--")
# ax_main.set_xlabel("pre-synaptic activity")
# ax_main.set_ylabel("post-synaptic activity")
ax_main.set_xticklabels([])
ax_main.set_yticklabels([])
ax_main.grid() 
ax_main.legend() 

# Left marginal histogram (horizontal, pointing left)
ax_left.hist(glomerular_activity.flatten(), bins=40, orientation='horizontal', color='tab:blue')
ax_left.invert_xaxis()
# ax_left.set_xticks([])
# ax_left.set_yticks([])
ax_left.grid() 
ax_left.set_xticklabels([])
ax_left.set_ylabel("post-synaptic activity")

# Bottom marginal histogram (vertical, pointing down)
ax_bottom.hist(neural_activity.flatten(), bins=40, orientation='vertical', color='tab:blue', zorder=3)
ax_bottom.invert_yaxis()
# ax_bottom.set_xticks([])
# ax_bottom.set_yticks([])
ax_bottom.set_xlabel("pre-synaptic activity")
ax_bottom.set_yticklabels([])
ax_bottom.grid() 

fig.savefig("glomerular_histogram_equalization.png", dpi=600)
fig.savefig("glomerular_histogram_equalization.pdf", dpi=600)
fig.savefig("glomerular_histogram_equalization.svg")