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
DEFAULT_DIRECTORY = os.path.join(RESULTS_DIR, "environment_sweep/opt_W")
FLAT_FREQUENCIES_DIRECTORY = os.path.join(RESULTS_DIR, "flat_frequencies/environment_sweep/opt_W") # just FYI 

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
    ax.hist(W_init.flatten(), alpha=0.6, density=True) 
    ax.hist(W_final.flatten(), alpha=0.6, density=True) 
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
    return normal_pdf, mean, std 

def compute_PCs(W): 
    log_W = jnp.log10(W) 
    cov = jnp.cov(log_W)
    evalues, evectors = jnp.linalg.eigh(cov) 
    return evalues / jnp.sum(evalues), evectors 

def make_title(config): 
    data_path = config["data_path"]
    blocks = os.path.basename(data_path).strip("samples_block_").strip(".npy") 
    sigma_c = config["hyperparams"]["sigma_c"]
    title = fr"$\sigma_c ={sigma_c}$, blocks={blocks}"
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

def W_heatmap(ax, W, config_path): 
    with open(config_path, "r") as c: 
        config = json.load(c) 
    log_W = jnp.log10(W) 
    frequencies = get_frequencies(config_path) 
    order = jnp.argsort(frequencies)[::-1] 
    im = ax.imshow(log_W[:, order], aspect="auto", interpolation="none", cmap="Blues")
    ax.set_title(make_title(config)) 
    return im 

def W_correlation_heatmap(ax, W, config_path): 
    with open(config_path, "r") as c: 
        config = json.load(c) 
    log_W = jnp.log10(W) 
    cov = jnp.cov(log_W) 
    im = ax.imshow(cov) 
    ax.set_title(make_title(config))
    return im 

key = jax.random.key(0) 

ts = [] 
ss = [] 
titles = [] 
true_top_evecs = [] 
shuffle_top_evecs = [] 
Ws = [] 

for i in range(len(W_init_files)): 
    with open(config_files[i], "r") as c: 
        config = json.load(c)
    output_dir = config["logging"]["output_dir"]
    key, subkey_sample, subkey_shuffle = jax.random.split(key, 3)
    W_init = jnp.load(W_init_files[i])
    W_final = jnp.clip(jnp.load(W_final_files[i]), min=CLIP)
    Ws.append(W_final) 
    # threshold = jnp.min(W_init) 
    # pdf, mean, std = fit_normal(W_final.flatten(), threshold)
    shuffle_indices = jax.random.permutation(subkey_shuffle, len(W_final.ravel())) 
    W_shuffle = W_final.ravel()[shuffle_indices].reshape(W_final.shape)
    true_spectrum, true_evecs = compute_PCs(W_final)
    shuffle_spectrum, shuffle_evecs = compute_PCs(W_shuffle) 
    ts.append(true_spectrum)
    ss.append(shuffle_spectrum)
    titles.append(make_title(config))
    true_top_evecs.append(true_evecs[:, 0])
    shuffle_top_evecs.append(shuffle_evecs[:, 0])

fig, axs = plt.subplots(7, 4,
    figsize=(7, 9), 
    layout="constrained")

true_spectra = jnp.array(ts)
shuffle_spectra = jnp.array(ss) 
for i, ax in enumerate(axs.flatten()): 
    ax.scatter(jnp.arange(len(true_spectra[i])) - 0.2, jnp.sort(true_spectra[i])[::-1], s=3, zorder=3, label="true")
    ax.scatter(jnp.arange(len(shuffle_spectra[i])) + 0.2, jnp.sort(shuffle_spectra[i])[::-1], s=3, zorder=3, label="shuffle") 
    ax.set_yscale("log") 
    ax.grid() 
    if i == 0: 
        ax.legend() 
    try: 
        ax.set_title(titles[i]) 
    except: 
        pass 

if "flat" in DIRECTORY: 
    fig.savefig(f"W_spectra_clip_{CLIP:.0e}_flat_frequencies.png", dpi=600) 
    fig.savefig(f"W_spectra_clip_{CLIP:.0e}_flat_frequencies.pdf", dpi=600) 
    fig.savefig(f"W_spectra_clip_{CLIP:.0e}_flat_frequencies.svg")
else: 
    fig.savefig(f"W_spectra_clip_{CLIP:.0e}.png", dpi=600) 
    fig.savefig(f"W_spectra_clip_{CLIP:.0e}.pdf", dpi=600) 
    fig.savefig(f"W_spectra_clip_{CLIP:.0e}.svg")

fig, axs = plt.subplots(7, 4,
    figsize=(7, 9), 
    layout="constrained")

for i, ax in enumerate(axs.flatten()): 
    ax = evec_barplot(ax, true_top_evecs[i], jitter=-0.2, label="true") 
    ax = evec_barplot(ax, shuffle_top_evecs[i], jitter=+0.2, label="shuffle") 
    ax.grid() 
    if i == 0: 
        ax.legend() 
    ax.set_title(titles[i])


if "flat" in DIRECTORY: 
    fig.savefig(f"top_W_evectors_clip_{CLIP:.0e}_flat_frequencies.png", dpi=600) 
    fig.savefig(f"top_W_evectors_clip_{CLIP:.0e}_flat_frequencies.pdf", dpi=600) 
    fig.savefig(f"top_W_evectors_clip_{CLIP:.0e}_flat_frequencies.svg")
else: 
    fig.savefig(f"top_W_evectors_clip_{CLIP:.0e}.png", dpi=600) 
    fig.savefig(f"top_W_evectors_clip_{CLIP:.0e}.pdf", dpi=600) 
    fig.savefig(f"top_W_evectors_clip_{CLIP:.0e}.svg")


example_W_index = 27 

fig, ax = plt.subplots(
    figsize=(7, 5), 
    layout="constrained"
)

im = W_heatmap(ax, Ws[example_W_index], config_files[example_W_index]) 
fig.colorbar(im, ax=ax) 
ax.set_xlabel("odorants (decreasing frequency)")
ax.set_ylabel("receptors") 


if "flat" in DIRECTORY: 
    fig.savefig(f"W_{example_W_index}_heatmap_clip_{CLIP:.0e}_flat_frequencies.png", dpi=600) 
    fig.savefig(f"W_{example_W_index}_heatmap_clip_{CLIP:.0e}_flat_frequencies.pdf", dpi=600) 
    fig.savefig(f"W_{example_W_index}_heatmap_clip_{CLIP:.0e}_flat_frequencies.svg")
else: 
    fig.savefig(f"W_{example_W_index}_heatmap_clip_{CLIP:.0e}.png", dpi=600) 
    fig.savefig(f"W_{example_W_index}_heatmap_clip_{CLIP:.0e}.pdf", dpi=600) 
    fig.savefig(f"W_{example_W_index}_heatmap_clip_{CLIP:.0e}.svg")

    
fig, ax = plt.subplots(
    figsize=(2, 2), 
    layout="constrained"
)

im = W_correlation_heatmap(ax, Ws[example_W_index], config_files[example_W_index]) 
fig.colorbar(im, ax=ax) 

if "flat" in DIRECTORY: 
    fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}_flat_frequencies.png", dpi=600) 
    fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}_flat_frequencies.pdf", dpi=600) 
    fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}_flat_frequencies.svg")
else: 
    fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}.png", dpi=600) 
    fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}.pdf", dpi=600) 
    fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}.svg")

fig.savefig(f"W_{example_W_index}_covariance_clip_{CLIP:.0e}.png", dpi=600)
