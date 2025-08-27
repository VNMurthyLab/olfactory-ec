import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import jax.numpy as jnp
import jax
import os
import json
import sys 
import glob
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba 


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

plt.rcParams['svg.fonttype'] = 'none'       # Preserve text (not paths)

RESULTS_DIR = os.environ['RESULTS_DIR'] 

DEFAULT_DATA_DIRECTORY = os.path.join(RESULTS_DIR, 'receptors/seeds_32') 
# Default config ID (use 32 to recreate receptor_figure.svg in paper)
DEFAULT_CONFIG_ID = '32_0'


# for panel e, which is specialists vs bottleneck. 
DEFAULT_M_SWEEP_DATA_DIRECTORY = os.path.join(RESULTS_DIR, 'sparsity_vs_N/seeds') 
DEFAULT_M_SWEEP_CONFIG_ID = 'config_'

# Use command-line arguments if provided, otherwise use defaults
DATA_DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIRECTORY
CONFIG_ID = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG_ID

M_SWEEP_DIRECTORY = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_M_SWEEP_DATA_DIRECTORY
M_SWEEP_CONFIG_ID = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_M_SWEEP_CONFIG_ID


def get_frequencies(): 
    config_path = os.path.join(DATA_DIRECTORY, f"config_{CONFIG_ID}.json")
    with open(config_path, "r") as c: 
        config = json.load(c) 
    odorant_path = config["data_path"]
    odorants = jnp.load(odorant_path) 
    frequencies = jnp.mean(odorants, axis=1) 
    return frequencies

def compute_sensitivity(kappa_inv, eta):
    # just do the algebra. set midpoint = hill function and multiply through by the denominator to calculate EC50. 
    max = 1 / (1 + 1 / eta) 
    midpoint = 0.5 * max 
    EC50 = -1 / (eta * kappa_inv + kappa_inv - eta * kappa_inv / midpoint)
    return 1 / EC50 

def compute_dynamic_tol(denom=50): 
    config_path = os.path.join(DATA_DIRECTORY, f"config_{CONFIG_ID}.json")
    with open(config_path, "r") as c: 
        config = json.load(c) 
    sigma_c = config["hyperparams"]["sigma_c"]
    mu_c = config["hyperparams"]["mu_c"]
    ''' Here's the reasoning. c is log normal (0, sigma_c). So c is at the very most e^3 sigma_c. 
    Firing rate is 1 / (1 + (Wc)^-1). Neural noise is 0.1. So for the firing rate to change appreciably compared to noise, let's conservatively say 
    1 / Wc < 10. Therefore 1 / We^3sigma_c < 10 --> W exp(3 sigma_c) > 1 / 10 --> W > 1 / 10 exp(3 sigma_c)''' 
    return 1 / (denom * jnp.exp(mu_c + 3 * sigma_c)) 

def plot_sensitivity_per_odorant_ticks(W, ax, tol, xmax, xmin=0, color="red"):
    row_maxima = jnp.max(W, axis=1) 
    sorted_indices = jnp.argsort(row_maxima)[::-1]
    W = W[sorted_indices]
    n_odorants, n_receptors = W.shape
    for i in range(n_odorants):
        y = i  # Row position (odorant)
        ax.hlines(y=y, xmin=xmin, xmax=xmax, color='gray', linewidth=0.1)  # Horizontal line
        for j in range(n_receptors):
            if W[i, j] > tol:
                ax.vlines(x=W[i, j], ymin=y-0.4, ymax=y+0.4, color=color, linewidth=.5)  # Vertical tick

    ax.set_xscale("log")
    ax.set_yticklabels([])
    # ax.set_ylabel("odorant") 
    ax.set_xlabel("sensitivity", labelpad=-1)
    return ax

def plot_tuning_curve_heatmap(W, ax, tol): 
    widths = jnp.sum(W > tol, axis=1)
    sorted_indices = jnp.argsort(widths)[::-1]
    W = jnp.sort(W[sorted_indices], axis=1) 
    ax.imshow(jnp.log10(W), aspect="auto", interpolation="nearest", cmap="Blues") 
    return ax 

def plot_CDF(W, ax, tol, color="red"):
    W = W[W > tol].flatten() 
    W = jnp.sort(W) 
    complement_CDF = 1 - jnp.arange(len(W)) / len(W) # this is just saying that the rank n largest value has probability 1 - n / total of some value being larger than it 
    ax.grid() 
    ax.scatter(W, complement_CDF, marker="o", facecolor="none", edgecolors=color, label="experiment") 
    ax.set_xscale("log") 
    ax.set_yscale("log")
    ax.set_ylabel(r"$P(X>x)$", labelpad=-2)
    ax.set_xlabel("x = sensitivity")
    return ax

def plot_CDF_line(W, ax, tol, color="red", alpha=1.0, logy=True):
    W = W[W > tol].flatten() 
    W = jnp.sort(W) 
    ys = 1 - jnp.arange(len(W)) / len(W)
    ax.plot(W, ys, color=color, alpha=alpha, linewidth=0.5) 
    ax.set_xscale("log") 
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(r"$P(X > x)$", labelpad=-2)
    ax.set_xlabel("x = sensitivity")
    return ax

def get_odorant_indices(num_odorants=34, index_type="evenly_spaced", k=34): 
    frequencies = get_frequencies()
    indices = jnp.argsort(frequencies)
    if index_type == "evenly_spaced": 
        evenly_spaced = jnp.linspace(0, 999, num_odorants, dtype=jnp.float32).astype(jnp.int32)
        return indices[evenly_spaced] 
    elif index_type == "top": # return top guys
        return indices[-k:]
    elif index_type == "from_top_k":
        evenly_spaced = jnp.linspace(0, k, num_odorants, dtype=jnp.float32).astype(jnp.int32)
        top_k = indices[-k:]
        return top_k[evenly_spaced]
    elif index_type == "from_bottom_k":
        evenly_spaced = jnp.linspace(0, k, num_odorants, dtype=jnp.float32).astype(jnp.int32)
        bottom_k = indices[:k]
        return bottom_k[evenly_spaced]
    
def odorant_frequency_barplot(indices, ax): 
    frequencies = get_frequencies() 
    ax.bar(indices, frequencies[indices]) 
    return ax 

def compute_dynamic_ranges(W, tol): # W should be odorant x receptor 
    dynamic_ranges = jnp.max(W, axis=1) / jnp.min(jnp.where(W > tol, W, jnp.inf), axis=1)    
    return dynamic_ranges  

def fetch_Ws(): 
    # Load simulation results
    W_init_paths = sorted(glob.glob(os.path.join(DATA_DIRECTORY, f"W_init_*.npy")))
    W_final_paths = sorted(glob.glob(os.path.join(DATA_DIRECTORY, f"W_final_*.npy")))
    W_inits, W_finals = [], [] 
    for W_init_path, W_final_path in zip(W_init_paths, W_final_paths): 
        W_init = jnp.load(W_init_path) 
        W_inits.append(W_init) 
        W_final = jnp.load(W_final_path) 
        W_finals.append(W_final) 
    return W_inits, W_finals

def fetch_W_and_M(top_k_odorants=1000): 
    # Load simulation results
    config_paths = sorted(glob.glob(os.path.join(M_SWEEP_DIRECTORY, f"{M_SWEEP_CONFIG_ID}*.json")))
    frequencies = get_frequencies()
    indices = jnp.argsort(frequencies)[::-1][:top_k_odorants]
    indep_var = []
    Ws = [] 
    gamma_Ts = [] 
    for config_path in config_paths: 
        with open(config_path, "r") as f: 
            config = json.load(f)
        sigma_c = config["hyperparams"]["sigma_c"]
        mu_c = config["hyperparams"]["mu_c"]
        M = config["hyperparams"]["M"]
        gamma = config["training"]["gamma_T"]
        id_ = config_path.split("/")[-1].strip("config_").strip(".json")
        W_path = os.path.join(M_SWEEP_DIRECTORY, f"W_final_{id_}.npy")
        
        if not os.path.exists(W_path):
            print(f"Skipping {id_}, missing W_final file.")
            continue

        W = jnp.load(W_path)[:, indices]
        # tol = compute_dynamic_tol(mu_c, sigma_c)
        tol = 1e-16
        Ws.append(jnp.clip(W, min=tol, max=jnp.inf))
        indep_var.append(M)
        gamma_Ts.append(gamma) 

        
    # Sort by sigma_c
    sorted_indices = jnp.argsort(jnp.array(indep_var))
    sigma_cs_sorted = [indep_var[i] for i in sorted_indices]
    Ws_sorted = [Ws[i] for i in sorted_indices]  
    gamma_Ts_sorted = [gamma_Ts[i] for i in sorted_indices] 
    return sigma_cs_sorted, Ws_sorted, gamma_Ts_sorted 

def compute_specialist_scores(W, quantile=0.5):
    denominators = jnp.quantile(W, quantile, axis=1) 
    maxs = jnp.max(W, axis=1) 
    return maxs / denominators 


OPTIMIZED_TOL = compute_dynamic_tol() # this computes an effective tolerance given the odor model parameters mu_c and sigma_c (which sets the largest possible odorant, and thus the smallest meaningful sensitivity)
EXPERIMENT_TOL = 10


# now make panel e

Ms, Ws, _ = fetch_W_and_M()

# key = jax.random.PRNGKey(0)
# keys = jax.random.split(key, len(Ws))
# shuffled_Ws = [
#     jax.random.permutation(k, W.ravel()).reshape(W.shape)
#     for k, W in zip(keys, Ws)
# ]

def compute_mean_and_std_specialist_counts(Ms, Ws): 
    specialist_receptor_counts = [] 
    for M, W, in zip(Ms, Ws):
        W_scores = compute_specialist_scores(W, quantile=0.99)
        specialist_receptor_counts.append(jnp.sum(W_scores > 1e2).item())
    Ms = jnp.array(Ms)
    counts = jnp.array(specialist_receptor_counts)
    unique_M = jnp.unique(Ms)
    means = [jnp.mean(counts[Ms == m]) for m in unique_M]
    stds = [jnp.std(counts[Ms == m]) for m in unique_M]
    return unique_M, means, stds 

def compute_rho(W): 
    rho = jnp.sum(W > 1e-6) / len(W.flatten())
    return rho 

def compute_receptor_std_dev(W): 
    return jnp.mean(jnp.std(W, axis=1))

def compute_multiplicative_dispersion(W): 
    row_means = jnp.mean(W, axis=1, keepdims=True)         # shape: (n_rows, 1)
    normalized = W / row_means                              # element-wise division
    rowwise_rms = jnp.sqrt(jnp.mean(normalized**2, axis=1))  # shape: (n_rows,)
    return jnp.mean(rowwise_rms)      

def rowwise_coefficient_of_variation(W):
    means = jnp.mean(W, axis=1)         # shape (n_rows,)
    stds = jnp.std(W, axis=1, ddof=0)   # population std
    return (jnp.mean(stds / means))

def fit_normal(log_W):
    log_vals = jnp.ravel(log_W)
    mean = jnp.mean(log_vals)
    std = jnp.std(log_vals)
    def normal_pdf(log_x):
        return (1 / (std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((log_x - mean) ** 2) / (2 * std ** 2))

    return normal_pdf

# unique_M, means, stds = compute_mean_and_std_specialist_counts(Ms, Ws) 
# unique_M, shuffled_means, shuffled_stds = compute_mean_and_std_specialist_counts(Ms, shuffled_Ws) 

# for one ax: 
# fig, ax = plt.subplots(figsize=(1.7, 1.2))
# Ms_array = jnp.array(Ms)  # or jnp.array(Ms)
# unique_Ms = jnp.unique(Ms_array)
# values_per_M = [] 
# for M_val in unique_Ms: 
#     # Get boolean mask for Ms == M_val
#     mask = Ms_array == M_val
#     matching_Ws = [W for W, match in zip(Ws, mask) if match]
#     print(matching_Ws[0].shape)
#     scores = []
#     for W in matching_Ws:
#         scores.append(rowwise_coefficient_of_variation(W)) 
#     values_per_M.append(jnp.mean(jnp.array(scores))) 

# ax.plot(unique_Ms, values_per_M)
# ax.grid() 
# ax.set_xscale("log")
# fig.savefig("scratch.png", dpi=600, bbox_inches=None)

# for many axs: 
fig, axs = plt.subplots(2, 5, figsize=(7, 3), sharey=True, sharex=True, layout="constrained")
Ms_array = jnp.array(Ms)  # or jnp.array(Ms)
unique_Ms = jnp.unique(Ms_array)
for M_val, ax in zip(unique_Ms, axs.flatten()) : 
    # Get boolean mask for Ms == M_val
    mask = Ms_array == M_val
    matching_Ws = jnp.clip(jnp.array([W for W, match in zip(Ws, mask) if match]), min=1e-9, max=jnp.inf)
    print(f"W shape = {matching_Ws[0].shape}")
    ax.hist(jnp.log10(matching_Ws.flatten()), density=True, bins=30) 
    log_xs = jnp.linspace(jnp.min(jnp.log10(matching_Ws)), jnp.max(jnp.log10(matching_Ws)), 100)
    pdf = fit_normal(jnp.log10(matching_Ws)) 
    ax.plot(log_xs, pdf(log_xs), label="log normal fit")
    ax.set_xlim(log_xs[0], log_xs[-1])
    ax.set_title(rf"$M={M_val}$", pad=-5)
    ax.set_yscale("log")
    if M_val == 10: 
        ax.legend() 
        ax.set_ylabel("density")
    if M_val == 244:
        ax.set_ylabel("density")
    if M_val > 200: 
        ax.set_xlabel("sensitivity")
    print(log_xs[0]) 

fig.suptitle("W histograms vs M: clip = 1e-9")
fig.savefig("W_histograms_vs_M_clip_9_logy.png", dpi=600, bbox_inches=None) 
fig.savefig("W_histograms_vs_M_clip_9_logy.pdf", dpi=600, bbox_inches=None)
fig.savefig("W_histograms_vs_M_clip_9_logy.svg", dpi=600, bbox_inches=None)


    