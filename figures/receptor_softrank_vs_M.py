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
from scipy.stats import spearmanr, kendalltau
import numpy as np
jax.config.update("jax_default_matmul_precision", "high") 

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
DEFAULT_M_SWEEP_DATA_DIRECTORY = os.path.join(RESULTS_DIR, "/sparsity_vs_N/threshold_init/seeds") 
DEFAULT_M_SWEEP_CONFIG_ID = 'config_'
CLIP=1e-16

M_SWEEP_DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_M_SWEEP_DATA_DIRECTORY
M_SWEEP_CONFIG_ID = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_M_SWEEP_CONFIG_ID

CONFIG_ID = "0_0" 

def get_frequencies(): 
    config_path = os.path.join(M_SWEEP_DIRECTORY, f"config_{CONFIG_ID}.json")
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
    config_path = os.path.join(M_SWEEP_DIRECTORY, f"config_{CONFIG_ID}.json")
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
        return indices[-34:]
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

def cosine_similarity(W, logscale=True): 
    if logscale: 
        W = jnp.log10(jnp.clip(W, min=1e-16)) 
    norms = jnp.linalg.norm(W, axis=1, keepdims=True)  # Shape: (n_receptors, 1)
    W_normalized = W / (norms + 1e-8)  # Add small epsilon to prevent NaN
    receptor_similarity = jnp.dot(W_normalized, W_normalized.T)
    # receptor_similarity = jnp.dot(W, W.T)
    return receptor_similarity 

def spearman_similarity(W, logscale=True):
    if logscale:
        W = jnp.log10(jnp.clip(W, a_min=1e-16))

    # Convert JAX to NumPy (scipy doesn't support JAX arrays)
    W_np = jnp.array(W)
    rho, _ = spearmanr(W_np, axis=1)
    return jnp.array(rho)

def covariance(W, logscale=True):
    if logscale:
        W = jnp.log10(jnp.clip(W, a_min=1e-16))
    return jnp.cov(W) 

def correlation(W, logscale=True):
    if logscale:
        W = jnp.log10(jnp.clip(W, a_min=1e-16))
    return jnp.corrcoef(W) 

def kendall_tau_similarity(W, logscale=True):
    if logscale:
        W = jnp.log10(jnp.clip(W, a_min=1e-16))
    W = np.array(W)  # Convert to NumPy for scipy
    n_receptors = W.shape[0]
    sim = np.ones((n_receptors, n_receptors))
    for i in range(n_receptors):
        for j in range(i + 1, n_receptors):
            tau, _ = kendalltau(W[i], W[j])
            sim[i, j] = tau
            sim[j, i] = tau
    return jnp.array(sim)

def compute_receptor_similarities(W, metric, metric_args=None, threshold=1e-16):
    if metric_args is None:
        metric_args = {}
    W = jnp.where(W < threshold, 0, W) 
    sim = metric(W, **metric_args)
    return sim 

def compute_mean_and_std_receptor_similarities(Ms, Ws, similarity_function, threshold):
    receptor_similarities = []
    for W in Ws:
        W_sim_matrix = compute_receptor_similarities(W, similarity_function, threshold=threshold)
        i_upper, j_upper = jnp.triu_indices(W_sim_matrix.shape[0], k=1)
        receptor_similarities.append(W_sim_matrix[i_upper, j_upper])
    Ms = jnp.array(Ms)
    unique_M = jnp.unique(Ms)
    means, stds = [], []
    for M_val in unique_M:
        selected = [receptor_similarities[i] for i in range(len(Ms)) if Ms[i] == M_val]
        all_similarities = jnp.concatenate([jnp.array(s) for s in selected])
        means.append(jnp.mean(all_similarities))
        stds.append(jnp.std(all_similarities))
    return unique_M, jnp.array(means), jnp.array(stds)

def compute_PCs(W): 
    log_W = jnp.log10(W) 
    cov = jnp.cov(log_W)
    evalues, evectors = jnp.linalg.eigh(cov) 
    return jnp.sort(evalues / jnp.sum(evalues))[::-1], evectors 

def compute_soft_rank(W): 
    log_W = jnp.log10(W) 
    sigma = jnp.linalg.svd(log_W, compute_uv=False)  # Singular values
    sum_sigma = jnp.sum(sigma)
    sum_sigma_sq = jnp.sum(sigma ** 2)
    return (sum_sigma ** 2) / sum_sigma_sq

def compute_mean_and_std_receptor_soft_ranks(Ms, Ws): 
    soft_ranks = [] 
    for W in Ws:
        soft_rank = compute_soft_rank(W) 
        soft_ranks.append(soft_rank)
    Ms = jnp.array(Ms)
    unique_M = jnp.unique(Ms)
    means, stds = [], []
    for M_val in unique_M:
        selected = [soft_ranks[i] for i in range(len(Ms)) if Ms[i] == M_val]
        all_similarities = jnp.array(selected) 
        means.append(jnp.mean(all_similarities))
        stds.append(jnp.std(all_similarities))
    return unique_M, jnp.array(means), jnp.array(stds)

OPTIMIZED_TOL = compute_dynamic_tol() # this computes an effective tolerance given the odor model parameters mu_c and sigma_c (which sets the largest possible odorant, and thus the smallest meaningful sensitivity)
EXPERIMENT_TOL = 10
Ms, Ws, _ = fetch_W_and_M()

# unique_M, means, stds = compute_mean_and_std_specialist_counts(Ms, Ws) 
# unique_M, shuffled_means, shuffled_stds = compute_mean_and_std_specialist_counts(Ms, shuffled_Ws) 

# for one ax: 
fig, ax = plt.subplots(
    figsize=(3.35, 2), 
    layout="constrained")

# unique_M, cos_means, cos_stds = compute_mean_and_std_receptor_PCs(Ms, Ws, covariance, threshold=1e-16) 
# unique_M, spearman_means, spearman_stds = compute_mean_and_std_receptor_PCs(Ms, Ws, spearman_similarity, threshold=1e-16)
# # unique_M, kt_means, kt_stds = compute_mean_and_std_receptor_similarities(Ms, Ws, kendall_tau_similarity)

# ax.errorbar(unique_M, cos_means, yerr=cos_stds, fmt='o', capsize=5, linestyle='None', color='blue')
# ax.plot(unique_M, cos_means, color='blue', label="cosine")
# ax.errorbar(unique_M, spearman_means, yerr=spearman_stds, fmt='o', capsize=5, linestyle='None', color='tab:cyan')
# ax.plot(unique_M, spearman_means, color='tab:cyan', label="spearman")
# # ax.errorbar(unique_M, kt_means, yerr=kt_stds, fmt='o', capsize=5, linestyle='None', color='tab:blue')
# # ax.plot(unique_M, kt_means, color='tab:blue', label="kendall")


# ax.legend() 
# ax.set_xscale("log") 
# ax.grid() 
# ax.set_xlabel("M")
# ax.set_ylabel("similarity")
# fig.savefig("receptor_similarity_vs_M.png", dpi=600)


unique_M, means, stds = compute_mean_and_std_receptor_soft_ranks(Ms, Ws) 
ax.errorbar(unique_M, means, yerr=stds, fmt='o', capsize=5, linestyle='None', color='blue')
ax.plot(unique_M, means, color='blue', label=r"$(\Sigma \sigma_i)^2 / \Sigma \sigma_i^2$")

ax.legend()
ax.set_xscale("log")
ax.grid()
ax.set_xlabel("M")
ax.set_ylabel("soft rank")
fig.savefig("receptor_softrank_vs_M.png", dpi=600)
fig.savefig("receptor_softrank_vs_M.pdf", dpi=600)
fig.savefig("receptor_softrank_vs_M.svg", dpi=600)

    