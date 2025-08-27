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
from collections import Counter
from scipy.stats import gaussian_kde



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

# Default values (use /n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/receptors/seeds_32/ to recreate receptor_figure.svg in paper)
RESULTS_DIR = os.environ['RESULTS_DIR'] 

ONE_OVER_MEAN_INIT_DIRECTORY = os.path.join("receptors/seeds_32/")
THRESHOLD_INIT_DIRECTORY = os.path.join("receptors/threshold_init/seeds_32")
DEFAULT_DATA_DIRECTORY = ONE_OVER_MEAN_INIT_DIRECTORY
# Default config ID (use 32_0 to recreate receptor_figure.svg in paper)
DEFAULT_CONFIG_ID = '32_0'


# for panel e, which is specialists vs bottleneck. 
ONE_OVER_MEAN_INIT_M_SWEEP_DIRECTORY = '/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/sparsity_vs_N/seeds'
THRESHOLD_INIT_M_SWEEP_DIRECTORY = "/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/sparsity_vs_N/threshold_init/seeds"
DEFAULT_M_SWEEP_DATA_DIRECTORY = ONE_OVER_MEAN_INIT_M_SWEEP_DIRECTORY
DEFAULT_M_SWEEP_CONFIG_ID = 'config_'

# Use command-line arguments if provided, otherwise use defaults
DATA_DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIRECTORY
CONFIG_ID = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CONFIG_ID

M_SWEEP_DIRECTORY = sys.argv[3] if len(sys.argv) > 1 else DEFAULT_M_SWEEP_DATA_DIRECTORY
M_SWEEP_CONFIG_ID = sys.argv[4] if len(sys.argv) > 2 else DEFAULT_M_SWEEP_CONFIG_ID

SEPARATE_SHUFFLE_SCALE = False # use true if plotting threshold init.  

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

def compute_specialist_scores(W, quantile=0.99):
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

def fit_log_normal(W):
    log_vals = jnp.log(W.flatten())
    mean = jnp.mean(log_vals)
    std = jnp.std(log_vals)
    def log_normal_pdf(x):
        # Add a small epsilon to avoid log(0) or divide-by-zero issues
        eps = 1e-16
        x = jnp.maximum(x, eps)
        return (1 / (x * std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((jnp.log(x) - mean) ** 2) / (2 * std ** 2))

    return log_normal_pdf

def fit_normal(log_W):
    log_vals = jnp.ravel(log_W)
    mean = jnp.mean(log_vals)
    std = jnp.std(log_vals)
    def normal_pdf(log_x):
        return (1 / (std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((log_x - mean) ** 2) / (2 * std ** 2))

    return normal_pdf

def get_specialist_indices(W, quantile=0.99, cutoff=1e2): 
    scores = compute_specialist_scores(W, quantile)
    indices = jnp.where(scores > cutoff)[0]
    return indices 



OPTIMIZED_TOL = compute_dynamic_tol() # this computes an effective tolerance given the odor model parameters mu_c and sigma_c (which sets the largest possible odorant, and thus the smallest meaningful sensitivity)
EXPERIMENT_TOL = 10

# make panels d and e 

trends_mosaic = [["dynamic_range", "specialists"]]
trends_fig, trends_ax = plt.subplot_mosaic(
    trends_mosaic,
    figsize=(3.42, 1.2), 
    layout="constrained")

_, W_finals = fetch_Ws() 
frequencies = get_frequencies() 
dynamic_ranges = [] 
for W in W_finals:
    dynamic_ranges.append(compute_dynamic_ranges(W.T, tol=OPTIMIZED_TOL)) 

dynamic_ranges = jnp.array(dynamic_ranges) 
means = jnp.mean(dynamic_ranges, axis=0)
stds = jnp.std(dynamic_ranges, axis=0) 
yerr = [jnp.zeros_like(stds), stds]
# Plot and capture the line objects

alpha = 0.05

rgba_blue = to_rgba("blue", alpha=alpha)

# points (no outline)
trends_ax["dynamic_range"].scatter(
    frequencies, means,
    s=30, c=[rgba_blue], edgecolors='none', zorder=3
)

# error bars only
(line, caps, bars) = trends_ax["dynamic_range"].errorbar(
    frequencies, means, yerr=yerr,
    fmt='none',                      # no markers here
    ecolor='blue', elinewidth=1.0, capsize=5, capthick=1.0,
    zorder=2
)
for a in caps + bars:
    a.set_alpha(alpha)



# ax.scatter(frequencies, means, color='blue', alpha=0.7)
trends_ax["dynamic_range"].grid() 
# ax.set_xticks([0, 0.05, 0.1])
# ax.set_xticklabels([0, '', 0.1])
trends_ax["dynamic_range"].set_xlabel("frequency", labelpad=-2)
trends_ax["dynamic_range"].set_ylabel("dynamic range")
trends_ax["dynamic_range"].set_yscale("log") 

trends_fig.savefig(f"receptor_trends_{alpha}.png", dpi=600, bbox_inches=None)
trends_fig.savefig(f"receptor_trends_{alpha}.pdf", dpi=600, bbox_inches=None)
trends_fig.savefig(f"receptor_trends_{alpha}.svg", dpi=600, bbox_inches=None)
exit() 
# now make panel e

Ms, Ws, _ = fetch_W_and_M()
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, len(Ws))
shuffled_Ws = [
    jax.random.permutation(k, W.ravel()).reshape(W.shape)
    for k, W in zip(keys, Ws)
]

Ms_array = jnp.array(Ms)  # or jnp.array(Ms)
unique_Ms = jnp.unique(Ms_array)
values_per_M = [] 
key = jax.random.key(0) 
analytic_Ws = [] 
for M_val in unique_Ms: 
    key, subkey = jax.random.split(key) 
    mask = Ms_array == M_val
    matching_Ws = jnp.clip(jnp.array([W for W, match in zip(Ws, mask) if match]), min=1e-9, max=jnp.inf)
    print(f"W shape = {matching_Ws[0].shape}")
    mean, std = jnp.mean(jnp.log10(matching_Ws)), jnp.std(jnp.log10(matching_Ws)) 
    matching_analytic_Ws = jnp.exp(mean + std * jax.random.normal(subkey, shape = matching_Ws.shape))
    analytic_Ws.extend(matching_analytic_Ws)

unique_M, means, stds = compute_mean_and_std_specialist_counts(Ms, Ws) 
unique_M, shuffled_means, shuffled_stds = compute_mean_and_std_specialist_counts(Ms, shuffled_Ws) 
unique_M, analytic_means, analytic_stds = compute_mean_and_std_specialist_counts(Ms, analytic_Ws) 

if SEPARATE_SHUFFLE_SCALE: 
    # Create main axis and twin axis
    ax_main = trends_ax["specialists"]
    ax_shuffle = ax_main.twinx()
    ax_main.errorbar(unique_M, means, yerr=stds, fmt='o', capsize=5, 
                    linestyle='None', color='blue')
    ax_main.plot(unique_M, means, color="blue", label="true") 
    [print(unique_M[i], means[i]) for i in range(len(unique_M))] 
    ax_main.errorbar(unique_M, analytic_means, yerr=analytic_stds, fmt='o', 
                    capsize=5, linestyle='None', color='tab:blue')
    ax_main.plot(unique_M, analytic_means, color="tab:blue", label="fit") 
    ax_main.set_xscale("log") 
    ax_main.set_xticks([1e1, 1e2, 1e3])
    ax_main.set_xlim(unique_M[0], unique_M[-1])
    ax_main.grid()
    ax_shuffle.errorbar(unique_M, shuffled_means, yerr=shuffled_stds, fmt='o', 
                    capsize=5, linestyle='None', color='tab:cyan')
    ax_shuffle.plot(unique_M, shuffled_means, color="tab:cyan", label="shuffle") 
    ax_shuffle.set_yscale("log") 
    ax_main.set_xlabel("M")
    ax_main.set_ylabel("true/fit", color="blue")
    ax_main.tick_params(axis="y", labelcolor="blue")
    ax_main.set_yticks([0, 60])
    ax_shuffle.set_yticks([1e0, 1e3])
    ax_shuffle.set_ylabel("shuffle", color='tab:cyan')
    ax_shuffle.tick_params(axis='y', labelcolor='tab:cyan')
    lines_main, labels_main = ax_main.get_legend_handles_labels()
    lines_shuffle, labels_shuffle = ax_shuffle.get_legend_handles_labels()
    # ax_main.legend(lines_main + lines_shuffle, labels_main + labels_shuffle, loc='best')
else: 
    trends_ax["specialists"].errorbar(unique_M, means, yerr=stds, fmt='o', capsize=5, linestyle='None', color='blue')
    trends_ax["specialists"].plot(unique_M, means, color="blue", label="true") 
    [print(unique_M[i], means[i]) for i in range(len(unique_M))] 
    trends_ax["specialists"].errorbar(unique_M, shuffled_means, yerr=shuffled_stds, fmt='o', capsize=5, linestyle='None', color='tab:cyan')
    trends_ax["specialists"].plot(unique_M, shuffled_means, color="tab:cyan", label="shuffle") 
    trends_ax["specialists"].errorbar(unique_M, analytic_means, yerr=analytic_stds, fmt='o', capsize=5, linestyle='None', color='tab:blue')
    trends_ax["specialists"].plot(unique_M, analytic_means, color="tab:blue", label="fit") 
    trends_ax["specialists"].set_xscale("log") 
    trends_ax["specialists"].grid() 
    # trends_ax["specialists"].set_yticks([0, 10, 20, 30])
    # trends_ax["specialists"].set_yscale("log")
    trends_ax["specialists"].set_xlabel("receptors", labelpad=-2)
    trends_ax["specialists"].set_ylabel("specialists")
    trends_ax["specialists"].legend(handlelength=0.5, handletextpad=0.5)

trends_fig.savefig("receptor_trends.png", dpi=600, bbox_inches=None)
trends_fig.savefig("receptor_trends.pdf", dpi=600, bbox_inches=None)
trends_fig.savefig("receptor_trends.svg", dpi=600, bbox_inches=None)

# make panels a b c 

W_init_path = os.path.join(DATA_DIRECTORY, f"W_init_{CONFIG_ID}.npy")   
W_init = jnp.load(W_init_path) 
W_init = jnp.clip(W_init, min=1e-16)
W_final_path = os.path.join(DATA_DIRECTORY, f"W_final_{CONFIG_ID}.npy")   
W_final = jnp.load(W_final_path) 
W_final = jnp.clip(W_final, min=1e-16)

MI_path = os.path.join(DATA_DIRECTORY, f"mutual_information_{CONFIG_ID}.npy")
MI = jnp.load(MI_path)

# only if you optimized kappa_inv and eta under the antagonism model should you run this. 
# kappa_inv_path = os.path.join(DATA_DIRECTORY, f"kappa_inv_{CONFIG_ID}.npy")
# eta_path = os.path.join(DATA_DIRECTORY, f"eta_{CONFIG_ID}.npy")
# kappa_inv = jnp.load(kappa_inv_path)
# eta = jnp.load(eta_path)
# W_final = jnp.clip(W_final, min=1e-9)
# W_final = compute_sensitivity(kappa_inv, eta) 
# W_final *= 10**9 # put our results on the same scale as Si et al. 2019 by matching the maximum sensitivity. This is slightly cheating! Think through this more properly... 


experimental_data = jnp.load("/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/si_et_al_2019_data/cMatrixMLE.npy") # this is odorant by receptor 
EC50 = jnp.nan_to_num(experimental_data, jnp.inf) # this sets W to 0 when EC50 is nan. 
W_exp = 1 / 10**EC50

ticks_mosaic = [["MI", "MI", "MI"], ["ticks_init", "ticks_opt", "ticks_exp"], ["CDF_init", "CDF_opt", "CDF_exp"]]
ticks_fig, ticks_axs = plt.subplot_mosaic(
    ticks_mosaic, 
    figsize=(3.42, 4.0),
    gridspec_kw={"wspace": 0.0, "hspace": 0.0, "height_ratios": [1, 3, 1]}, 
    layout="constrained"
)

# === Manually share x-axis between rows 2 and 3 ===
for key_upper, key_lower in zip(["ticks_init", "ticks_opt", "ticks_exp"],
                                ["CDF_init", "CDF_opt", "CDF_exp"]):
    ticks_axs[key_lower].sharex(ticks_axs[key_upper])

# === Manually share y-axis within row 3 ===
ticks_axs["CDF_opt"].sharey(ticks_axs["CDF_init"])
ticks_axs["CDF_exp"].sharey(ticks_axs["CDF_init"])

# make figure 3C of Si et al. 2019. y axis is odorants, x axis is sensitivity, for each odorant place a tick mark where each receptor has that sensitivity value. 

W_indices = get_odorant_indices(34, index_type="from_top_k", k=500) # get the top odorants
ticks_axs["ticks_init"] = plot_sensitivity_per_odorant_ticks(W_init[:, W_indices].T, ticks_axs["ticks_init"], OPTIMIZED_TOL, xmax=W_final.max(), color="blue")
ticks_axs["ticks_opt"] = plot_sensitivity_per_odorant_ticks(W_final[:, W_indices].T, ticks_axs["ticks_opt"], OPTIMIZED_TOL, xmax=W_final.max(), color="blue")
ticks_axs["ticks_exp"] = plot_sensitivity_per_odorant_ticks(W_exp, ticks_axs["ticks_exp"], EXPERIMENT_TOL, xmax=W_exp.max())
ticks_axs["ticks_init"].set_xlim(1e-9, 1e-1)
ticks_axs["ticks_opt"].set_xlim(1e-9, 1e-1)
ticks_axs["ticks_exp"].set_xlim(1e1, 2 * 1e9)
ticks_axs["ticks_init"].set_xticks([1e-8, 1e-5, 1e-2])
ticks_axs["ticks_opt"].set_xticks([1e-8, 1e-5, 1e-2])
ticks_axs["ticks_exp"].set_xticks([1e2, 1e5, 1e8])

ticks_axs["MI"].plot(MI, color="blue")
ticks_axs["MI"].set_ylim(0, 1.5)
ticks_axs["MI"].set_yticks([0, 0.5, 1, 1.5])
ticks_axs["MI"].set_yticklabels([0, '', '', 1.5])
ticks_axs["MI"].grid()
ticks_axs["MI"].set_ylabel(r"$\widehat{MI}(r, c)$")
ticks_axs["MI"].set_xlabel("epochs", labelpad=-12)
ticks_axs["MI"].ticklabel_format(style='plain', axis='x')
ticks_axs["MI"].set_xticks([0, .5 * 1e6, 1e6])
ticks_axs["MI"].set_xticklabels(['0', '', '1e6'])


ticks_axs["ticks_init"].set_title(r"initial")
ticks_axs["ticks_opt"].set_title(r"optimized")
ticks_axs["ticks_exp"].set_title(r"fly larva")
ticks_axs["ticks_init"].set_ylabel("odorants")



colors = [mcolors.hsv_to_rgb((0.66, s, 0.9)) for s in 
          jnp.linspace(0.3, 1.0, 20)]  # Vary saturation


ticks_axs["CDF_exp"] = plot_CDF_line(W_exp, ticks_axs['CDF_exp'], EXPERIMENT_TOL, logy=True)
W_inits, W_finals = fetch_Ws() 
for W_init, W_final, c in zip(W_inits, W_finals, colors):
    ticks_axs["CDF_init"] = plot_CDF_line(W_init[:, W_indices], ticks_axs["CDF_init"], OPTIMIZED_TOL, color=c, logy=True)
    ticks_axs["CDF_opt"] = plot_CDF_line(W_final[:, W_indices], ticks_axs["CDF_opt"], OPTIMIZED_TOL, color=c, logy=True)

ticks_axs["CDF_init"].set_xticks([1e-8, 1e-5, 1e-2])
ticks_axs["CDF_opt"].set_xticks([1e-8, 1e-5, 1e-2])
ticks_axs["CDF_exp"].set_xticks([1e2, 1e5, 1e8])

[ticks_axs[ax].grid() for ax in ["CDF_init", "CDF_exp", "CDF_opt"]]


CDF_linear_mosaic = [["CDF_init", "CDF_opt", "CDF_exp"]]
CDF_linear_fig, CDF_linear_axs = plt.subplot_mosaic(
    CDF_linear_mosaic, 
    figsize=(4.5, 1.5), 
    gridspec_kw={"wspace": 0.0}, 
    sharey=True, 
    layout="constrained"
    )


CDF_linear_axs["CDF_exp"] = plot_CDF_line(W_exp, CDF_linear_axs['CDF_exp'], EXPERIMENT_TOL, logy=False)
W_inits, W_finals = fetch_Ws() 
for W_init, W_final, c in zip(W_inits, W_finals, colors):
    CDF_linear_axs["CDF_init"] = plot_CDF_line(W_init[:, W_indices], CDF_linear_axs["CDF_init"], OPTIMIZED_TOL, color=c, logy=False)
    CDF_linear_axs["CDF_opt"] = plot_CDF_line(W_final[:, W_indices], CDF_linear_axs["CDF_opt"], OPTIMIZED_TOL, color=c, logy=False)

CDF_linear_axs["CDF_init"].set_xticks([1e-8, 1e-5, 1e-2])
CDF_linear_axs["CDF_opt"].set_xticks([1e-8, 1e-5, 1e-2])
CDF_linear_axs["CDF_exp"].set_xticks([1e2, 1e5, 1e8])

[CDF_linear_axs[ax].grid() for ax in ["CDF_init", "CDF_exp", "CDF_opt"]]

base_path = "/n/home10/jfernandezdelcasti/noncanonical-olfaction"
relative_path = os.path.relpath(DATA_DIRECTORY, base_path)

for ax in ["ticks_init", "ticks_opt", "ticks_exp"]:
    ticks_axs[ax].tick_params(labelbottom=False, bottom=False)
    ticks_axs[ax].set_xlabel('')  # Optional: remove x-axis label

for ax in ["CDF_opt", "CDF_exp"]:
    ticks_axs[ax].tick_params(labelleft=False, left=False)
    ticks_axs[ax].set_ylabel('')

ticks_fig.savefig("receptor_ticks_figure.png", dpi=600, bbox_inches=None) 
ticks_fig.savefig("receptor_ticks_figure.pdf", dpi=600, bbox_inches=None)
ticks_fig.savefig("receptor_ticks_figure.svg", dpi=600, bbox_inches=None)