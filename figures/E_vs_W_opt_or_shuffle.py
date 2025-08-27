import jax 
import jax.numpy as jnp
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import plot_utils
from scipy.stats import spearmanr, kendalltau, mannwhitneyu, gaussian_kde
import glob 
import re 
import json 


jax.config.update("jax_default_matmul_precision", "high") 

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"] 
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

'''config_{CONFIG_ID}.json contains all the hyperparameters and configurations needed to run the model and reproduce these results. For example, run 
python run_optimization.py --config config_0.json''' 

RESULTS_DIR = os.environ['RESULTS_DIR'] 

W_OPT_DIRECTORY = os.path.join(RESULTS_DIR, "environment_sweep/opt_E_given_opt_W/noncanonical_init") 
W_SHUFFLE_DIRECTORY = os.path.join(RESULTS_DIR, "environment_sweep/opt_E_given_shuffle_W/noncanonical_init/") 
CONFIG_ID = "17" # this corresponds to 64 sources, sigma_c = 2 in our phase diagram. A reasonable point in parameter space. 

def get_frequencies(config_path):
    with open(config_path, "r") as c: 
        config = json.load(c) 
    odorant_path = config["data_path"]
    odorants = jnp.load(odorant_path) 
    frequencies = jnp.mean(odorants, axis=1) 
    return frequencies

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

def compute_binarized_coexpression_matrix(E, threshold=0.2, normalize=False): 
    binary_matrix = (E >= threshold).astype(int)
    coexpression_matrix = jnp.dot(binary_matrix.T, binary_matrix)
    # coexpression_matrix = jnp.cov(binary_matrix.T)
    if normalize: 
        coexpression_matrix /= jnp.mean(jnp.diag(coexpression_matrix)) 
    return coexpression_matrix

def compute_coexpression_matrix(E): 
    return jnp.corrcoef(E) 

def plot_expression_heatmap(ax, expression): 
    threshold = 0.1 * jnp.max(expression, axis=1)
    indices = plot_utils.sort_rows_by_first_threshold(expression, threshold)
    im = ax.imshow(expression[indices, :], aspect="auto", cmap="Blues", interpolation="none")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([im.norm.vmin, im.norm.vmax])  # min and max values
    cbar.set_ticklabels([f"{im.norm.vmin:.0f}", f"{im.norm.vmax:.0f}"]) 
    ax.set_xticks([0, expression.shape[1] - 1]) 
    ax.set_yticks([0, expression.shape[0] - 1])
    ax.set_xticklabels(["1", str(expression.shape[1])])
    ax.set_yticklabels(["1", str(expression.shape[0])])
    ax.set_ylabel("neurons", labelpad=-20) 
    ax.set_xlabel("receptors", labelpad=-10) 
    return ax, cbar

# make panels a and b: expression heatmaps 
E_W_opt_path = os.path.join(W_OPT_DIRECTORY, f"E_final_{CONFIG_ID}.npy")
E_W_shuffle_path = os.path.join(W_SHUFFLE_DIRECTORY, f"E_final_{CONFIG_ID}.npy")

E_W_opt = jnp.load(E_W_opt_path)
E_W_shuffle = jnp.load(E_W_shuffle_path) 

fig, axs = plt.subplots(
    1, 2, 
    figsize=(3.2, 1.2),
    gridspec_kw={'wspace': 0.1}, 
    layout="constrained"
)

axs[0], cbar0 = plot_expression_heatmap(axs[0], E_W_opt)
axs[1], cbar1 = plot_expression_heatmap(axs[1], E_W_shuffle)

axs[0].set_title(r"using $W_{opt}$")
axs[1].set_title(r"using $W_{shuffle}$")
cbar0.remove() 

fig.savefig("E_vs_W_opt_or_shuffle.png", dpi=600)
fig.savefig("E_vs_W_opt_or_shuffle.pdf", dpi=600)
fig.savefig("E_vs_W_opt_or_shuffle.svg")

# make panel c: histograms of spearman correlations between pairs of receptors 
E_opt_files = sorted(
    glob.glob(os.path.join(W_OPT_DIRECTORY, "E_final_*.npy")),
    key=lambda x: int(re.search(r'E_final_(\d+)\.npy', x).group(1))
)

W_opt_files = sorted(
    glob.glob(os.path.join(W_OPT_DIRECTORY, "W_final_*.npy")),
    key=lambda x: int(re.search(r'W_final_(\d+)\.npy', x).group(1))
)

W_init_files = sorted(
    glob.glob(os.path.join(W_OPT_DIRECTORY, "W_init_*.npy")),
    key=lambda x: int(re.search(r'W_init_(\d+)\.npy', x).group(1))
)

W_shuffle_files = sorted(
    glob.glob(os.path.join(W_SHUFFLE_DIRECTORY, "W_final_*.npy")),
    key=lambda x: int(re.search(r'W_final_(\d+)\.npy', x).group(1))
)

config_files = sorted(
    glob.glob(os.path.join(W_OPT_DIRECTORY, "config_*.json")),
    key=lambda x: int(re.search(r'config_(\d+)\.json', x).group(1))
)

METRIC = "spearman" 
similarity_function = spearman_similarity

W_opt_coexpressed_sims = [] 
W_opt_not_coexpressed_sims = [] 
W_all_sims = [] 
W_shuffle_sims = [] 
coexpression_matrices = []
coexpression_scores = [] 

cutoff = 3 # coexpression needs to be somewhat robust... 

to_exclude = [3, 7, 11, 15, 19, 23, 27] # these have sigma_c = 4, which is too much noise. Those optimizations didn't move at all. 

for E_file, W_file, W_shuffle_file, W_init_file, config_file in zip(E_opt_files, W_opt_files, W_shuffle_files, W_init_files, config_files):
    id_ = int(re.search(r'E_final_(\d+)\.npy', E_file).group(1)) 
    if id_ in to_exclude: 
        print(f"excluding {E_file}") 
        continue 
    E = jnp.load(E_file) 
    W = jnp.load(W_file) 
    W_init = jnp.load(W_init_file) 
    # frequencies = get_frequencies(config_file) 
    # indices = jnp.argsort(frequencies) 
    # top_indices = indices[-100:]
    # W = W[:, top_indices] # get the top 100 odorants
    threshold = 1e-16
    try: 
        W_sim_matrix = compute_receptor_similarities(W, similarity_function, threshold=threshold)
    except: 
        print(f"problem file: {W_file}")
    coexpression_matrix = compute_binarized_coexpression_matrix(E)
    coexpression_score = compute_coexpression_matrix(E) 
    coexpressed_idx = jnp.argwhere((coexpression_matrix >= cutoff) & (jnp.triu(jnp.ones_like(coexpression_matrix), k=1) == 1))
    non_coexpressed_idx = jnp.argwhere((coexpression_matrix == 0) & (jnp.triu(jnp.ones_like(coexpression_matrix), k=1) == 1))
    i_upper, j_upper = jnp.triu_indices(coexpression_matrix.shape[0], k=1)
    coexpression_matrices.extend(coexpression_matrix[i_upper, j_upper])
    coexpression_scores.extend(coexpression_score[i_upper, j_upper])
    W_all_sims.extend(W_sim_matrix[i_upper, j_upper])
    W_opt_coexpressed = W_sim_matrix[coexpressed_idx[:, 0], coexpressed_idx[:, 1]]
    W_opt_coexpressed_sims.extend(W_opt_coexpressed)
    W_opt_not_coexpressed = W_sim_matrix[non_coexpressed_idx[:, 0], non_coexpressed_idx[:, 1]]
    W_opt_not_coexpressed_sims.extend(W_opt_not_coexpressed)
    W_shuffle = jnp.load(W_shuffle_file) 
    sim_matrix = compute_receptor_similarities(W_shuffle, similarity_function)
    upper_idx = jnp.triu_indices(sim_matrix.shape[0], k=1)
    upper_values = sim_matrix[upper_idx]
    W_shuffle_sims.extend(upper_values.tolist())
    

stat, p = mannwhitneyu(W_opt_not_coexpressed_sims, W_opt_coexpressed_sims, alternative="less")
power = int(np.floor(np.log10(p)))  # this gives the exponent such that 10^power <= p
threshold_power = power if 10**power == p else power + 1
p_string = f"p < 10^{threshold_power}"
print(p_string)

print(p) 

p_label = rf"$p < 10^{{{threshold_power}}}$"

if METRIC == "spearman": 
    hr = 4 
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.2, 2.0),
                                gridspec_kw={'height_ratios': [1, hr],
                                'hspace': 0.00}, 
                                layout="constrained")

    # Define histogram bins (optional, for consistency)
    bins = np.linspace(-0.4, 0.6, 40)

    # Plot histograms
    ax1.hist(W_opt_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:purple", label = fr"$\langle \text{{coexp}} \rangle = {jnp.mean(jnp.array(W_opt_coexpressed_sims)):.2f}$")
    ax1.hist(W_opt_not_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:blue", label=fr"$\langle \text{{others}} \rangle = {jnp.mean(jnp.array(W_opt_not_coexpressed_sims)):.2f}$")
    ax1.hist(W_shuffle_sims, bins=bins, alpha=0.6, density=True, color="tab:orange", label=fr"$\langle \text{{shuffle}} \rangle = {jnp.mean(jnp.array(W_shuffle_sims)):.2f}$")

    ax2.hist(W_opt_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:purple", label = fr"$\langle \text{{coexp.}} \rangle = {jnp.mean(jnp.array(W_opt_coexpressed_sims)):.2f}$")
    ax2.hist(W_opt_not_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:blue", label=fr"$\langle \text{{others}} \rangle = {jnp.mean(jnp.array(W_opt_not_coexpressed_sims)):.2f}$")
    ax2.hist(W_shuffle_sims, bins=bins, alpha=0.6, density=True, color="tab:orange", label=fr"$\langle \text{{shuffle}} \rangle = {jnp.mean(jnp.array(W_shuffle_sims)):.2f}$")

    # Set y-axis limits
    ax1.set_ylim(4, 20)  # Top plot shows the high spike
    ax2.set_ylim(0, 4)    # Bottom plot zooms into the rest
    ax2.set_yticks(jnp.arange(5))
    # Hide the spines between ax1 and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(bottom=False, labelbottom=False)
    ax2.tick_params(top=False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labelbottom=True)
    ax2.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])

    # Diagonal lines to indicate the break
    d = .005
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-hr*d, +hr*d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-hr*d, +hr*d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels and legend
    ax2.set_xlabel(rf"Spearman's $\rho$")
    ax2.set_ylabel("density")
    ax1.legend(handlelength=0.3, handletextpad=0.2, 
            borderpad=0.4, labelspacing=0.3, loc="upper left")
    ax2.text(0.7, 0.8, p_label, transform=ax2.transAxes)

elif METRIC == "correlation": 
    fig, ax = plt.subplots(figsize=(3.2, 2.0),
                        layout="constrained")
    bins = np.linspace(-0.4, 0.6, 40)
    ax.hist(W_opt_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:purple", label = fr"$\langle \text{{coexp}} \rangle = {jnp.mean(jnp.array(W_opt_coexpressed_sims)):.2f}$")
    ax.hist(W_opt_not_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:blue", label=fr"$\langle \text{{others}} \rangle = {jnp.mean(jnp.array(W_opt_not_coexpressed_sims)):.2f}$")
    ax.hist(W_shuffle_sims, bins=bins, alpha=0.6, density=True, color="tab:orange", label=fr"$\langle \text{{shuffle}} \rangle = {jnp.mean(jnp.array(W_shuffle_sims)):.2f}$")
    ax.set_xlabel(rf"{METRIC} of pair")
    ax.set_ylabel("density")
    ax.legend(handlelength=0.3, handletextpad=0.2, 
            borderpad=0.4, labelspacing=0.3, loc="upper right")
    ax.text(0.6, 0.5, p_label, transform=ax.transAxes)

else: 
    # fig, axs = plt.subplots(figsize=(3.2, 3.0),
    #                        layout="constrained")
    
    # bins = np.linspace(-4, 6, 200)
    # # Plot histograms
    # print(jnp.max(jnp.ravel(jnp.array(W_opt_coexpressed_sims)))) 
    # ax.hist(W_opt_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:purple", label = fr"$\langle \text{{coexp}} \rangle = {jnp.mean(jnp.array(W_opt_coexpressed_sims)):.2f}$")
    # ax.hist(W_opt_not_coexpressed_sims, bins=bins, alpha=0.6, density=True, color="tab:blue", label=fr"$\langle \text{{others}} \rangle = {jnp.mean(jnp.array(W_opt_not_coexpressed_sims)):.2f}$")
    # ax.hist(W_shuffle_sims, bins=bins, alpha=0.6, density=True, color="tab:orange", label=fr"$\langle \text{{shuffle}} \rangle = {jnp.mean(jnp.array(W_shuffle_sims)):.2f}$")
    # ax.set_xlabel(rf"{METRIC} of pair")
    # ax.set_ylabel("density")
    # ax.legend(handlelength=0.3, handletextpad=0.2, 
    #         borderpad=0.4, labelspacing=0.3, loc="upper right")
    # ax.text(0.6, 0.5, p_label, transform=ax.transAxes)

    coexp = np.array(W_opt_coexpressed_sims)
    non_coexp = np.array(W_opt_not_coexpressed_sims)
    shuffle = np.array(W_shuffle_sims)

    # Bin settings
    n_bins = 40
    bin_edges = np.linspace(-2, 3, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Prepend < -2 and append > 3 bins
    extended_bin_centers = np.concatenate(([-2.25], bin_centers, [3.25]))
    bin_width = bin_edges[1] - bin_edges[0]

    def extended_hist(data):
        clipped_low = np.sum(data < -2)
        clipped_high = np.sum(data > 3)
        in_range = data[(data >= -2) & (data <= 3)]
        hist, _ = np.histogram(in_range, bins=bin_edges, density=True)
        total_area = hist.sum() * bin_width
        # Add clipped bins as densities
        low_bin = clipped_low / len(data) / bin_width
        high_bin = clipped_high / len(data) / bin_width
        return np.concatenate(([low_bin], hist, [high_bin]))

    # Compute histograms
    hist_coexp = extended_hist(coexp)
    hist_noncoexp = extended_hist(non_coexp)
    hist_shuffle = extended_hist(shuffle)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(3.2, 2.4), layout="constrained")
    axs[0].bar(extended_bin_centers, hist_coexp, width=bin_width, alpha=0.6, color="tab:purple",
        label=fr"$\langle \text{{coexp}} \rangle = {np.mean(coexp):.2f}$")
    axs[0].bar(extended_bin_centers, hist_noncoexp, width=bin_width, alpha=0.6, color="tab:blue",
        label=fr"$\langle \text{{others}} \rangle = {np.mean(non_coexp):.2f}$")
    axs[0].bar(extended_bin_centers, hist_shuffle, width=bin_width, alpha=0.6, color="tab:orange",
        label=fr"$\langle \text{{shuffle}} \rangle = {np.mean(shuffle):.2f}$")

    # Label for axes
    axs[0].set_xlabel(rf"{METRIC} of pair")
    axs[0].set_ylabel("density")

    # Custom ticks to include clipping labels
    xticks = jnp.linspace(-2, 3, 6)
    xticklabels = ['<-2', '-1', '0', '1', '2', '>3']
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels(xticklabels)

    # Legend and text
    axs[0].legend(handlelength=0.3, handletextpad=0.2, 
            borderpad=0.4, labelspacing=0.3, loc="upper right")
    axs[0].text(0.6, 0.5, p_label, transform=axs[0].transAxes, fontsize=8)
    



# Assume W_all_sims and coexpression_scores are 1D numpy arrays
# x = np.array(W_all_sims)
# y = np.array(coexpression_scores)

# # Evaluate a gaussian kde on the data
# xy = np.vstack([x, y])
# z = gaussian_kde(xy)(xy)

# # Sort the points by density, so high-density points are plotted on top
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]

# fig, ax = plt.subplots(figsize=(3.4, 2.0), layout="constrained")
# sc = ax.scatter(x, y, c=z, cmap='viridis', s=10, edgecolor='none', alpha=0.8)
# fig.colorbar(sc, ax=ax, label='Density')
# fig.savefig("W_opt_coexpression_vs_similarity_density.png", dpi=600)

W_all_sims_np = np.array(W_all_sims)
coexpression_scores_np = np.array(coexpression_matrices)

# Binarize coexpression: 1 if coexpressed, 0 otherwise
coexpressed_binary = (coexpression_scores_np >= cutoff).astype(int)

# Define bins for similarity values
bins = np.linspace(0, 6, 31)  # 20 bins from 0.0 to 1.0
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# Digitize similarities into bins
bin_indices = np.digitize(W_all_sims_np, bins) - 1  # subtract 1 to make 0-indexed

# Compute conditional probability: P(coexpression | similarity bin)
prob_coexpress_given_sim = []
for i in range(len(bins) - 1):
    in_bin = (bin_indices == i)
    if np.sum(in_bin) > 0:
        prob = np.mean(coexpressed_binary[in_bin])
    else:
        prob = np.nan  # or 0.0
    prob_coexpress_given_sim.append(prob)

# Plotting
axs[1].grid() 
axs[1].set_yscale("log") 
axs[1].scatter(bin_centers, prob_coexpress_given_sim, marker='o', linestyle='-')
axs[1].set_xlabel("covariance of pair")
axs[1].set_ylabel("P(coexp | sim)")

fig.savefig(f"W_opt_or_shuffle_{METRIC}_similarity_histograms.png", dpi=600)
fig.savefig(f"W_opt_or_shuffle_{METRIC}_similarity_histograms.pdf", dpi=600)
fig.savefig(f"W_opt_or_shuffle_{METRIC}_similarity_histograms.svg")


# fig.savefig("P_coexpression_given_similarity.png", dpi=600)
# fig.savefig("P_coexpression_given_similarity.pdf", dpi=600)
# fig.savefig("P_coexpression_given_similarity.svg")