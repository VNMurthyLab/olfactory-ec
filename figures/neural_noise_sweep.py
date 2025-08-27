import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import os 
import plot_utils
from matplotlib.colors import rgb2hex

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
BASE_DIRECTORY = os.path.join(RESULTS_DIR, "neural_noise_sweep") 
SIGMA_0_DICT = {0: 0.01, 1: 0.03, 2: 0.1, 3: 0.3, 4: 1.0}
def plot_expression_heatmap(ax, expression, cbar=None, vmin=None, vmax=None, labels=False): 
    threshold = 0.1 * jnp.max(expression, axis=1)
    indices = plot_utils.sort_rows_by_first_threshold(expression, threshold)
    im = ax.imshow(expression[indices, :], aspect="auto", cmap="Blues", interpolation="none", vmin=vmin, vmax=vmax)
    if cbar is not None: 
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([im.norm.vmin, im.norm.vmax])  # min and max values
        cbar.set_ticklabels([f"{im.norm.vmin:.0f}", f"{im.norm.vmax:.0f}"])
    else: 
        cbar = None 
    ax.set_xticks([0, expression.shape[1] - 1]) 
    ax.set_yticks([0, expression.shape[0] - 1])
    if labels: 
        ax.set_xticklabels(["1", str(expression.shape[1])])
        ax.set_yticklabels(["1", str(expression.shape[0])])
        ax.set_ylabel("neurons", labelpad=-20) 
        ax.set_xlabel("receptors", labelpad=-10)
    else: 
        ax.set_xticklabels(["", ""]) 
        ax.set_yticklabels(["", ""]) 
    return ax, cbar

def compute_entropy(E): 
    return - jnp.mean(E * jnp.log(E))

def compute_canonical_score(E): 
    max_ent_E = 1 / E.shape[1] * jnp.ones(E.shape) 
    max_entropy = compute_entropy(max_ent_E)
    score = (max_entropy - compute_entropy(E)) / max_entropy 
    return score 

def plot_expression_for_increasing_noise(subdir, axs, ax_titles=False): 
    path = os.path.join(BASE_DIRECTORY, subdir)
    E_init_path = os.path.join(path, "E_init_0.npy")
    E_init = jnp.load(E_init_path) 
    axs[0], _ = plot_expression_heatmap(axs[0], E_init, labels=True)
    axs[0].set_title(r"$E_{init}$")
    for i in range(5): 
        E_path = os.path.join(path, f"E_final_{i}.npy")
        E = jnp.load(E_path) 
        cbar = None 
        if i == 4: 
            cbar = True 
        axs[i + 1], _ = plot_expression_heatmap(axs[i+1], E, cbar)
        if ax_titles: 
            if i > 0: 
                axs[i+1].set_title(rf"$\sigma_0 = {SIGMA_0_DICT[i]}$" + "\n" + r"$E_{opt}$" + f" ({compute_canonical_score(E):.2f})")
            else: 
                axs[i+1].set_title(r"$\sigma_0=0.01$" + "\n" + r"$E_{opt}$" + f" ({compute_canonical_score(E):.2f})")
        else: 
            axs[i+1].set_title(r"$E_{opt}$" + f" ({compute_canonical_score(E):.2f})") 
    return axs

fig, axs = plt.subplots(4, 6, figsize=(6.0, 4.5), layout="constrained", gridspec_kw={"wspace": 0.01, "hspace": 0.01})


subdir = "W_opt/canonical_init/"
axs[0, :] = plot_expression_for_increasing_noise(subdir, axs[0, :], ax_titles=True) 
subdir = "W_opt/noncanonical_init/"
axs[1, :] = plot_expression_for_increasing_noise(subdir, axs[1, :]) 
subdir = "W_shuffle/canonical_init/"
axs[2, :] = plot_expression_for_increasing_noise(subdir, axs[2, :]) 
subdir = "W_shuffle/noncanonical_init/"
axs[3, :] = plot_expression_for_increasing_noise(subdir, axs[3, :]) 

fig.savefig("neural_noise_sweep.png", dpi=600)
fig.savefig("neural_noise_sweep.pdf", dpi=600)
fig.savefig("neural_noise_sweep.svg")

