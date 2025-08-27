import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt 
import os 
import plot_utils
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
BASE_DIRECTORY = os.path.join(RESULTS_DIR, "vary_W_model") 

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
    ax.set_xticklabels(["", ""]) 
    ax.set_yticklabels(["", ""]) 
    if labels: 
        # ax.set_xticklabels(["1", str(expression.shape[1])])
        # ax.set_yticklabels(["1", str(expression.shape[0])])
        ax.set_ylabel("neurons") 
        ax.set_xlabel("receptors")
    return ax, cbar

def compute_entropy(E): 
    return - jnp.mean(E * jnp.log(E))

def compute_canonical_score(E): 
    max_ent_E = 1 / E.shape[1] * jnp.ones(E.shape) 
    max_entropy = compute_entropy(max_ent_E)
    score = (max_entropy - compute_entropy(E)) / max_entropy 
    return score 


def plot_init_final_pair(dir, id_, axs, cbar=None, vmin=None, vmax=None, labels=False, titles=False): 
    E_init_path = os.path.join(dir, f"E_init_{id_}.npy")
    E_init = jnp.clip(jnp.load(E_init_path), min=1e-16)
    E_final_path = os.path.join(dir, f"E_final_{id_}.npy")
    E_final = jnp.load(E_final_path) 
    axs[0], _ = plot_expression_heatmap(axs[0], E_init, vmin=vmin, vmax=vmax, labels=labels)
    axs[1], _ = plot_expression_heatmap(axs[1], E_final, cbar=cbar)
    axs[0].set_title(r"$E_{init}$" + f" ({compute_canonical_score(E_init):.2f})", fontsize=8) 
    axs[1].set_title(r"$E_{opt}$" + f" ({compute_canonical_score(E_final):.2f})", fontsize=8) 
    # else: 
    #     axs[0].set_title(f" ({compute_canonical_score(E_init):.2f})", fontsize=8) 
    #     axs[1].set_title(f" ({compute_canonical_score(E_final):.2f})", fontsize=8) 
    return axs 

def plot_row(dir, canonical_index, axs, ax_titles=False, labels=False, top_row=False):
    noncanonical_index = canonical_index + 1
    config_path = os.path.join(dir, f"config_{canonical_index}.json")
    with open(config_path, "r") as c:
        config = json.load(c)
    W_path = config["hyperparams"]["W_path"]
    print(f"canonical index: {canonical_index}; W path: {W_path}")
    W = jnp.load(W_path) 
    logW = jnp.log10(jnp.clip(W, 1e-16)) 
    W_cov = jnp.corrcoef(logW) 
    # W_cov = jnp.cov(logW)
    # axs[0].hist(logW.flatten())
    im = axs[0].imshow(W_cov, aspect=1.0, interpolation="none", cmap="Greens")
    axs[0].set_xticks([0, 59])
    axs[0].set_yticks([0, 59])
    axs[0].set_title(r"$\Sigma_W$")
    cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, location="left")
    cbar.set_ticks([im.norm.vmin, im.norm.vmax])  # min and max values
    # cbar.set_ticklabels([f"{im.norm.vmin:.0f}", f"{im.norm.vmax:.0f}"])
    cbar.set_ticklabels(["0", "1"])
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[[1, 2]] = plot_init_final_pair(dir, noncanonical_index, axs[[1, 2]], labels=labels, titles=top_row, cbar=True)
    # axs[[3, 4]] = plot_init_final_pair(dir, noncanonical_index, axs[[3, 4]], cbar=True, labels=False, titles=top_row)
    return axs 

fig, axs = plt.subplots(
    6, 3, 
    figsize=(3.4, 5), 
    gridspec_kw={"wspace": 0.05, "hspace": 0.05}, 
    layout="constrained"
)

w_models = [r"$W_{opt}$", r"$W_{shuffle}$", r"$W_{row shuffle}$", r"$W_{analytic}$", r"$W_{block}$", r"$W_{toeplitz}$"]


for i in range(6): 
    if i == 0: 
        top_row = True
    else: 
        top_row = False
    axs[i] = plot_row(BASE_DIRECTORY, i*2, axs[i], labels=top_row, top_row=top_row)
    bbox = axs[i, 0].get_position()  # Get the bounding box of the axis in figure coordinates
    x0, y0, width, height = bbox.bounds  # Extract the bounding box coordinates

    # Now place the text using the height of the axis
    fig.text(
        x0 - 0.5,  # X position is just to the left of the leftmost axis
        y0 + height / 2,  # Center the text vertically based on the axis height
        w_models[i],  # The text to display
        ha='center',  # Horizontal alignment
        va='center',  # Vertical alignment
        fontsize=10,  # Font size
        rotation=90  # Rotate text to be vertical
    )
    

fig.savefig("vary_W_model.png", dpi=600)
fig.savefig("vary_W_model.pdf")
fig.savefig("vary_W_model.svg")