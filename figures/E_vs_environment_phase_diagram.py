import jax.numpy as jnp
import numpy as np 
import json
import glob
import sys 
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import matplotlib.ticker as ticker
import argparse
import plot_utils 
import os 
import matplotlib as mpl
import re 

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
mpl.rcParams['svg.fonttype'] = 'none'

# use defaults to reproduce figure in paper 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate contour plots with optional log scaling')
    
    # Required arguments
    parser.add_argument('--output_directory', type=str, help='output directory', 
                        default="results/environment_sweep/opt_E_given_opt_W/noncanonical_init")
    parser.add_argument('--param1_key', type=str, help='Key for first parameter', 
                        default="data_path")
    parser.add_argument('--param2_key', type=str, help='Key for second parameter', 
                        default="sigma_c")
    # Optional argument with default False
    parser.add_argument('--log_param1', action='store_true',
                       help='Use log scale for param1 (default: False)')
    parser.add_argument('--overlay_values', action='store_true',
                       help='Write numbers on heatmap (default: False)')
    
    return parser.parse_args()

def compute_entropy(E): 
    return - jnp.mean(E * jnp.log(E))

def compute_canonical_score(E): 
    max_ent_E = 1 / E.shape[1] * jnp.ones(E.shape) 
    max_entropy = compute_entropy(max_ent_E)
    score = (max_entropy - compute_entropy(E)) / max_entropy 
    return score 

def compute_dynamic_tol(mu_c, sigma_c, denom=50): 
    return 1 / (denom * jnp.exp(mu_c + 3 * sigma_c)) 

def plot_expression_heatmap(expression): 
    fig, ax = plt.subplots(
        figsize=(1.7, 1.3), 
        layout="constrained", 
    )
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
    return fig, ax

def extract_W_key(output_dir): 
    if "shuffle_W" in output_dir: 
        W_key = "shuffle_W"
    elif "opt_W" in output_dir: 
        W_key = "opt_W"
    elif "analytic_fit_W": 
        W_key = "analytic_fit_W" 

    return W_key 

args = parse_arguments()

W_key = extract_W_key(args.output_directory) 
print(f"Computing phase diagram for {W_key}")

OD = args.output_directory
param1_key = args.param1_key
param2_key = args.param2_key
log_param1_scale = args.log_param1  # Will be False if not specified

E_files = sorted(glob.glob(f"{OD}/E_final_*.npy"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
W_files = sorted(glob.glob(f"{OD}/W_final_*.npy"), key=lambda x: int(x.split('_')[-1].split('.')[0]))
config_files = sorted(glob.glob(f"{OD}/config_*.json"), key=lambda x: int(x.split('_')[-1].split('.')[0]))

data = {
    param1_key: [],  # List of values for parameter 1 (from config)
    param2_key: [],  # List of values for parameter 2 (from config)
    "canonical score": [],  # List of computed metrics (from E_final)
}
to_exclude = [3, 7, 11, 15, 19, 23, 27] # these have sigma_c = 4, which is too much noise. Those optimizations didn't move at all. 
to_exclude = [] 

for E_file, config_file in zip(E_files, config_files):
    id_ = int(re.search(r'E_final_(\d+)\.npy', E_file).group(1)) 
    if id_ in to_exclude: 
        print(f"excluding {E_file}") 
        continue 
    # Load config and extract parameters
    with open(config_file, 'r') as f:
        config = json.load(f)
    if param1_key == "data_path": 
        data_path = config[param1_key] # this is not in hyperparams! 
        param1 = os.path.basename(data_path).strip("samples_block_").strip(".npy") 
    else: 
        param1 = config["hyperparams"][param1_key]  # Replace with your keys
    param2 = config["hyperparams"][param2_key]
    
    # Load E_final and compute metric (e.g., mean energy)
    E_final = jnp.load(E_file)
    metric = compute_canonical_score(E_final)
    
    # Append to dictionary
    data[param1_key].append(int(param1))
    data[param2_key].append(float(param2))
    data["canonical score"].append(float(metric))

param1 = jnp.array(data[param1_key])
param2 = jnp.array(data[param2_key])
metric = jnp.array(data["canonical score"])

if jnp.min(metric) < 1e-5:
    metric = jnp.log10(metric) 
    cbar_label = r'$\log_{10}(score)$'  # Label for log scale
else:
    cbar_label = 'score'  # Label for linear scale

# Get sorted unique values
unique_param1 = jnp.sort(jnp.unique(param1))
unique_param2 = jnp.sort(jnp.unique(param2))

# Number of bins
n_x = len(unique_param1)
n_y = len(unique_param2)

# Calculate bin edges
if log_param1_scale:
    log_param1 = jnp.log2(unique_param1)
    param1_edges = np.concatenate([
        [log_param1[0] - (log_param1[1]-log_param1[0])/2],  # Left edge
        (log_param1[1:] + log_param1[:-1])/2,               # Midpoints
        [log_param1[-1] + (log_param1[-1]-log_param1[-2])/2] # Right edge
    ])
else:
    param1_edges = np.concatenate([
        [unique_param1[0] - (unique_param1[1]-unique_param1[0])/2],
        (unique_param1[1:] + unique_param1[:-1])/2,
        [unique_param1[-1] + (unique_param1[-1]-unique_param1[-2])/2]
    ])

param2_edges = np.concatenate([
    [unique_param2[0] - (unique_param2[1]-unique_param2[0])/2],
    (unique_param2[1:] + unique_param2[:-1])/2,
    [unique_param2[-1] + (unique_param2[-1]-unique_param2[-2])/2]
])

# Create plot
fig, ax = plt.subplots(figsize=(3.4, 1.7), layout="constrained")

    # mosaic = [["E_init", "E_final", "MI"], ["hist_init", "hist_final", "params"]]
    # fig, axs = plt.subplot_mosaic(
    #     mosaic, 
    #     figsize=(18, 10),
    #     gridspec_kw={"hspace": 0.5}
    # )

# Plot using pcolormesh with exact edges
plot_data = metric.reshape(n_x, n_y).T  # Transpose for correct orientation

vmin = 0.74
vmax = 0.99

if log_param1_scale:
    mesh = ax.pcolormesh(param1_edges, param2_edges, plot_data,
                        shading='flat', cmap='viridis', vmin=vmin, vmax=vmax)
else:
    mesh = ax.pcolormesh(param1_edges, param2_edges, plot_data,
                        shading='flat', cmap='viridis', vmin=vmin, vmax=vmax)

# Set ticks at box centers
if log_param1_scale:
    ax.set_xticks(log_param1)
    # ax.set_xticklabels([fr'$2^{{{x:.0f}}}$' for x in log_param1])
    ax.set_xticklabels([f"{int(2**val)}" for val in log_param1])
else:
    ax.set_xticks(unique_param1)
    ax.set_xticklabels([f'{x:.1f}' for x in unique_param1])

ax.set_yticks(unique_param2)
ax.set_yticklabels([f'{y:.1f}' for y in unique_param2])

# Critical fix: Set grid lines at the EDGES (not centers)
ax.set_xticks(param1_edges, minor=True)
ax.set_yticks(param2_edges, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
ax.tick_params(which='minor', length=0)  # Hide minor ticks
# Increase tick label size for both axes

if args.overlay_values: 
    for i in range(plot_data.shape[0]):  # Loop over y-axis (rows)
        for j in range(plot_data.shape[1]):  # Loop over x-axis (columns)
            # Get the center of the current cell
            x_center = (param1_edges[j] + param1_edges[j+1]) / 2
            y_center = (param2_edges[i] + param2_edges[i+1]) / 2
            
            # Add the metric value as text
            ax.text(
                x_center, y_center,
                f"{plot_data[i, j]:.2f}",  # Format to 2 decimal places
                ha='center', va='center',  # Center the text
                color='white' if plot_data[i, j] > 0.5 * plot_data.max() else 'black', 
                fontsize="small"
        )


cbar = fig.colorbar(mesh, label=cbar_label) 
# vmin, vmax = mesh.get_clim()  # Get min/max from data
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

# Adjust layout
# plt.setp(ax.get_xticklabels())
# ax.set_xlabel(param1_key) 
# ax.set_ylabel(param2_key)

ax.set_xlabel(r"sources")
ax.set_ylabel(r"$\sigma_c$")


# Save
with open(f"{OD}/E_vs_environment_{W_key}_noncanonical_init_phase_diagram_data.json", 'w') as f:
    json.dump(data, f, indent=4) 

fig.suptitle(r"using $W_{opt}$")
fig.savefig(f"{OD}/E_vs_environment_{W_key}_noncanonical_init_phase_diagram.png", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram.png", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram.pdf", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram.svg")

# now plot the corners of the phase diagram: the distribution of W_{ij} and the resulting expression 

# bottom left: 
E = jnp.load(E_files[0])
# W = jnp.load("/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/expression/log_normal/sparsity_sweep/W_final_0.npy")
# E = jnp.load("/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/expression/log_normal/sparsity_sweep/E_final_0.npy")

fig, ax = plot_expression_heatmap(E)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_bottom_left.png", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_bottom_left.pdf", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_bottom_left.svg")

# bottom right: 
E = jnp.load(E_files[24])
# W = jnp.load("/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/expression/log_normal/sparsity_sweep/W_final_9.npy")
# E = jnp.load("/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/expression/log_normal/sparsity_sweep/E_final_9.npy")
fig, ax = plot_expression_heatmap(E)
ax.set_title("score = 0.84")
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_bottom_right.png", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_bottom_right.pdf", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_bottom_right.svg")

# top left: 
E = jnp.load(E_files[3])
fig, ax = plot_expression_heatmap(E)
ax.set_title("score = 0.74")
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_top_left.png", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_top_left.pdf", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_top_left.svg")


# top left: 
E = jnp.load(E_files[27])
fig, ax = plot_expression_heatmap(E)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_top_right.png", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_top_right.pdf", dpi=600)
fig.savefig(f"E_vs_environment/{W_key}_noncanonical_init_phase_diagram_top_right.svg")