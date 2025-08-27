import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob 
import sys 
import os 
import re 
import json 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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

DEFAULT_DATA_DIRECTORY = os.path.join(RESULTS_DIR, "environment_sweep", "opt_W")

DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIRECTORY
DEFAULT_CONFIG_ID = '17'

def extract_sigma_and_blocks(config): 
    sigma = config["hyperparams"]["sigma_c"]
    blocks = os.path.basename(config["data_path"]).strip("samples_block_").strip(".npy")
    return sigma, blocks 

def make_title(config): 
    sigma_c, blocks = extract_sigma_and_blocks(config) 
    title = fr"$\sigma_c ={sigma_c}$, blocks={blocks}"
    return title 

def get_frequencies(config_path):
    with open(config_path, "r") as c: 
        config = json.load(c) 
    odorant_path = config["data_path"]
    odorants = jnp.load(odorant_path) 
    frequencies = jnp.mean(odorants, axis=1) 
    return frequencies

def W_heatmap(ax, W, config_path): 
    with open(config_path, "r") as c: 
        config = json.load(c) 
    log_W = jnp.log10(W) 
    frequencies = get_frequencies(config_path) 
    order = jnp.argsort(frequencies)[::-1] 
    im = ax.imshow(log_W[:, order], aspect="auto", interpolation="none", cmap="Blues")
    ax.set_title("Optimized W (adult fly)")
    return im, log_W[:, order], order

def W_heatmap_tuning_curves(ax, W, config_path, clip=CLIP):
    with open(config_path, "r") as c:
        config = json.load(c)
    W_clipped = jnp.clip(W, a_min=clip)
    W_log = jnp.log10(W_clipped)
    W_sorted_per_receptor = jnp.sort(W_log, axis=1)[:, ::-1]  # sort descending per row
    tuning_widths = jnp.sum(W > clip, axis=1)
    receptor_order = jnp.argsort(tuning_widths)
    W_sorted_final = W_sorted_per_receptor[receptor_order, :]
    im = ax.imshow(W_sorted_final, aspect="auto", interpolation="none", cmap="Blues")
    ax.set_title("Optimized W (adult fly)")
    ax.set_xlabel("odorants (sorted per neuron by sensitivity)")
    ax.set_ylabel("receptors (sorted by tuning breadth)")
    return im


# Load data
W_path = os.path.join(DIRECTORY, f"W_final_{DEFAULT_CONFIG_ID}.npy")
config_path = os.path.join(DIRECTORY, f"config_{DEFAULT_CONFIG_ID}.json")
W_final = jnp.clip(jnp.load(W_path), min=CLIP)

key = jax.random.PRNGKey(0)

# Flatten, shuffle, and reshape W_final to create W_shuf
W_flat = W_final.flatten()
W_flat_shuffled = jax.random.permutation(key, W_flat)
W_shuf = W_flat_shuffled.reshape(W_final.shape)

fig = plt.figure(figsize=(7, 7), layout="constrained")
gs = gridspec.GridSpec(2, 4, width_ratios=[10, 1.5, 0.5, 0.5], height_ratios=[1, 1], figure=fig)

# --- Top panel: original heatmap with tuning bar ---
ax_heatmap_top = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_heatmap_top)
ax_cbar_top = fig.add_subplot(gs[0, 2])

im_top, log_W_ordered, order = W_heatmap(ax_heatmap_top, W_final, config_path)
W_ordered = W_final[:, order]
W_shuf_ordered = W_shuf[:, order]

# Mask and log
mask_real = W_ordered > CLIP
mask_shuf = W_shuf_ordered > CLIP
log_W_ordered = jnp.where(mask_real, jnp.log10(W_ordered), jnp.nan)
log_W_shuf_ordered = jnp.where(mask_shuf, jnp.log10(W_shuf_ordered), jnp.nan)

# Bar plot
mean_real = jnp.nanmean(log_W_ordered, axis=1)
mean_shuf = jnp.nanmean(log_W_shuf_ordered, axis=1)
y_positions = jnp.arange(len(mean_real))
bar_height = 0.35
ax_bar.barh(y_positions + bar_height/2, mean_real, height=bar_height, color="tab:blue", label="real")
ax_bar.barh(y_positions - bar_height/2, mean_shuf, height=bar_height, color="tab:orange", label="shuffled")
ax_bar.set_xlabel("mean tuning")
ax_bar.tick_params(left=False, labelleft=False)
ax_bar.set_xticks(-jnp.arange(8))
ax_bar.set_xticklabels(["0", "", "", "", "", "-5", "", ""])
ax_bar.grid()
ax_bar.invert_xaxis()
ax_heatmap_top.set_ylabel("receptors")
ax_heatmap_top.set_xlabel("odorants (decreasing frequency)")
cbar_top = fig.colorbar(im_top, cax=ax_cbar_top)
cbar_top.set_label(r"$\log_{10}(W_{ij})$")

# --- Bottom panel: tuning-curve-sorted heatmap + bar plot ---
ax_heatmap_bottom = fig.add_subplot(gs[1, 0], sharey=ax_heatmap_top)
ax_bar_bottom = fig.add_subplot(gs[1, 1], sharey=ax_heatmap_bottom)
ax_cbar_bottom = fig.add_subplot(gs[1, 2])

im_bottom = W_heatmap_tuning_curves(ax_heatmap_bottom, W_final, config_path)


# Bar plot (same values)
ax_bar_bottom.barh(y_positions + bar_height/2, mean_real, height=bar_height, color="tab:blue", label="real")
ax_bar_bottom.barh(y_positions - bar_height/2, mean_shuf, height=bar_height, color="tab:orange", label="shuffled")
ax_bar_bottom.set_xlabel("mean tuning")
ax_bar_bottom.tick_params(left=False, labelleft=False)
ax_bar_bottom.set_xticks(-jnp.arange(8))
ax_bar_bottom.set_xticklabels(["0", "", "", "", "", "-5", "", ""])
ax_bar_bottom.grid()
ax_bar_bottom.invert_xaxis()
ax_heatmap_bottom.set_xlabel("odorants (sorted per neuron by sensitivity)")
ax_heatmap_bottom.set_ylabel("receptors")
cbar_bottom = fig.colorbar(im_bottom, cax=ax_cbar_bottom)
cbar_bottom.set_label(r"$\log_{10}(W_{ij})$")


fig.savefig(f"W_adult_fly_dual_heatmap_clip_{CLIP:.0e}.png", dpi=600)
fig.savefig(f"W_adult_fly_dual_heatmap_clip_{CLIP:.0e}.pdf", dpi=600)
fig.savefig(f"W_adult_fly_dual_heatmap_clip_{CLIP:.0e}.svg")


# # Layout
# fig = plt.figure(figsize=(7, 3.5), layout="constrained")
# gs = gridspec.GridSpec(1, 3, width_ratios=[10, 1.5, 0.5], figure=fig)

# ax_heatmap = fig.add_subplot(gs[0])
# ax_heatmap.set_yticks(range(60))
# ax_heatmap.set_yticklabels(["1"] + [""]*58 + ["60"])
# ax_bar = fig.add_subplot(gs[1], sharey=ax_heatmap)
# ax_cbar = fig.add_subplot(gs[2])

# # Heatmap
# im, log_W_ordered, order = W_heatmap(ax_heatmap, W_final, config_path)

# W_ordered = W_final[:, order]
# W_shuf_ordered = W_shuf[:, order]

# # Mask out values below the clip
# mask_real = W_ordered > CLIP
# mask_shuf = W_shuf_ordered > CLIP

# # Avoid log10 on masked values (will be dropped from mean)
# log_W_ordered = jnp.where(mask_real, jnp.log10(W_ordered), jnp.nan)
# log_W_shuf_ordered = jnp.where(mask_shuf, jnp.log10(W_shuf_ordered), jnp.nan)

# # Compute means ignoring nan
# mean_real = jnp.nanmean(log_W_ordered, axis=1)
# mean_shuf = jnp.nanmean(log_W_shuf_ordered, axis=1)
# y_positions = jnp.arange(len(mean_real))

# # Plot bars: real (above), shuffled (below)
# bar_height = 0.35
# ax_bar.barh(y_positions + bar_height/2, mean_real, height=bar_height, color="tab:blue", label="real")
# ax_bar.barh(y_positions - bar_height/2, mean_shuf, height=bar_height, color="tab:orange", label="shuffled")

# ax_bar.set_xlabel("mean tuning")
# ax_bar.tick_params(left=False, labelleft=False)
# ax_bar.set_xticks(-jnp.arange(8))
# ax_bar.set_xticklabels(["0", "", "", "", "", "-5", "", ""])
# ax_bar.grid()
# # ax_bar.legend(loc="lower right", fontsize="x-small")
# ax_bar.invert_xaxis() 

# # Colorbar
# cbar = fig.colorbar(im, cax=ax_cbar)
# cbar.set_label(r"$\log_{10}(W_{ij})$")

# # Labels
# ax_heatmap.set_xlabel("odorants (decreasing frequency)")
# ax_heatmap.set_ylabel("receptors")

# fig.savefig(f"W_adult_fly_heatmap_clip_{CLIP:.0e}.png", dpi=600)
# fig.savefig(f"W_adult_fly_heatmap_clip_{CLIP:.0e}.pdf", dpi=600)
# fig.savefig(f"W_adult_fly_heatmap_clip_{CLIP:.0e}.svg")

# fig = plt.figure(figsize=(6, 5), layout="constrained")
# ax = fig.add_subplot(111)

# ax_heatmap.set_yticks(range(60))
# ax_heatmap.set_yticklabels(["1"] + [""]*58 + ["60"])
# im = W_heatmap_tuning_curves(ax, W_final, config_path)
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(r"$\log_{10}(W_{ij})$")


# fig.savefig(f"W_adult_fly_heatmap_sorted_per_receptor_clip_{CLIP:.0e}.png", dpi=600)
# fig.savefig(f"W_adult_fly_heatmap_sorted_per_receptor_clip_{CLIP:.0e}.pdf", dpi=600)
# fig.savefig(f"W_adult_fly_heatmap_sorted_per_receptor_clip_{CLIP:.0e}.svg")