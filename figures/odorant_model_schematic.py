import jax.numpy as jnp 
import matplotlib.pyplot as plt 
import os 

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

DATA_DIR = os.environ['DATA_DIR'] 
corr1_path = os.path.join(DATA_DIR, "data_for_optimization/samples_block_8.npy") 
corr2_path = os.path.join(DATA_DIR, "data_for_optimization/samples_block_128.npy") 

samples1 = jnp.load(corr1_path)
# samples2 = jnp.load(corr2_path) 

fig, axs = plt.subplots(
    1, 2,
    figsize=(3.4, 1), 
    gridspec_kw={"wspace": 0.2, "hspace": 0.0, "width_ratios": [1, 1]}, 
    layout="constrained"
    )

frequencies = jnp.mean(samples1, axis=1) 

corr1 = jnp.nan_to_num(jnp.corrcoef(samples1), 0)  
# corr2 = jnp.nan_to_num(jnp.corrcoef(samples2), 0)  

im1 = axs[0].plot(range(len(frequencies)), jnp.sort(frequencies))
axs[0].set_yscale("log")
axs[0].set_xticks([1, 500, 1000])
axs[0].set_xticklabels(["1", "", "1000"])
axs[0].set_xlabel("odorants", labelpad=-10)
axs[0].set_ylabel("frequency")
axs[0].grid() 

def plot_corr(ax, corr): 
    im = ax.imshow(corr, cmap="Greens", rasterized=False, interpolation="none")
    ax.set_xticks([]) 
    ax.set_yticks([])
    return im 

im1 = plot_corr(axs[1], corr1)
axs[1].set_xlabel("odorants") 
axs[1].set_ylabel("odorants")

# Add colorbars with same parameters to both
cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
cbar1.set_label(r"$\Sigma_{ij}$", rotation=0, labelpad=15)

# Make the axes the same size by adjusting their positions
# axs["corr_1"].set_position(axs["corr_2"].get_position())
# cbar2.set_ticks([im2.norm.vmin, im2.norm.vmax])  # min and max values
# cbar2.set_ticklabels([f"{im2.norm.vmin:.1f}", f"{im2.norm.vmax:.1f}"]) 


# fig.text(
#     0.65,              # x position (adjust to move left/right)
#     0.6,               # y position (centered vertically between both rows)
#     r"$\Sigma$ =",     # LaTeX string
#     va='center',       # vertical alignment
#     ha='center'       # horizontal alignment
# )
fig.savefig("odorant_model_schematic.png", dpi=600) 
fig.savefig("odorant_model_schematic.pdf", dpi=600) 
fig.savefig("odorant_model_schematic.svg") 

fig, ax = plt.subplots(figsize=(1, 0.8))

x = jnp.linspace(-1.5, 1.5, 100)

ax.plot(x, jnp.tanh(3*x))

fig.savefig("tanh.svg") 
