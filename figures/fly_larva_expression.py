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

DATA_DIRECTORY = os.path.join(RESULTS_DIR, "expression/fly_larva") 
CONFIG_ID = "0" 
key = jax.random.key(int(CONFIG_ID))
MI_CLIP = 0 # this is to make the mutual information scale reasonable--there are large fluctuations in the beginning before the T network has stabilized 

E_init_path = os.path.join(DATA_DIRECTORY, f"E_init_{CONFIG_ID}.npy")
E_final_path = os.path.join(DATA_DIRECTORY, f"E_final_{CONFIG_ID}.npy")   
mutual_information_path = os.path.join(DATA_DIRECTORY, f"mutual_information_{CONFIG_ID}.npy")  
r_init_path = os.path.join(DATA_DIRECTORY, f"r_init_{CONFIG_ID}.npy")
r_final_path = os.path.join(DATA_DIRECTORY, f"r_final_{CONFIG_ID}.npy")   

E_init = jnp.load(E_init_path) 
E_final = jnp.load(E_final_path) 
mi = jnp.load(mutual_information_path)
r_init = jnp.load(r_init_path)
r_final = jnp.load(r_final_path)



mosaic = [["mutual_information", "mutual_information"], ["E_init", "E_final"], ["spectrum", "spectrum"]]
fig, axs = plt.subplot_mosaic(
    mosaic, 
    figsize=(3.3, 3),
    gridspec_kw={"wspace": 0.5, "hspace": 0.5, "height_ratios": [1, 3, 2]}
)

# make panel a: expression and optimization trajectory 
axs["E_init"].set_xlabel("receptors", labelpad=-10)
axs["E_init"].set_ylabel("neurons", labelpad=-4)  
axs["E_final"].set_xlabel("receptors", labelpad=-10)
axs["E_final"].set_ylabel("neurons")

for ax in [axs["E_init"], axs["E_final"]]: 
    ax.set_xticks([0, E_init.shape[1] - 1]) 
    ax.set_yticks([0, E_init.shape[0] - 1])
    ax.set_xticklabels(["1", str(E_init.shape[1])])
    ax.set_yticklabels(["1", str(E_init.shape[0])])

# axs["E_final"].set_xticks([])
axs["E_final"].set_yticks([])

threshold = 0.1 * jnp.max(E_final, axis=1)
indices = plot_utils.sort_rows_by_first_threshold(E_final, threshold)

im1 = axs["E_init"].imshow(E_init[indices, :], aspect="auto", cmap="Blues", interpolation="none")
im2 = axs["E_final"].imshow(E_final[indices, :], aspect="auto", cmap="Blues", interpolation="none")

axs["E_init"].set_aspect("auto")
axs["E_final"].set_aspect("auto")

# axs["E_init"].set_title("initial")
# axs["E_final"].set_title("optimized")

cbar1 = fig.colorbar(im1, ax=axs["E_init"])
cbar2 = fig.colorbar(im2, ax=axs["E_final"])

for cbar, im in zip([cbar1, cbar2], [im1, im2]):
    cbar.set_ticks([im.norm.vmin, im.norm.vmax])  # min and max values
    cbar.set_ticklabels([f"{im.norm.vmin:.2f}", f"{im.norm.vmax:.2f}"]) 

cbar2.set_ticklabels(["0", "1"]) 

# axs["E_init"].set_title("Initial expression")
# axs["E_final"].set_title("Optimized expression") 
axs["mutual_information"].plot(jnp.clip(mi, min=MI_CLIP), color="blue") 
# axs["mutual_information"].set_title(r"Mutual information")
axs["mutual_information"].set_xlabel("epoch", labelpad=-10) 
axs["mutual_information"].set_ylabel(r"$\widehat{MI}(r, c)$", labelpad=-2)
axs["mutual_information"].set_yticks([0, 0.5, 1, 1.5])
axs["mutual_information"].set_xticks([0, 1e6, 2e6])
axs["mutual_information"].set_xticklabels(["0", "", "2e6"])
axs["mutual_information"].set_yticklabels(["0", "", "", "1.5"])
axs["mutual_information"].grid()

# make panel d: 
def get_PC_variances(r):
    cov = jnp.cov(r) 
    evalues, _ = jnp.linalg.eigh(cov) 
    return jnp.real(evalues) / jnp.sum(jnp.real(evalues)) 

r_init_spectrum = get_PC_variances(r_init) 
r_final_spectrum = get_PC_variances(r_final)

baby_blue = plt.cm.Blues(0.5)  # 30% - light baby blue
navy_blue = plt.cm.Blues(1.0)  # 100% - deep navy

# Convert to hex (optional, if you want hex codes)
baby_blue_hex = rgb2hex(baby_blue)
navy_blue_hex = rgb2hex(navy_blue)

axs["spectrum"].scatter(jnp.arange(len(r_init_spectrum)) - 0.2, jnp.sort(r_init_spectrum)[::-1], s=5, zorder=3, label="initial", color=baby_blue_hex)
axs["spectrum"].scatter(jnp.arange(len(r_final_spectrum)) + 0.2, jnp.sort(r_final_spectrum)[::-1], s=5, zorder=3, color=navy_blue_hex, label="optimized")
axs["spectrum"].set_yscale("log")
axs["spectrum"].set_yscale("log") 
axs["spectrum"].grid() 
axs["spectrum"].legend(handlelength=0.3, handletextpad=0.2, 
           borderpad=0.4, labelspacing=0.3, loc="upper right", ncol=2)
axs["spectrum"].set_xlabel("principal component", labelpad=-5)
axs["spectrum"].set_ylabel("fraction variance", labelpad=-10)
axs["spectrum"].set_xticks(jnp.arange(0, 25, 5))
axs["spectrum"].set_xticklabels(["0", "", "", "", "20"])

plt.tight_layout() 

fig.savefig("fly_larva_expression.png", dpi=600) 
fig.savefig("fly_larva_expression.pdf", dpi=600) 
fig.savefig("fly_larva_expression.svg")
