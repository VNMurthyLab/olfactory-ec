import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob 
import sys 
import os 
import re 
import json 

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


DEFAULT_DIRECTORY = "/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/flat_frequencies/environment_sweep/opt_W"
DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY
BLOCK_SIZE = 5

W_init_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "W_init_[0-9]*.npy")),
    key=lambda x: int(re.search(r'W_init_(\d+)\.npy', x).group(1))
)
W_final_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "W_final_[0-9]*.npy")),
    key=lambda x: int(re.search(r'W_final_(\d+)\.npy', x).group(1))
)
E_final_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "E_final_*.npy")),
    key=lambda x: int(re.search(r'E_final_(\d+)\.npy', x).group(1))
)
config_files = sorted(
    glob.glob(os.path.join(DIRECTORY, "config_*.json")),
    key=lambda x: int(re.search(r'config_(\d+)\.json', x).group(1))
)

def extract_sigma_and_blocks(config): 
    sigma = config["hyperparams"]["sigma_c"]
    blocks = os.path.basename(config["data_path"]).strip("samples_block_").strip(".npy")
    return sigma, blocks 

def plot_hist(W_init, W_final, ax, config, threshold=None): 
    W_init = jnp.log10(W_init) 
    if threshold: 
        W_final = W_final[W_final > threshold] 
    W_final = jnp.log10(W_final) 
    ax.hist(W_init.flatten(), alpha=0.6, density=True) 
    ax.hist(W_final.flatten(), alpha=0.6, density=True) 
    ax.set_yscale("log")
    sigma, blocks = extract_sigma_and_blocks(config) 
    ax.set_title(fr"$\sigma_c = {sigma}$, blocks = {blocks}")
    return ax 

def fit_normal(W, threshold=None):
    if threshold: 
        W = W[W > threshold] 
    log_W = jnp.log10(W)
    log_vals = jnp.ravel(log_W)
    mean = jnp.mean(log_vals)
    std = jnp.std(log_vals)
    def normal_pdf(log_x):
        return (1 / (std * jnp.sqrt(2 * jnp.pi))) * jnp.exp(-((log_x - mean) ** 2) / (2 * std ** 2))
    return normal_pdf, mean, std 

def generate_tunable_W(key, W_final, threshold, covariance_generator, rho=0.0, mean=0.0, std=1.0):
    n_receptors, n_odorants = W_final.shape[0], W_final.shape[1] 
    key, mask_key = jax.random.split(key) 
    cov = covariance_generator(n_receptors, rho) # rho = 0.0 gives you the identity. 
    if jnp.ndim(std) == 0:
        std = jnp.full((n_receptors,), std)
    if jnp.ndim(mean) == 0: 
        mean = jnp.full((n_receptors,), mean) 
    # Apply std^2 to the covariance
    scaled_cov = cov * std[:, None] * std[None, :]
    L = jnp.linalg.cholesky(scaled_cov + 1e-6 * jnp.eye(n_receptors))  # stabilize
    Z = jax.random.normal(key, (n_receptors, n_odorants)) 
    logW = L @ Z + mean.reshape(n_receptors, 1)
    mask = W_final > threshold 
    W_analytic = 10**logW 
    W_analytic = jnp.where(W_analytic > jnp.max(W_final), 0, W_analytic) # this is important. Otherwise, the max value of W_analytic can be 6 OOM greater than the max of W_final. This is because log normals are not perfect fits to the long heavy negative tail of W_final
    W_analytic = W_analytic * mask 
    return W_analytic 

def get_means_and_stds(W, threshold=None): 
    means, stds = [], [] # the thresholding is why you can't do this vectorized. You need to filter not clip the values below the threshold. 
    for i in range(W.shape[0]): 
        receptor = W[i] 
        if threshold:
            receptor = receptor[receptor > threshold]  
        log_vals = jnp.log10(receptor) 
        mean, std = jnp.mean(log_vals), jnp.std(log_vals)
        means.append(mean) 
        stds.append(std) 
    return jnp.array(means), jnp.array(stds) 

def get_mean_and_std(W, threshold=None): 
    if threshold:
        W = W[W > threshold]
    log_vals = jnp.log10(W) 
    mean, std = jnp.mean(log_vals), jnp.std(log_vals)
    return mean, std 

def make_block_covariance(n=60, rho=1.0):
    assert n == 60, "This function is configured for 6 blocks of size 10 (i.e., n = 60)."
    block_size = BLOCK_SIZE # Or define BLOCK_SIZE at the top
    num_blocks = n // block_size
    blocks = []
    for _ in range(num_blocks):
        block = jnp.full((block_size, block_size), rho)
        block = block.at[jnp.diag_indices(block_size)].set(1.0)
        blocks.append(block)
    cov = jnp.block([
        [blocks[i] if i == j else jnp.zeros((block_size, block_size))
         for j in range(num_blocks)]
        for i in range(num_blocks)
    ])
    return cov


key = jax.random.key(0) 

for i in range(len(W_init_files)): 
    with open(config_files[i], "r") as c: 
        config = json.load(c)
    output_dir = config["logging"]["output_dir"]
    key, subkey_sample, subkey_shuffle = jax.random.split(key, 3)
    W_init = jnp.load(W_init_files[i])
    W_final = jnp.load(W_final_files[i])
    threshold = jnp.mean(W_init) # this is the effective threshold below which W doesn't matter, so we don't want to fit to these. 
    log_xs = jnp.linspace(jnp.log10(threshold), jnp.max(jnp.log10(W_final)))
    pdf, mean, std = fit_normal(W_final.flatten(), threshold)
    means, stds = get_mean_and_std(W_final, threshold) 
    W_analytic = generate_tunable_W(subkey_sample, W_final, threshold, make_block_covariance, 0.0, mean, std)
    shuffle_indices = jax.random.permutation(subkey_shuffle, len(W_final.ravel())) 
    W_shuffle = W_final.ravel()[shuffle_indices].reshape(W_final.shape)
    id_ = os.path.basename(W_final_files[i]).strip("W_final_").strip(".npy")
    W_analytic_path = os.path.join(output_dir, f"W_final_analytic_fit_{id_}.npy")
    jnp.save(W_analytic_path, W_analytic)
    W_shuffle_path = os.path.join(output_dir, f"W_final_shuffle_{id_}.npy") 
    jnp.save(W_shuffle_path, W_shuffle) 

fig, axs = plt.subplots(1, 2, 
                        figsize=(3.4, 2), 
                        layout="constrained") 
plot_hist(W_init, W_final, axs[0], config, threshold)
plot_hist(W_shuffle[W_shuffle > threshold], W_analytic[W_analytic > threshold], axs[1], config, threshold)
log_xs = jnp.linspace(jnp.log10(threshold), jnp.max(jnp.log10(W_final)))
axs[0].plot(log_xs, pdf(log_xs), label="log normal fit")
axs[1].plot(log_xs, pdf(log_xs), label="log normal fit")
axs[0].set_title("init, final")
axs[1].set_title("shuffle, analytic")
fig.savefig("example_shuffle_and_analytic_W_fit.png", dpi=600) 

