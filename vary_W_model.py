import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt
import glob 
import sys 
import os 
import re 
import json 
from scipy.linalg import toeplitz

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

BLOCK_SIZE = 6
DEFAULT_DIRECTORY = "/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/environment_sweep/opt_W"
DIRECTORY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY
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

def make_toeplitz_covariance(n, rho): # this doesn't lead to noncanonical expression! great, include it. 
    # rho = correlation between receptors (rows of log W)
    return toeplitz(rho ** jnp.arange(n))

def make_block_covariance(n=60, rho=1.0):
    assert n == 60, "This function is configured for 12 blocks of size 5 (i.e., n = 60)."
    block_size = BLOCK_SIZE
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

def generate_tunable_W(key, W_final, threshold, covariance_generator, rho=0.0, mean=0.0, std=1.0, mask_bool=True):
    n_receptors, n_odorants = W_final.shape[0], W_final.shape[1] 
    key, mask_key = jax.random.split(key) 
    cov = covariance_generator(n_receptors, rho)
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
    if mask_bool: 
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

def make_noncanonical_E(key, L, M, lower=1, upper=5):
    E_rows = []
    keys = jax.random.split(key, L)

    for i in range(L):
        subkey1, subkey2, subkey3 = jax.random.split(keys[i], 3)
        k = jax.random.randint(subkey1, shape=(), minval=lower, maxval=upper + 1)
        perm = jax.random.permutation(subkey2, M)
        indices = perm[:k]
        values = jax.random.uniform(subkey3, shape=(k,))
        row = jnp.zeros(M)
        row = row.at[indices].set(values)
        row /= jnp.sum(row) 
        E_rows.append(row)

    E = jnp.stack(E_rows)
    return E

def make_descent_E(key, L, M, lower=1, upper=5, block_size=BLOCK_SIZE):
    assert M % block_size == 0, "M must be divisible by block_size"
    num_blocks = M // block_size

    E_rows = []
    keys = jax.random.split(key, L)

    for i in range(L):
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(keys[i], 4)

        # Pick one block index
        block_idx = jax.random.randint(subkey1, shape=(), minval=0, maxval=num_blocks)
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # Pick how many receptors to coexpress within this block
        k = jax.random.randint(subkey2, shape=(), minval=lower, maxval=upper + 1)

        # Choose k indices within this block
        perm = jax.random.permutation(subkey3, block_size)
        block_indices = perm[:k] + block_start

        # Generate expression values
        values = jax.random.uniform(subkey4, shape=(k,))
        row = jnp.zeros(M)
        row = row.at[block_indices].set(values)
        row /= jnp.sum(row)

        E_rows.append(row)

    E = jnp.stack(E_rows)
    return E

def make_random_E(key, L, M):
    phi_E = lambda u: jax.nn.softmax(
        u, axis=1
    )
    E = phi_E(
            0.5 + 0.1 * jax.random.normal(key, shape=(L, M))
        )  # this is a random initialization where every neuron expresses roughly the same amount of every receptor.
    return E 

def make_E(key, L, M, E_type, lower=1, upper=6):
    if E_type == "canonical": 
        E = jnp.repeat(
            jnp.eye(M), L // M, axis=0
            ).astype(float)
    elif E_type == "descent": 
        E = make_descent_E(key, L, M, lower, upper) 
    elif E_type == "co_option": 
        E = make_noncanonical_E(key, L, M, lower, upper) 
    elif E_type == "random": 
        E = make_random_E(key, L, M)
    else: 
        print("E_type needs to be one of {canonical, descent, co_option}")
    return E 

def write_Es(key, output_dir, config): 
    key, E_can, E_non, E_descent = jax.random.split(key, 4) 
    E_canonical = make_E(E_can, config["hyperparams"]["L"], config["hyperparams"]["M"], "canonical") 
    E_canonical_path = os.path.join(output_dir, "E_canonical.npy") 
    jnp.save(E_canonical_path, E_canonical)
    E_descent = make_E(E_non, config["hyperparams"]["L"], config["hyperparams"]["M"], "descent") # this is noncanonical, but it respects the block structure! 
    E_descent_path = os.path.join(output_dir, "E_descent.npy") 
    jnp.save(E_descent_path, E_descent) 
    E_co_option = make_E(E_non, config["hyperparams"]["L"], config["hyperparams"]["M"], "co_option") 
    E_co_option_path = os.path.join(output_dir, "E_co_option.npy") 
    jnp.save(E_co_option_path, E_co_option)
    return E_canonical_path, E_descent_path, E_co_option_path 


def write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_noncanonical_path): 
    config["hyperparams"]["E_path"] = E_canonical_path 
    config_path = os.path.join(config_dir, f"config_{config_counter}.json") 
    with open(config_path, "w") as f: 
        json.dump(config, f, indent=2)

    config["hyperparams"]["E_path"] = E_noncanonical_path
    config_path = os.path.join(config_dir, f"config_{config_counter + 1}.json") 
    with open(config_path, "w") as f: 
        json.dump(config, f, indent=2)

    return config_counter + 2

key = jax.random.key(0) 

i = 17
with open(config_files[i], "r") as c: 
    config = json.load(c)
output_dir = "/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/vary_W_model/"
config_dir = "/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/configs/vary_W_model/"
config["hyperparams"]["W_init"] = "read in"
config["hyperparams"]["read_in_E"] = True 
config["hyperparams"]["canonical_E_init"] = False 
config["hyperparams"]["balanced_E_init"] = False 
config["logging"]["output_dir"] = output_dir
config["training"]["gamma_W"] = 0.0
config["training"]["gamma_E"] = 0.1
config["training"]["epochs_per_scan"] = 1000000
W_init = jnp.load(W_init_files[i])
W_final = jnp.load(W_final_files[i])
threshold = jnp.mean(W_init) # we take the MLE std deviation, but we need to filter W_final before doing so, or else it's dominated by junk values below this threshold and even below machine precision. Look carefully at fits! 
means, stds = get_means_and_stds(W_final, threshold)
# write your Es: canonical, noncanonical, and descent
key, subkey = jax.random.split(key)
E_canonical_path, E_descent_path, E_co_option_path = write_Es(subkey, output_dir, config) 

# W_opt 
config_counter = 0 
W_opt_path = os.path.join(output_dir, "W_opt.npy") 
jnp.save(W_opt_path, W_final) 
config["hyperparams"]["W_path"] = W_opt_path 
config_counter = write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_co_option_path)

# W_shuffle
key, subkey = jax.random.split(key)
shuffle_indices = jax.random.permutation(subkey, len(W_final.ravel())) 
W_shuffle = W_final.ravel()[shuffle_indices].reshape(W_final.shape)
W_shuffle_path = os.path.join(output_dir, "W_shuffle.npy") 
jnp.save(W_shuffle_path, W_shuffle) 
config["hyperparams"]["W_path"] = W_shuffle_path 
config_counter = write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_co_option_path)

# W_shuffle_rows 
key, subkey = jax.random.split(key)
shuffled_rows = jax.vmap(lambda r, k: r[jax.random.permutation(k, len(r))])(W_final, jax.random.split(subkey, W_final.shape[0]))
W_shuffle_rows = shuffled_rows.reshape(W_final.shape)
W_shuffle_rows_path = os.path.join(output_dir, "W_shuffle_rows.npy") 
jnp.save(W_shuffle_rows_path, W_shuffle_rows) 
config["hyperparams"]["W_path"] = W_shuffle_rows_path 
config_counter = write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_co_option_path)

# W analytic fit 
key, subkey = jax.random.split(key)
W_analytic = generate_tunable_W(subkey, W_final, threshold, make_block_covariance, 0.0, means, stds) # this has no covariance structure--just a fit to the mean and variance of P(W_{ij}) 
W_analytic_path = os.path.join(output_dir, "W_fit.npy") 
jnp.save(W_analytic_path, W_analytic) 
config["hyperparams"]["W_path"] = W_analytic_path 
config_counter = write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_co_option_path)


# W analytic fit block covariance 
key, subkey = jax.random.split(key)
W_analytic_block = generate_tunable_W(subkey, W_final, threshold, make_block_covariance, 0.8, means, stds, mask_bool=False) # this has strong block covariance structure
W_analytic_block_path = os.path.join(output_dir, "W_fit_block.npy") 
jnp.save(W_analytic_block_path, W_analytic_block) 
config["hyperparams"]["W_path"] = W_analytic_block_path 
config_counter = write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_descent_path)

# W analytic fit toeplitz covariance 
key, subkey = jax.random.split(key)
W_analytic_toeplitz = generate_tunable_W(subkey, W_final, threshold, make_toeplitz_covariance, 0.8, means, stds, mask_bool=False) # this has strong toeplitz covariance structure
W_analytic_toeplitz_path = os.path.join(output_dir, "W_fit_toeplitz.npy") 
jnp.save(W_analytic_toeplitz_path, W_analytic_toeplitz) 
config["hyperparams"]["W_path"] = W_analytic_toeplitz_path 
config_counter = write_canonical_and_noncanonical(config, config_dir, config_counter, E_canonical_path, E_descent_path)

# List all your W variants and their names
W_variants = [
    (W_final, "W_opt"),
    (W_shuffle, "W_shuffle"),
    (W_shuffle_rows, "W_shuffle_rows"),
    (W_analytic, "W_fit"),
    (W_analytic_block, "W_fit_block"),
    (W_analytic_toeplitz, "W_fit_toeplitz")
]

# Create figure
fig, axs = plt.subplots(2, 6, figsize=(6, 1.5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

for i, (W, name) in enumerate(W_variants):
    # Convert to numpy for plotting if needed
    W_np = jnp.array(W)
    
    # Top row: log10 distribution
    axs[0,i].hist(jnp.log10(jnp.abs(W_np[W_np != 0]) + 1e-20).ravel(), bins=50)
    axs[0,i].set_xticklabels([])
    axs[0,i].set_yticklabels([])
    # Bottom row: covariance matrix
    cov = jnp.cov(jnp.log10(jnp.clip(W_np, min=1e-16)))  # Compute covariance
    im = axs[1,i].imshow(cov, cmap='Greens', aspect="auto", interpolation="none")
    axs[1, i].set_xticklabels([])
    axs[1, i].set_yticklabels([])
    plt.colorbar(im, ax=axs[1,i], fraction=0.046, pad=0.04)

plt.tight_layout()
fig.savefig("vary_W_model_distributions.png", dpi=600)