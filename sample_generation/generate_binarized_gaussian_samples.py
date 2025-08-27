import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import argparse 
import json 
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib.ticker as mtick

def format_sci(value):
    """Formats a number in scientific notation as x × 10ⁿ using LaTeX."""
    if value == 0:
        return "0"
    exponent = int(jnp.floor(jnp.log10(jnp.abs(value))))
    base = value / (10**exponent)
    return r"{:.2f} \times 10^{{{}}}".format(base, exponent)


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



def make_block_diagonal_cov(J_vals, N):
    J = len(J_vals)
    if J == 0:
        return jnp.zeros((N, N))
    
    K = N // J
    remainder = N % J
    block_sizes = [K + 1 if i < remainder else K for i in range(J)]
    
    mat = jnp.zeros((N, N))
    pos = 0 
    
    for j in range(J):
        block_size = block_sizes[j]
        block = (1 - J_vals[j]) * jnp.eye(block_size) + J_vals[j]
        end_pos = pos + block_size
        mat = mat.at[pos:end_pos, pos:end_pos].set(block)
        pos = end_pos
    
    return mat

def gaussian_copula_binary(key, means, cov, P):
    n = len(means)
    cov = cov + 1e-6 * jnp.eye(cov.shape[0])
    Z = jax.random.multivariate_normal(key, mean=jnp.zeros(n), cov=cov, shape=(P, ))
    thresholds = norm.ppf(1 - means)  # Inverse CDF
    X = (Z > thresholds).astype(int)
    return X.T, Z.T

def manually_thin_frequent_odorants(key, X, max_freq=0.05): 
    frequencies = jnp.mean(X, axis=1)
    desired_probabilities = jnp.clip(max_freq / frequencies, max=1.0)
    mask = jax.random.bernoulli(key, shape=X.shape, p=desired_probabilities.reshape(-1, 1))
    return jnp.where(mask, X, 0)

if __name__ == "__main__": 
    jax.config.update("jax_default_matmul_precision", "high")
    print(jax.default_backend())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        params = json.load(f)

    config_id = args.config.split("/")[1].strip("config").strip(".json")
    
    # Initialize random keys
    key = jax.random.key(params["seed"])
    key, subkey_mean, subkey_samples, subkey_thin = jax.random.split(key, 4)
    
    # Generate h vector
    # Means calculation
    if params["balanced_means"]:
        means = jnp.full(params["N"], params["mean_parameter"])
    else:
        means = jnp.clip(
            jax.random.gamma(subkey_mean, a=params["gamma_shape"], shape=(params["N"],)) * params["gamma_scale"] + params["gamma_loc"],
            min=0,
            max=0.9
        )

    # Generate lambdas and J matrix
    lambdas = params["lambda_value"] * jnp.ones((params["block_number"],))
    J = make_block_diagonal_cov(lambdas, params["N"])
    
    sample_list = []
    total_samples = 0
    
    while total_samples < params["samples"]:
        key, subkey_samples = jax.random.split(key)
        key, subkey_thin = jax.random.split(key) 
        P = min(params["samples"], 100000)
        X, _ = gaussian_copula_binary(subkey_samples, means, J, P=P)
        X = manually_thin_frequent_odorants(
            subkey_thin,
            X,
            max_freq=params["thinning_parameter"]
        )
        
        X = X[:, jnp.sum(X, axis=0) > 0]
        sample_list.append(X)
        total_samples += X.shape[1]
        print(f"Generated {total_samples} samples")
    
    X = jnp.concatenate(sample_list, axis=1)[:, :params["samples"]]

    fig, axs = plt.subplots(1, 3, figsize=(7, 2), layout="constrained")
    frequencies = jnp.mean(X, axis=1)
    axs[0].scatter(range(len(frequencies)), jnp.sort(frequencies))
    axs[0].set_title("Odorant Frequencies")
    axs[0].set_yscale("log")
    axs[0].grid()

    # Add μ and σ text for frequencies
    mean_freq = jnp.mean(frequencies)
    std_freq = jnp.std(frequencies)
    axs[0].text(0.02, 0.98,
        r'$\mu = {}$'.format(format_sci(mean_freq)) + '\n' + r'$\sigma = {}$'.format(format_sci(std_freq)),
        transform=axs[0].transAxes,
        ha='left', va='top',
        fontsize=10,
        bbox=dict(facecolor='none', alpha=0.0, edgecolor='none'))

    # Correlation matrix
    im = axs[1].imshow(jnp.nan_to_num(jnp.corrcoef(X), 0), cmap="Greens")
    axs[1].set_title('Correlation Matrix')
    plt.colorbar(im, ax=axs[1])

    components = jnp.sum(X, axis=0)
    axs[2].scatter(range(len(components)), jnp.sort(components))
    axs[2].set_title("Components per Mixture")
    axs[2].set_yscale("log")

    # Add μ and σ text for components
    mean_comp = jnp.mean(components)
    std_comp = jnp.std(components)
    axs[2].text(0.02, 0.98,
        r'$\mu = {}$'.format(format_sci(mean_comp)) + '\n' + r'$\sigma = {}$'.format(format_sci(std_comp)),
        transform=axs[2].transAxes,
        ha='left', va='top',
        fontsize=10,
        bbox=dict(facecolor='none', alpha=0.0, edgecolor='none'))
    axs[2].grid() 

    print(f'{params["output_dir"]}/summary_stats_block{config_id}.png')
    fig.savefig(f'{params["output_dir"]}/summary_stats_block{config_id}.png', dpi=600)
    fig.savefig(f'{params["output_dir"]}/summary_stats_block{config_id}.pdf', dpi=600)
    fig.savefig(f'{params["output_dir"]}/summary_stats_block{config_id}.svg')
    # jnp.save(f"{params["output_dir"]}/samples_block{config_id}", X)
    array_cpu = jax.device_get(X)  # Transfers to CPU
    np.save(f'{params["output_dir"]}/samples_block{config_id}.npy', array_cpu)

