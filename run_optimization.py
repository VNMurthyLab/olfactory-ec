import argparse
import json
import jax
import jax.numpy as jnp 
from model import (
    FullConfig,
    HyperParams,
    TrainingConfig,
    LoggingConfig,
    initialize_p,
    initialize_training_state,
    make_constant_gammas,
    closure_draw_cs_data_driven,
    train_natural_gradient_scan_over_epochs,
    linear_filter_plus_glomerular_layer
)
import model 
from plotting import (
    plot_E,
    plot_G, 
    plot_W, 
    plot_r,
    plot_kappa_inv,
    plot_eta
)
import shutil
import matplotlib.pyplot as plt 
import sys 
import jax.profiler
import os 
import re



print(jax.default_backend())
jax.config.update("jax_default_matmul_precision", "high") 

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=False)
args = parser.parse_args()

def resolve_config_paths(config_dict):
    """Replace {{PLACEHOLDERS}} with environment variables"""
    resolved = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config_paths(value)
        elif isinstance(value, str):
            def replace_match(match):
                var_name = match.group(1)
                if var_name not in os.environ:
                    raise ValueError(f"Environment variable {var_name} must be set")
                return os.environ[var_name]
            resolved[key] = re.sub(r'{{(\w+)}}', replace_match, value)
        else:
            resolved[key] = value
    return resolved

def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        cfg_dict = resolve_config_paths(json.load(f))
    return FullConfig(
        hyperparams=HyperParams(**cfg_dict["hyperparams"]),
        training=TrainingConfig(**cfg_dict["training"]),
        logging=LoggingConfig(**cfg_dict["logging"]),
        seed=cfg_dict["seed"], 
        data_path=cfg_dict["data_path"]
    )

def compute_dynamic_tol(hp, denom=50): 
    sigma_c = hp.sigma_c
    mu_c = hp.mu_c
    return 1 / (denom * jnp.exp(mu_c + 3 * sigma_c)) 

config = load_config(args.config)

config_id = args.config.split("/")[-1].strip("config_").strip(".json")
key = jax.random.key(config.seed)
key, *subkeys = jax.random.split(key, 20)
hp = config.hyperparams

# this is a patch because we don't want to load all 10 data-driven odor models, each of which is 10^3 x 10^6. Besides this, it's nice to have all this logic in model.py
if hp.odor_model == "block covariance binary": 
    model.ODOR_MODEL_REGISTRY = {
        "block covariance binary": closure_draw_cs_data_driven(config.data_path)[0]
    }
elif hp.odor_model == "block covariance log normal": 
    model.ODOR_MODEL_REGISTRY = {
        "block covariance log normal": closure_draw_cs_data_driven(config.data_path)[1]
    }

draw_cs = model.ODOR_MODEL_REGISTRY[hp.odor_model]
activity_function = model.ACTIVITY_FUNCTION_REGISTRY[hp.activity_model]
cs = draw_cs(subkeys[0], hp)
norms = jnp.linalg.norm(cs, axis=0)
tol = compute_dynamic_tol(hp)

hp, p_init = initialize_p(subkeys[1], mean_norm_c=jnp.mean(norms), threshold=tol, hp=hp)

init_state = initialize_training_state(subkeys[2], hp, p_init, config.training)

t = config.training
# gammas = make_constant_gammas(
#     t.scans, t.epochs_per_scan, gamma_W=t.gamma_W, gamma_E=t.gamma_E, gamma_G=t.gamma_G, gamma_kappa_inv=t.gamma_kappa_inv, gamma_eta=t.gamma_eta, gamma_T=t.gamma_T, 
# )

gammas = jnp.array([t.gamma_W, t.gamma_E, t.gamma_G, t.gamma_gain, t.gamma_kappa_inv, t.gamma_eta, t.gamma_T])

def compute_activity(key, hp, p, activity_function): 
    key, *subkeys = jax.random.split(key, 5)
    draw_cs = model.ODOR_MODEL_REGISTRY[hp.odor_model]
    cs = draw_cs(subkeys[0], hp)
    r = activity_function(hp, p, cs, subkeys[1])
    return r 

activity_function = model.ACTIVITY_FUNCTION_REGISTRY["linear filter"]

r_init = compute_activity(subkeys[3], hp, init_state.p, activity_function) 

state, metrics = train_natural_gradient_scan_over_epochs(
    init_state, hp, gammas, t.scans, t.epochs_per_scan
)

fig, axs = plot_E(init_state.p.E, state.p.E, metrics["mi"], hp, t, key=subkeys[7], log_scale=False)
figtitle = f"{hp.activity_model} activity, {hp.odor_model} odor model"
if "block covariance" in hp.odor_model: 
    k = config.data_path.split("_")[-1].split(".")[0]
    figtitle += f" (blocks = {k})"
fig.suptitle(figtitle)
fig.savefig(f"{config.logging.output_dir}/expression_{config_id}.png", bbox_inches="tight", dpi=300)

fig, axs = plot_G(init_state.p.G, state.p.G, metrics["mi"], hp, t) 
fig.savefig(f"{config.logging.output_dir}/G_{config_id}.png", bbox_inches="tight", dpi=300)

fig, axs = plot_W(init_state.p.W, state.p.W, metrics["mi"], hp, t, key=subkeys[8]) 
fig.savefig(f"{config.logging.output_dir}/W_{config_id}.png", bbox_inches="tight", dpi=300)

fig, axs = plot_kappa_inv(jnp.log(init_state.p.kappa_inv), jnp.log(state.p.kappa_inv), metrics["mi"], hp, t, key=subkeys[9]) 
fig.savefig(f"{config.logging.output_dir}/kappa_inv_{config_id}.png", bbox_inches="tight", dpi=300)

fig, axs = plot_eta(jnp.log(init_state.p.eta), jnp.log(state.p.eta), metrics["mi"], hp, t, key=subkeys[10]) 
fig.savefig(f"{config.logging.output_dir}/eta_{config_id}.png", bbox_inches="tight", dpi=300)

r_final = compute_activity(subkeys[3], hp, state.p, activity_function) 

fig, axs = plot_r(r_init, r_final, metrics["mi"], hp, init_state.p, state.p, t, activity_function, mi_clip=-1, key=subkeys[9])
fig.savefig(f"{config.logging.output_dir}/r_{config_id}.png", bbox_inches="tight", dpi=300)

jnp.save(f"{config.logging.output_dir}/mutual_information_{config_id}", metrics["mi"])
jnp.save(f"{config.logging.output_dir}/W_init_{config_id}", init_state.p.W)
jnp.save(f"{config.logging.output_dir}/W_final_{config_id}", state.p.W)
jnp.save(f"{config.logging.output_dir}/E_init_{config_id}", init_state.p.E)
jnp.save(f"{config.logging.output_dir}/E_final_{config_id}", state.p.E)
jnp.save(f"{config.logging.output_dir}/r_init_{config_id}", r_init)
jnp.save(f"{config.logging.output_dir}/r_final_{config_id}", r_final)
jnp.save(f"{config.logging.output_dir}/kappa_inv_init_{config_id}", init_state.p.kappa_inv)
jnp.save(f"{config.logging.output_dir}/eta_init_{config_id}", init_state.p.eta)
jnp.save(f"{config.logging.output_dir}/kappa_inv_{config_id}", state.p.kappa_inv)
jnp.save(f"{config.logging.output_dir}/eta_{config_id}", state.p.eta)
jnp.save(f"{config.logging.output_dir}/G_init_{config_id}", init_state.p.G)
jnp.save(f"{config.logging.output_dir}/G_final_{config_id}", state.p.G)
jnp.save(f"{config.logging.output_dir}/gain_final_{config_id}", state.p.gain)

# with open(f"{config.logging.output_dir}/config_{config_id}.json", "w") as c: 
#     json.dump(config, c)

shutil.copy2(args.config, f"{config.logging.output_dir}/config_{config_id}.json")