#!/bin/bash
# File: transform_jsons_dynamic.sh
# Usage: ./transform_jsons_dynamic.sh

TARGET_DIR="./read_in_E"

process_file() {
    local file="$1"
    local config_num=$(basename "$file" | grep -oE '[0-9]+')  # Extract number from filename
    
    echo "Processing: $file (Config $config_num)"
    
    # Base path for E files
    local base_path="/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/environment_sweep/opt_E_given_shuffle_W/noncanonical_init"
    local new_E_path="${base_path}/E_final_${config_num}.npy"
    
    jq --arg new_E_path "$new_E_path" '
    .hyperparams.activity_model = "glomerular convergence" |
    .hyperparams.balanced_E_init = false |
    .hyperparams.canonical_E_init = false |
    .hyperparams.noncanonical_E_init = false |
    .hyperparams.read_in_E = true | 
    .hyperparams.E_path = $new_E_path |
    .hyperparams.canonical_G_init = false |
    .training.gamma_E = 0.0 |
    .training.gamma_G = 0.1 |
    .training.gamma_gain = 0.1 |
    .logging.output_dir = "/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/glomerular_convergence/read_in_E"
    ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
}

# Main execution
find "$TARGET_DIR" -type f -name "config_*.json" -print0 | while IFS= read -r -d '' file; do
    process_file "$file"
done

echo "Transformation complete"