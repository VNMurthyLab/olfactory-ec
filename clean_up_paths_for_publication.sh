# Replace ALL absolute paths with simple placeholders
find sample_generation/configs_flat_frequencies/ -name "*.json" -exec sed -i '' \
  -e 's|/n/home10/jfernandezdelcasti/noncanonical-olfaction/model/results/|{{RESULTS_DIR}}/|g' \
  -e 's|/n/holylabs/LABS/murthy_users/Lab/juancarlos/efficient_coding_olfaction/|{{DATA_DIR}}/|g' \
  {} \;