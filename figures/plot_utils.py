import jax.numpy as jnp
import matplotlib.pyplot as plt 


def sort_rows_by_first_threshold(matrix, threshold):
    first_above_thresh = jnp.argmax(matrix > threshold[:, None], axis=1)
    all_below_thresh = ~jnp.any(matrix > threshold[:, None], axis=1)
    first_above_thresh = first_above_thresh.at[all_below_thresh].set(matrix.shape[1])  # Assign a large column index
    sorted_row_indices = jnp.argsort(first_above_thresh)
    return sorted_row_indices 