import numpy as np


def reduce_by_id(vals, ids, K=None, mean=False, return_counts=False):
    """
    Group-reduce along axis 0 by dense integer ids.

    vals: array, shape (N, ...)
    ids:  array, shape (N,), dtype int, values in [0, K-1]
    K:    optional int; default is ids.max()+1
    mean: if True, return per-id mean; else per-id sum
    return_counts: if True, also return counts per id

    Returns:
      out:  shape (K, ...) with sums or means
      (optional) counts: shape (K,)
    """
    vals = np.asarray(vals)
    ids  = np.asarray(ids)
    if vals.shape[0] != ids.shape[0]:
        raise ValueError("vals.shape[0] must equal ids.shape[0].")
    if K is None:
        if ids.size == 0:
            raise ValueError("Empty ids with K=None.")
        K = int(ids.max()) + 1
    if ids.min(initial=0) < 0 or ids.max(initial=-1) >= K:
        raise ValueError("ids must be in [0, K-1].")

    out_shape = (K,) + vals.shape[1:]
    out = np.zeros(out_shape, dtype=(float if mean else vals.dtype))

    # Sum per id (works for any trailing dims)
    np.add.at(out, ids, vals)

    if not mean:
        return (out, np.bincount(ids, minlength=K)) if return_counts else out

    # Divide by counts to get means
    counts = np.bincount(ids, minlength=K)
    denom = counts.reshape((K,) + (1,) * (out.ndim - 1))
    # Safe in-place division; leaves zeros where denom==0 (filled with NaN next)
    np.divide(out, denom, out=out, where=denom != 0)

    # For ids with zero count, set mean to NaN (broadcast across trailing dims)
    if (counts == 0).any():
        out[counts == 0, ...] = np.nan

    return (out, counts) if return_counts else out

def assign_lattice_positions(n_particles: int, box_size: np.ndarray) -> np.ndarray:
    # Handle edge case of zero particles
    if n_particles == 0:
        return np.empty((0, 2), dtype=np.float64)
    
    # Determine optimal grid dimensions for square lattice
    # Try to make grid as square as possible
    grid_x = int(np.ceil(np.sqrt(n_particles)))
    grid_y = int(np.ceil(n_particles / grid_x))
    
    # Calculate spacing between lattice points
    spacing_x = box_size[0] / grid_x
    spacing_y = box_size[1] / grid_y

    # Pre-allocate position array
    pos = np.empty((n_particles, 2), dtype=np.float64)
    
    # Generate lattice positions row by row
    particle_count = 0
    for iy in range(grid_y):
        for ix in range(grid_x):
            if particle_count >= n_particles:
                break
            
            # Position particles at center of lattice cells
            # This ensures even spacing and centering within the box
            x = (ix + 0.5) * spacing_x
            y = (iy + 0.5) * spacing_y
            
            pos[particle_count, 0] = x
            pos[particle_count, 1] = y
            particle_count += 1
            
        if particle_count >= n_particles:
            break
    
    return pos