"""
Demonstration of run_binned_ragged for handling variable-size kernel outputs.

This example shows how to use run_binned_ragged when your kernel returns
arrays of varying sizes that cannot be averaged together (e.g., neighbor lists,
cluster assignments, etc.).
"""

import numpy as np
from pydpmd.calc import run_binned_ragged, TimeBins, requires_fields

# Create some simple test data: 10 particles moving in 2D over 100 time steps
np.random.seed(42)
n_particles = 10
n_frames = 100

# Generate random walk trajectory
positions = np.cumsum(np.random.randn(n_frames, n_particles, 2) * 0.1, axis=0)

# Create a simple in-memory trajectory (dict format)
trajectory = {
    'pos': positions,
}


@requires_fields("pos")
def neighbor_kernel(idxs, get_frame, cutoff=1.0):
    """
    Example kernel that returns ragged arrays (varying number of neighbors per particle).
    
    For each particle, find all other particles within cutoff distance.
    Since different particles have different numbers of neighbors, the output
    is a ragged array that cannot be represented as a uniform numpy array.
    """
    # idxs is a single time index (or list with one element) for TimeBins
    if isinstance(idxs, list):
        t = idxs[0]
    else:
        t = idxs
    
    pos = get_frame(t)["pos"]
    n = pos.shape[0]
    
    # Find neighbors for each particle
    neighbor_lists = []
    neighbor_distances = []
    
    for i in range(n):
        # Compute distances to all other particles
        dists = np.linalg.norm(pos - pos[i], axis=1)
        # Find neighbors within cutoff (excluding self)
        neighbors = np.where((dists < cutoff) & (dists > 0))[0]
        neighbor_dists = dists[neighbors]
        
        neighbor_lists.append(neighbors)
        neighbor_distances.append(neighbor_dists)
    
    # Return tuple of ragged arrays
    return (neighbor_lists, neighbor_distances)


# Create time bins (single-index bins, not lag pairs)
# We'll divide the trajectory into 10 time bins
time_bins = TimeBins.from_source(trajectory, t_min=0, t_max=n_frames-1)

print(f"Number of bins: {time_bins.num_bins()}")
print(f"Number of particles: {n_particles}")
print(f"Number of frames: {n_frames}")
print()

# Run the ragged binned calculation
result = run_binned_ragged(
    neighbor_kernel,
    trajectory,
    time_bins,
    kernel_kwargs=dict(cutoff=1.5),
    show_progress=True,
)

print(f"\nResults structure:")
print(f"  Number of bins: {len(result.results)}")
print(f"  Type of result.results: {type(result.results)}")
print()

# Examine the first bin
print(f"First bin analysis:")
print(f"  Number of kernel calls in bin 0: {len(result.results[0])}")
if result.results[0]:
    first_call = result.results[0][0]
    print(f"  Type of first kernel output: {type(first_call)}")
    print(f"  First kernel output is a tuple of length: {len(first_call)}")
    
    neighbor_lists, neighbor_distances = first_call
    print(f"  Type of neighbor_lists: {type(neighbor_lists)}")
    print(f"  Number of particles: {len(neighbor_lists)}")
    print(f"  Number of neighbors for particle 0: {len(neighbor_lists[0])}")
    print(f"  Number of neighbors for particle 1: {len(neighbor_lists[1])}")
    print(f"  Neighbors of particle 0: {neighbor_lists[0]}")
    print(f"  Distances to neighbors of particle 0: {neighbor_distances[0]}")

print("\n✓ Successfully collected ragged array outputs!")
print("  Each bin contains a list of kernel outputs (one per time frame in that bin)")
print("  Each kernel output is whatever the kernel returns (in this case, neighbor lists)")
