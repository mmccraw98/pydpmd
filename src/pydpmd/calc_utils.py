import numpy as np

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