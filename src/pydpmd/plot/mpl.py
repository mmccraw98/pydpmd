import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm
import os

def init_box(ax, data, step, system_id, location='trajectory'):
    if location is None:
        loc = data
    else:
        loc = getattr(data, location)
    if 'box_size' in loc.fields():
        box_size = loc[step].box_size[system_id]
    elif 'box_size' in data.fields():
        box_size = data.box_size[system_id]
    else:
        raise ValueError(f"box_size not found in {location} or base fields")
    ax.set_xlim(0, box_size[0])
    ax.set_ylim(0, box_size[1])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def get_pos_rad_ids(data, step, system_id, which, location='trajectory'):
    if location is None:
        loc = data
    else:
        loc = getattr(data, location)
    if which == 'vertex':
        mask = data.vertex_system_id == system_id
        if 'vertex_pos' in loc.fields():
            pos = loc[step].vertex_pos[mask]
        else:
            raise ValueError(f"vertex_pos not found in {location} or base fields")
        if 'vertex_rad' in loc.fields():
            rad = loc[step].vertex_rad[mask]
        elif 'vertex_rad' in data.fields():
            rad = data.vertex_rad[mask]
        else:
            raise ValueError(f"vertex_rad not found in {location} or base fields")
        ids = data.vertex_particle_id[mask]
    elif which == 'particle':
        mask = data.system_id == system_id
        if 'pos' in loc.fields():
            pos = loc[step].pos[mask]
        else:
            raise ValueError(f"pos not found in {location}")
        if 'rad' in loc.fields():
            rad = loc[step].rad[mask]
        elif 'rad' in data.fields():
            rad = data.rad[mask]
        else:
            raise ValueError(f"rad not found in {location} or base fields")
        ids = np.arange(pos.shape[0])
    else:
        raise ValueError("which must be either 'vertex' or 'particle'")
    return pos, rad, ids

def draw_circle(ax, pos, rad, **kwargs):
    ax.add_artist(plt.Circle(pos, rad, **kwargs))

def create_animation(update_func, frames, filename, fps=30, dpi=100, bitrate=1800, **kwargs):
    """
    Create a matplotlib animation with progress bar and save to file.
    
    Parameters:
    -----------
    update_func : callable
        Function that updates the plot for each frame. Should accept (frame_num, ax, **kwargs)
    frames : int or iterable
        Number of frames or iterable of frame data
    filename : str
        Path where to save the animation file
    fps : int, default=30
        Frames per second
    dpi : int, default=100
        Resolution in dots per inch
    bitrate : int, default=1800
        Bitrate for video encoding
    **kwargs : dict
        Additional keyword arguments passed to update_func
        
    Returns:
    --------
    None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Progress bar setup
    if isinstance(frames, int):
        frame_count = frames
        frame_list = range(frames)
    else:
        frame_list = list(frames)
        frame_count = len(frame_list)
    
    pbar = tqdm(total=frame_count, desc="Creating animation")
    
    def animate_with_progress(frame_num):
        ax.clear()  # Clear the previous frame
        if isinstance(frames, int):
            actual_frame = frame_num
        else:
            actual_frame = frame_list[frame_num]
        update_func(actual_frame, ax, **kwargs)
        pbar.update(1)
        return []
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate_with_progress, frames=frame_count, 
        interval=1000/fps, blit=False, repeat=False
    )
    
    # Determine file extension and writer
    _, ext = os.path.splitext(filename)
    if ext.lower() in ['.mp4', '.avi']:
        writer = 'ffmpeg'
    elif ext.lower() == '.gif':
        writer = 'pillow'
    else:
        # Default to mp4
        writer = 'ffmpeg'
        if not filename.endswith('.mp4'):
            filename += '.mp4'
    
    # Save animation
    print(f"Saving animation to {filename}...")
    anim.save(filename, writer=writer, fps=fps, dpi=dpi, bitrate=bitrate)
    
    pbar.close()
    plt.close(fig)
    print(f"Animation saved successfully!")


def downsample(data, n_steps):
    """
    Downsample trajectory indices to get evenly spaced steps.
    
    Parameters:
    -----------
    data : object
        Trajectory data object with data.trajectory.timestep array
    n_steps : int
        Desired number of steps to sample
        
    Returns:
    --------
    np.ndarray
        Array of indices for trajectory sampling
    """
    # Get total number of available timesteps
    total_timesteps = len(data.trajectory.timestep[:])
    
    # If requesting more steps than available, return all indices
    if n_steps >= total_timesteps:
        return np.arange(total_timesteps)
    
    # Create evenly spaced indices for exact n_steps
    # Using linspace to get floating point positions, then round to integers
    indices = np.linspace(0, total_timesteps - 1, n_steps)
    indices = np.round(indices).astype(int)
    
    # Remove any potential duplicates while preserving order
    # This can happen when total_timesteps is only slightly larger than n_steps
    _, unique_idx = np.unique(indices, return_index=True)
    indices = indices[np.sort(unique_idx)]
    
    return indices


def draw_particles_frame(step, ax, data, system_id=0, which='vertex', cmap_name='viridis', location='trajectory'):
    """
    Draw particles for a single animation frame.
    
    Parameters:
    -----------
    step : int
        Time step to visualize
    ax : matplotlib.axes.Axes
        Matplotlib axis to draw on
    data : object
        Trajectory data object
    system_id : int, default=0
        System ID to visualize
    which : str, default='vertex'
        Whether to draw 'vertex' or 'particle' data
    cmap_name : str, default='viridis'
        Colormap name for particle coloring
    """
    # Initialize box
    init_box(ax, data, step, system_id, location)
    
    # Get particle data
    pos, rad, ids = get_pos_rad_ids(data, step, system_id, which, location)
    
    # Set up colormap
    # TODO: make this a function: color by: id, rad, n_vertices, etc...
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(min(ids), max(ids))
    
    # Draw particles
    for p, r, particle_id in zip(pos, rad, ids):
        draw_circle(ax, p, r, color=cmap(norm(particle_id)))
    
    # TODO: add pbc images for particles less than 1 rad from the wall

    # Set title with step information
    ax.set_title(f'Step {step}, System {system_id} ({which})', fontsize=14)
