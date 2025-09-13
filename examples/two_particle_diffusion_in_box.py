"""
Simple investigation of the diffusion of a 2-particle system in a walled boundary.
The 2-particle system is composed of a disk and a bumpy particle (possibly a disk if desired) with a variable number of vertices and friction coefficient.
First, the maximum packing fraction is estimated by jamming n_jam_duplicates of the system from random initial positions within the box.
Once the maximum packing fraction (phi_j) is estimated, max_n_dynamics_duplicates of the system are create with random initial positions corresponding -
to packing fractions from phi_j to phi_j - max_phi_offset.  Each system is then equilibrated to a 0-overlap state and the velocities are set to the -
desired temperature.  Finally, NVE dynamics are run for n_steps steps.
The idea is to determine the translational and rotational diffusion coefficients for the bumpy (rotational) particles and their relationship to nearby -
free volume.
"""

data_root = ""  # fill in where data is stored
script_root = ""  # fill in where dpmd is installed


import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.data.bumpy_utils import get_closest_vertex_radius_for_mu_eff
from pydpmd.utils import join_systems, split_systems
from pydpmd.fields import NeighborMethod, DT_INT
import matplotlib.pyplot as plt
import subprocess
import os

# create bumpy - disk 2-particle system
def create_2_particle_bumpy_disk_system(n_vertices: int, mu_eff: float, packing_fraction: float):
    n_vertices_per_particle = np.array([n_vertices, 1], dtype=DT_INT)
    particle_radius = 0.5
    particle_mass = 1.0
    e_interaction = 1.0

    rb = RigidBumpy()
    rb.allocate_particles(n_vertices_per_particle.size)
    rb.allocate_systems(1)
    rb.allocate_vertices(n_vertices_per_particle.sum())
    rb.n_vertices_per_particle = n_vertices_per_particle
    rb.set_ids()
    rb.validate()
    rb.rad.fill(particle_radius)
    rb.mass.fill(particle_mass)
    rb.e_interaction.fill(e_interaction)
    rb.calculate_uniform_vertex_mass()

    vertex_radius = get_closest_vertex_radius_for_mu_eff(mu_eff, particle_radius, n_vertices)
    rb.vertex_rad.fill(vertex_radius)
    rb.vertex_rad[-1] = particle_radius

    rb.box_size.fill(1.0)
    rb.set_positions(0, 0)
    rb.set_vertices_on_particles_as_disk()
    rb.calculate_inertia()
    rb.scale_to_packing_fraction(packing_fraction)

    return rb

jam_data_path = os.path.join(data_root, "jam")
offset_data_path = os.path.join(data_root, "offset")
dynamics_data_path = os.path.join(data_root, "dynamics")

mu_eff = 0.1

n_jam_duplicates = 100

n_phi_steps = 50
min_phi_offset = 1e-4
max_phi_offset = 4e-1

max_n_dynamics_duplicates = 10000

temperature = 1e-5
n_steps = 1e6

n_vertices = 3
initial_packing_fraction = 0.2
rng_seed = 0

rb = create_2_particle_bumpy_disk_system(n_vertices, mu_eff, initial_packing_fraction)
rb.set_neighbor_method(NeighborMethod.Naive)

# place n_jam_duplicates of the system randomly within the box, and jam them
jam_data = join_systems([rb for _ in range(n_jam_duplicates)])
jam_data.set_positions(1, rng_seed)  # set random positions
jam_data.set_vertices_on_particles_as_disk()  # update the vertex positions  # may not need this anymore
jam_data.save(jam_data_path, locations=["init"], save_trajectory=False)
subprocess.run([
    os.path.join(script_root, "jam_rigid_bumpy_wall_final"),
    jam_data_path,
    jam_data_path,
], check=True)
jam_data = load(jam_data_path, location=["final", "init"])

# find the unique phi_j values and pick the highest one
pe_tol = 1e-15
pe_mask = jam_data.final.pe_total / jam_data.system_size < pe_tol
phi_j = np.max(jam_data.final.packing_fraction[pe_mask])
if phi_j - max_phi_offset < 0.1:  # do not let the packing fraction be too small - it isnt very meaningful and will cause a problem if <= 0
    max_phi_offset = phi_j - 0.1
delta_phi = np.logspace(np.log10(min_phi_offset), np.log10(max_phi_offset), n_phi_steps)
phi = phi_j - delta_phi

# create a block of n_phi_steps systems, each with a different phi value
offset_data = join_systems([rb for _ in range(n_phi_steps)])
offset_data.scale_to_packing_fraction(phi)
offset_data.add_array(delta_phi, 'delta_phi')

# create max_n_dynamics_duplicates / n_phi_steps duplicates of the concatenated dynamics_data
offset_data = join_systems([offset_data for _ in range(max_n_dynamics_duplicates // n_phi_steps)])
offset_data.set_positions(1, rng_seed)  # set random positions
offset_data.save(offset_data_path, locations=["init"])

subprocess.run([  # equilibrate the system and save the data to the dynamics_data_path
    os.path.join(script_root, "rigid_bumpy_equilibrate_wall"),
    offset_data_path,
    dynamics_data_path,
], check=True)
dynamics_data = load(dynamics_data_path, location=["final"])  # set the velocities and overwrite the data
dynamics_data.set_velocities(temperature, rng_seed)
dynamics_data.add_array(offset_data.delta_phi.copy(), 'delta_phi')
dynamics_data.save(dynamics_data_path, locations=["init"])
subprocess.run([  # run the dynamics
    os.path.join(script_root, "nve_rigid_bumpy_wall_final"),
    dynamics_data_path,
    dynamics_data_path,
    str(n_steps),
], check=True)
