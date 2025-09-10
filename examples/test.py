import numpy as np
from pydpmd.data import RigidBumpy, load
from pydpmd.data.bumpy_utils import get_closest_vertex_radius_for_mu_eff
from pydpmd.utils import join_systems, split_systems
from pydpmd.fields import DT_FLOAT, DT_INT, NeighborMethod

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


n_vertices = 3
mu_eff = 0.5
packing_fraction = 0.2
n_duplicates = 100

rb = create_2_particle_bumpy_disk_system(n_vertices, mu_eff, packing_fraction)
rb.set_neighbor_method(NeighborMethod.Naive)

joined = join_systems([rb for _ in range(n_duplicates)])
joined.save("/home/mmccraw/dev/data/09-09-25/bumpy/in")

new = load("/home/mmccraw/dev/data/09-09-25/bumpy/out", location=["final", "init", "restart"], load_trajectory=True, load_full=False)

print(new.trajectory)
print(new.trajectory.num_frames())

print(new.trajectory[0:10].timestep)

print(new.init)

# import h5py
# with h5py.File("/home/mmccraw/dev/data/09-09-25/bumpy/out/trajectory.h5", "r") as f:
#     print(f.keys())
#     print(f['angle'][()])


# TODO:
# fix the mu eff fitting stuff
# add minimization routines (initialize disks and call dpmd to minimize - for pbc and walls)

