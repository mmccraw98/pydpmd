from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

from .data.base_particle import BaseParticle
from .trajectory import ConcatTrajectory, SliceTrajectory
from .fields import IndexSpace, DT_INT, FieldSpec


def join_systems(particles: List[BaseParticle]) -> BaseParticle:
    if not particles:
        raise ValueError("join_systems: empty list")
    # Ensure same class
    cls = particles[0].__class__
    if any(p.__class__ is not cls for p in particles):
        raise TypeError("join_systems: all particles must be of the same class")
    # Ensure same set of present arrays (dynamic aware): names present across inputs must match
    def present_fields(p: BaseParticle) -> set:
        spec_names = set(p._spec_fn().keys())
        return {name for name in spec_names if getattr(p, name) is not None}
    ref_fields = present_fields(particles[0])
    for p in particles[1:]:
        if present_fields(p) != ref_fields:
            raise ValueError("join_systems: all particles must have the same present arrays")

    out = cls()
    spec_map = out._spec_fn()

    # Build union of present field names across inputs
    name_union: set[str] = set()
    per_name_space: Dict[str, IndexSpace] = {}
    per_name_dtype: Dict[str, np.dtype] = {}
    for p in particles:
        p_spec_map = p._spec_fn()
        for name, spec in p_spec_map.items():
            if getattr(p, name) is not None:
                name_union.add(name)
                per_name_space[name] = spec.index_space
                per_name_dtype[name] = np.dtype(spec.dtype)

    # Concatenate all arrays generically (including vertex arrays if present)
    for name in sorted(name_union):
        arrays = [getattr(p, name) for p in particles if getattr(p, name) is not None]
        if not arrays:
            continue
        if name in ("system_id", "system_size", "system_offset"):
            # handled below to remap ids
            continue
        # vertex linkage arrays handled below explicitly
        if name in ("particle_offset", "vertex_particle_id", "vertex_system_offset", "vertex_system_size", "vertex_system_id"):
            continue
        out_arr = np.concatenate(arrays, axis=0) if arrays[0].ndim >= 1 else np.array(arrays)
        setattr(out, name, out_arr)
        # Ensure spec exists on out for dynamic fields
        if name not in spec_map:
            space = per_name_space.get(name, None)
            if space is not None:
                tail_shape = out_arr.shape[1:]
                def _make_expected_shape_fn(space: IndexSpace, tail: Tuple[int, ...]):
                    return (lambda s=space, t=tail: ((
                        (out.n_systems() if s == IndexSpace.System else (
                            out.n_particles() if s == IndexSpace.Particle else (
                                out.n_vertices() if s == IndexSpace.Vertex else 0
                            )
                        )),
                    ) + t))
                spec_map[name] = FieldSpec(
                    name=name,
                    index_space=space,
                    dtype=per_name_dtype.get(name, arrays[0].dtype),
                    expected_shape_fn=_make_expected_shape_fn(space, tail_shape),
                )
                # Preserve static marking if consistent
                if all(hasattr(p, "_extra_static_fields") and (name in getattr(p, "_extra_static_fields")) for p in particles):
                    getattr(out, "_extra_static_fields", set()).add(name)

    # Remap system arrays
    total_systems = 0
    sys_id_list = []
    sys_size_list = []
    sys_offset_list = [0]
    for p in particles:
        if p.system_size is None or p.system_offset is None or p.system_id is None:
            continue
        S = len(p.system_size)
        sys_size_list.append(p.system_size)
        sys_id_list.append(p.system_id + total_systems)
        last_offset = sys_offset_list[-1]
        sys_offset_list.extend((p.system_offset[1:] + last_offset).tolist())
        total_systems += S
    if sys_id_list:
        out.system_id = np.concatenate(sys_id_list, axis=0).astype(DT_INT, copy=False)
    if sys_size_list:
        out.system_size = np.concatenate(sys_size_list, axis=0)
    out.system_offset = np.asarray(sys_offset_list, dtype=DT_INT)

    # Remap vertex linkage arrays if present
    has_vertex = any(getattr(p, "vertex_pos", None) is not None for p in particles) or any(getattr(p, "particle_offset", None) is not None for p in particles)
    if has_vertex:
        # Determine N per particle and Nv per particle
        particle_count_offsets: List[int] = []
        nv_offsets: List[int] = []
        n_acc = 0
        nv_acc = 0
        for p in particles:
            Np = getattr(p, "system_id")
            N = int(Np.shape[0]) if Np is not None else (p.pos.shape[0] if getattr(p, "pos", None) is not None else 0)
            if getattr(p, "particle_offset", None) is not None:
                Nv = int(p.particle_offset[-1])
            else:
                vpos = getattr(p, "vertex_pos", None)
                Nv = int(vpos.shape[0]) if vpos is not None else 0
            particle_count_offsets.append(n_acc)
            nv_offsets.append(nv_acc)
            n_acc += N
            nv_acc += Nv

        # vertex_particle_id
        if all(getattr(p, "vertex_particle_id", None) is not None for p in particles):
            vpid_list = []
            for idx, p in enumerate(particles):
                vpid = p.vertex_particle_id + particle_count_offsets[idx]
                vpid_list.append(vpid)
            out.vertex_particle_id = np.concatenate(vpid_list, axis=0)

        # particle_offset
        if all(getattr(p, "particle_offset", None) is not None for p in particles):
            new_offsets: List[int] = [0]
            for idx, p in enumerate(particles):
                po = p.particle_offset
                base = nv_offsets[idx]
                # append shifted offsets excluding the leading zero
                new_offsets.extend((po[1:] + base).tolist())
            out.particle_offset = np.asarray(new_offsets, dtype=DT_INT)

        # vertex_system_size and vertex_system_offset
        if all(getattr(p, "vertex_system_size", None) is not None for p in particles):
            out.vertex_system_size = np.concatenate([p.vertex_system_size for p in particles], axis=0).astype(DT_INT)
            out.vertex_system_offset = np.concatenate([[0], np.cumsum(out.vertex_system_size)], axis=0).astype(DT_INT)
        
        # vertex_system_id
        if all(getattr(p, "vertex_system_id", None) is not None for p in particles):
            out.vertex_system_id = out.system_id[out.vertex_particle_id]

    # Trajectory: if all inputs have trajectories with identical frames/fields, create a concatenated view
    if all(getattr(p, "trajectory", None) is not None for p in particles):
        out.trajectory = ConcatTrajectory([p.trajectory for p in particles])

    # Preserve neighbor_method if consistent across inputs
    neighbor_methods = [getattr(p, "neighbor_method", None) for p in particles]
    if neighbor_methods and all(nm == neighbor_methods[0] for nm in neighbor_methods):
        out.neighbor_method = neighbor_methods[0]

    # Validate output
    out.validate()
    return out


def split_systems(p: BaseParticle) -> List[BaseParticle]:
    if p.system_offset is None or p.system_id is None or p.system_size is None:
        raise ValueError("split_systems: missing system arrays")
    parts: List[BaseParticle] = []
    S = len(p.system_size)
    for s in range(S):
        cls = p.__class__
        q = cls()
        i0 = int(p.system_offset[s]); i1 = int(p.system_offset[s+1])
        # Ensure q knows about dynamic fields present on p
        p_spec_map = p._spec_fn()
        q_spec_map = q._spec_fn()
        for name, pspec in p_spec_map.items():
            if name not in q_spec_map and getattr(p, name, None) is not None:
                parr = getattr(p, name)
                tail_shape = parr.shape[1:] if hasattr(parr, 'shape') and parr is not None and parr.ndim >= 1 else ()
                def _make_expected_shape_fn(space: IndexSpace, tail: Tuple[int, ...]):
                    return (lambda s=space, t=tail: ((
                        (q.n_systems() if s == IndexSpace.System else (
                            q.n_particles() if s == IndexSpace.Particle else (
                                q.n_vertices() if s == IndexSpace.Vertex else 0
                            )
                        )),
                    ) + t))
                q_spec_map[name] = FieldSpec(
                    name=name,
                    index_space=pspec.index_space,
                    dtype=pspec.dtype,
                    expected_shape_fn=_make_expected_shape_fn(pspec.index_space, tail_shape),
                )
                if hasattr(p, "_extra_static_fields") and name in getattr(p, "_extra_static_fields"):
                    getattr(q, "_extra_static_fields", set()).add(name)

        # slice each present array according to index space
        for name, pspec in p_spec_map.items():
            arr = getattr(p, name, None)
            if arr is None:
                continue
            if name in ("system_id",):
                # After split, single-system IDs are all zero
                q.system_id = np.zeros((i1-i0,), dtype=DT_INT)
            elif name in ("system_size",):
                q.system_size = p.system_size[s:s+1].copy()
            elif name in ("system_offset",):
                q.system_offset = np.array([0, i1-i0], dtype=DT_INT)
            else:
                # Use FieldSpec index space to slice
                space = pspec.index_space
                if space == IndexSpace.Particle and hasattr(p, "system_id") and p.system_id is not None:
                    setattr(q, name, arr[i0:i1].copy())
                elif space == IndexSpace.System and hasattr(p, "box_size") and p.box_size is not None:
                    setattr(q, name, arr[s:s+1].copy())
                elif space == IndexSpace.Vertex:
                    vso = getattr(p, "vertex_system_offset", None)
                    if vso is not None:
                        j0 = int(vso[s]); j1 = int(vso[s+1])
                        setattr(q, name, arr[j0:j1].copy())
        # vertex split for poly data if present
        vso = getattr(p, "vertex_system_offset", None)
        if vso is not None:
            j0 = int(vso[s]); j1 = int(vso[s+1])
            for name in ("vertex_pos", "vertex_vel", "vertex_force", "vertex_pe", "vertex_mass", "vertex_rad", "vertex_particle_id", "vertex_system_id", "vertex_system_size"):
                arr = getattr(p, name, None)
                if arr is None:
                    continue
                if name == "vertex_system_id":
                    # After split, system ids are zero-based single system
                    setattr(q, name, np.zeros((j1-j0,), dtype=DT_INT))
                elif name == "vertex_particle_id":
                    # Rebase vertex->particle mapping to 0..(i1-i0-1)
                    vpid = arr[j0:j1].astype(np.int64, copy=True)
                    vpid -= int(i0)
                    setattr(q, name, vpid.astype(DT_INT, copy=False))
                else:
                    setattr(q, name, arr[j0:j1].copy())
            # vertex_system_size and vertex_system_offset for single system
            vss = getattr(p, "vertex_system_size", None)
            if vss is not None:
                q.vertex_system_size = vss[s:s+1].copy()
                q.vertex_system_offset = np.array([0, int(q.vertex_system_size[0])], dtype=DT_INT)
            # particle_offset rebased to local vertex indices
            po = getattr(p, "particle_offset", None)
            if po is not None:
                # particle_offset is length N+1; select [i0:i1+1] and shift so first is 0
                local_po = po[i0:i1+1].astype(np.int64, copy=True)
                local_po -= int(po[i0])
                q.particle_offset = local_po.astype(DT_INT, copy=False)
        # Slice trajectory if present using SliceTrajectory
        if getattr(p, "trajectory", None) is not None:
            # Map common dataset names to index spaces (extend as needed)
            idx_spaces: Dict[str, IndexSpace] = {}
            for name in p.trajectory.fields():
                # crude mapping based on known names
                if name in ("pos", "vel", "force", "angle", "torque", "angular_vel", "pe", "ke", "area"):
                    idx_spaces[name] = IndexSpace.Particle
                elif name in ("vertex_pos", "vertex_force", "vertex_vel", "vertex_pe"):
                    idx_spaces[name] = IndexSpace.Vertex
                elif name in ("box_size", "packing_fraction", "pe_total", "ke_total"):
                    idx_spaces[name] = IndexSpace.System
            ranges = {
                IndexSpace.Particle: (i0, i1),
                IndexSpace.System: (s, s+1),
                # Vertex slicing would require vertex_system_offset; if present on p, we can compute
            }
            # Vertex range if we have vertex_system_offset
            vso = getattr(p, "vertex_system_offset", None)
            if vso is not None:
                j0 = int(vso[s]); j1 = int(vso[s+1])
                ranges[IndexSpace.Vertex] = (j0, j1)
            q.trajectory = SliceTrajectory(p.trajectory, idx_spaces, ranges)

        # Preserve neighbor_method from source
        q.neighbor_method = getattr(p, "neighbor_method", None)

        q.validate()
        parts.append(q)
    return parts


