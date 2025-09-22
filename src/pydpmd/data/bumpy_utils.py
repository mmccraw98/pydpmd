import numpy as np
from scipy.optimize import minimize_scalar, brentq

def calc_mu_eff(vertex_radius, outer_radius, num_vertices):
    return 1 / np.sqrt(((2 * vertex_radius) / ((outer_radius - vertex_radius) * np.sin(np.pi / num_vertices))) ** 2 - 1)

# TODO: check these because it seems to always return a result - this makes no sense since it should fail sometimes

def get_closest_vertex_radius_for_mu_eff(mu_eff, outer_radius, num_vertices):
    # Calculate mathematically valid bounds
    sin_term = np.sin(np.pi / num_vertices)
    min_vertex_radius = outer_radius * sin_term / (2 + sin_term) + 1e-12
    max_vertex_radius = outer_radius - 1e-12
    
    # Check if target mu_eff is achievable
    max_mu_eff = calc_mu_eff(min_vertex_radius, outer_radius, num_vertices)
    min_mu_eff = calc_mu_eff(max_vertex_radius, outer_radius, num_vertices)
    
    if mu_eff > max_mu_eff or mu_eff < min_mu_eff:
        # Target mu_eff is outside achievable range
        return np.nan
    try:
        # Use root finding since we want calc_mu_eff(vertex_radius) = mu_eff
        def objective(vertex_radius):
            return calc_mu_eff(vertex_radius, outer_radius, num_vertices) - mu_eff
        
        # Brent's method is robust for this monotonic function
        result = brentq(objective, min_vertex_radius, max_vertex_radius, xtol=1e-12)
        return result
        
    except (ValueError, RuntimeError, ZeroDivisionError):
        # Fallback to bounded scalar minimization if root finding fails
        def obj_squared(vertex_radius):
            try:
                return (calc_mu_eff(vertex_radius, outer_radius, num_vertices) - mu_eff) ** 2
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.inf
        
        result = minimize_scalar(obj_squared, bounds=(min_vertex_radius, max_vertex_radius), method='bounded')
        return result.x if result.success else np.nan

def get_closest_num_vertices_for_mu_eff_and_radii(mu_eff, outer_radius, vertex_radius, min_nv=1, max_nv=np.inf):
    pass


def get_closest_num_vertices_for_friction_and_segment_length(vertex_radius, outer_radius, target_segment_length, target_friction, target_num_vertices, vertex_count_offset=5, min_num_vertices=2):
    ideal_num_vertices = target_num_vertices
    min_cost = np.inf
    for num_vertices in range(max(min_num_vertices, ideal_num_vertices - vertex_count_offset), ideal_num_vertices + vertex_count_offset):
        if num_vertices <= min_num_vertices:
            continue
        vertex_angle = 2 * np.pi / num_vertices
        inner_radius = outer_radius - vertex_radius
        friction = calc_mu_eff(vertex_radius, outer_radius, num_vertices)
        segment_length = inner_radius / vertex_radius * np.sin(vertex_angle / 2)

        cost = abs(segment_length / target_segment_length - 1) + abs(friction / target_friction - 1)
        if ~np.isnan(cost) and cost < min_cost:
            ideal_num_vertices = num_vertices
            min_cost = cost
    return ideal_num_vertices

def get_bumpy_dists(num_vertices, outer_radius, vertex_radius):
    sigma = outer_radius * 2
    sigma_v = vertex_radius * 2
    n_v = num_vertices
    sigma_p_i = sigma - sigma_v
    # the closest center-to-center distance between two particles of the same species at a symmetric contact
    d_0 = (np.cos(np.pi / n_v) / 2 + 1 / 2 + np.sqrt((sigma_v / sigma_p_i) ** 2 - (np.sin(np.pi / n_v) / 2) ** 2)) * sigma_p_i
    # the absolute closest center-to-center distance between two particles of the same species
    d_min = (np.sqrt(np.cos(np.pi / n_v) ** 2 + (sigma_v / sigma_p_i) ** 2 + np.cos(np.pi / n_v) * np.sqrt(4 * (sigma_v / sigma_p_i) ** 2 - np.sin(np.pi / n_v) ** 2))) * sigma_p_i
    return d_0, d_min