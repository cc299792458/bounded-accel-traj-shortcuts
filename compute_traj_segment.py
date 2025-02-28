import numpy as np

from is_segment_feasible import is_segment_feasible
from minimum_acceleration import minimum_acceleration_interpolants
from univariate_time_optimal import univariate_time_optimal_interpolants

def compute_traj_segment(start_state, end_state, vmax, amax, collision_checker, bounds, n_dim):
    traj_segment_time = compute_traj_segment_time(start_state, end_state, vmax, amax, n_dim)
    traj_segment_param = compute_traj_segment_param(start_state, end_state, traj_segment_time, vmax, amax, n_dim)

    traj_feasibility = is_segment_feasible(start_state=start_state, traj_segment_time=traj_segment_time, traj_segment_param=traj_segment_param,
                               collision_checker=collision_checker, bounds=bounds, n_dim=n_dim)

    return traj_segment_time, traj_segment_param, traj_feasibility

def compute_traj_segment_time(start_state, end_state, vmax, amax, n_dim):
    """
    Calculate the maximum time required to traverse a segment across all dimensions,
    considering vmax and amax constraints.
    """
    t_requireds = []
    for dim in range(n_dim):
        trajectories, optimal_label = univariate_time_optimal_interpolants(
            start_pos=start_state[0][dim],
            end_pos=end_state[0][dim],
            start_vel=start_state[1][dim],
            end_vel=end_state[1][dim],
            vmax=vmax[dim],
            amax=amax[dim]
        )
        t_requireds.append(trajectories[optimal_label][0])

    return max(t_requireds)

def compute_traj_segment_param(start_state, end_state, traj_segment_time, vmax, amax, n_dim):
    """
    Calculate the traj for a single segment using minimum acceleration interpolants.
    """
    # Vectorized calculation for all dimensions
    traj_params = []
    for dim in range(n_dim):
        trajectories, optimal_label = minimum_acceleration_interpolants(
            start_pos=start_state[0][dim],
            end_pos=end_state[0][dim],
            start_vel=start_state[1][dim],
            end_vel=end_state[1][dim],
            vmax=vmax[dim],
            T=traj_segment_time,
            a_threshold=amax[dim]
        )
        if trajectories is None:
            # NOTE: This is actually possible. See consistency_validation.py for an example.
            return None
        traj_params.append((trajectories[optimal_label], optimal_label))

    return np.array(traj_params, dtype=object)
