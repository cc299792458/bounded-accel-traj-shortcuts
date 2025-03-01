import numpy as np

from is_segment_feasible import is_segment_feasible

def time_optimal_starigh_line(start_state, end_state, vmax, amax, collision_checker, bounds, n_dim):
    """
    Compute a time-optimal straight-line trajectory between two states.

    Parameters:
      start_state: Initial state with positions in the first element.
      end_state: Final state with positions in the first element.
      vmax: List/array of maximum velocities per dimension.
      amax: List/array of maximum accelerations per dimension.
      collision_checker: Function to verify if the trajectory is collision-free.
      bounds: Limits for collision checking.
      n_dim: Number of dimensions.

    Returns:
      traj_segment_time (float): Duration of the trajectory segment.
      traj_segment_param (np.array): Array of tuples for each dimension containing 
                                     (acceleration parameter(s), switching time(s), label).
      traj_feasibility (bool): True if the trajectory is feasible, otherwise False.
    """
    vel_s, acc_s = np.inf, np.inf
    for i in range(n_dim):
        a, b = start_state[0][i], end_state[0][i]
        vel_s = vmax[i] / (abs(b - a)) if vmax[i] / (abs(b - a)) < vel_s else vel_s
        acc_s = amax[i] / (abs(b - a)) if amax[i] / (abs(b - a)) < acc_s else acc_s

    traj_segment_param = []
    t_p = np.sqrt(1 / acc_s)
    if acc_s * t_p < vel_s:
        traj_segment_time = t_p * 2
        for i in range(n_dim):
            a, b = start_state[0][i], end_state[0][i]
            traj_segment_param.append(((abs((b - a)) * acc_s, t_p, None), "P+P-" if b >= a else "P-P+"))
    else:
        t1 = vel_s / acc_s
        t2 = 1 / vel_s - vel_s / acc_s
        traj_segment_time = t1 * 2 + t2
        for i in range(n_dim):
            a, b = start_state[0][i], end_state[0][i]
            traj_segment_param.append(((abs((b - a)) * acc_s, t1, t1 + t2), "P+L+P-" if b >= a else "P-L-P+"))
    
    traj_segment_param = np.array(traj_segment_param, dtype=object)

    traj_feasibility = is_segment_feasible(start_state=start_state, traj_segment_time=traj_segment_time, traj_segment_param=traj_segment_param,
                                           collision_checker=collision_checker, bounds=bounds, n_dim=n_dim)

    return traj_segment_time, traj_segment_param, traj_feasibility