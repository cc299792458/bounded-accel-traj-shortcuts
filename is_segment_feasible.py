import numpy as np

from get_motion_state import get_motion_states_at_local_t

def is_segment_feasible(start_state, traj_segment_time, traj_segment_param, collision_checker, bounds, n_dim, time_step=0.01):
    """
        Check if the trajectory segment is within bounds and collision free.
    """
    if traj_segment_time is None or traj_segment_param is None:
        return False
    # Generate time points to sample along the traj
    num_samples = int(traj_segment_time / time_step) + 1
    sampled_times = np.linspace(0, traj_segment_time, num_samples)

    for time in sampled_times:
        state = get_motion_states_at_local_t(start_state=start_state, traj_segment_param=traj_segment_param, t=time, n_dim=n_dim)
        if np.any(state[0] < bounds[0]) or np.any(state[0] > bounds[1]):
            return False
        if not collision_checker(state):
            return False
    return True
