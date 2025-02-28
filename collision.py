import numpy as np

from get_motion_state import get_motion_states_at_local_t

def is_segment_collision_free(start_state, traj_segment_time, traj_segment_param, collision_checker, n_dim, time_step=0.01):
    # Generate time points to sample along the traj
    num_samples = int(traj_segment_time / time_step) + 1
    sampled_times = np.linspace(0, traj_segment_time, num_samples)

    for time in sampled_times:
        state = get_motion_states_at_local_t(start_state=start_state, traj_segment_param=traj_segment_param, t=time, n_dim=n_dim)
        if not collision_checker(state):
            return False
    return True
