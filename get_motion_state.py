import numpy as np

def get_motion_states_at_global_t(path, traj_segment_times, traj_segment_params, t, n_dim):
    """
    Find the interpolated state (position, velocity) at a specific time t within the trajectory.

    This function computes the position and velocity at a specific global time `t` by iterating through
    the trajectory segments and interpolating the motion within the relevant segment. The function will
    return the state based on the current position, velocity, and the specific motion parameters for each segment.

    Input:
    - path: A list of states (positions and velocities) at the start of each segment. Each entry in the path corresponds 
      to a segment and contains the initial state (position, velocity) at the beginning of that segment.
    - traj_segment_times: A numpy array of time durations for each segment. Each element represents the duration 
      of the corresponding trajectory segment.
    - traj_segment_params: A list of parameters for each trajectory segment, containing:
        - For each segment, the parameters like acceleration limits, switch times, and trajectory type.
    - t: The target time at which the position and velocity are to be interpolated, given as a global time.
    - n_dim: The number of dimensions in the trajectory (e.g., 3 for 3D motion).

    Return:
    - position: A numpy array of positions at time `t` for each dimension.
    - velocity: A numpy array of velocities at time `t` for each dimension.
    
    If the target time `t` is beyond the total duration of the trajectory, the function will return `None`.
    """
    elapsed_time = 0

    # Iterate over all trajectory segments to find which segment the time `t` belongs to
    for i in range(traj_segment_times.shape[0]):  # Use shape[0] to get the number of trajectory segments
        traj_segment_time = traj_segment_times[i]
        if elapsed_time + traj_segment_time >= t:
            # Relative time within the current segment
            relative_t = t - elapsed_time
            start_state = path[i]

            # Compute and return the interpolated state at time t for the current segment
            return get_motion_states_at_local_t(start_state=start_state, traj_segment_param=traj_segment_params[i], t=relative_t, n_dim=n_dim)

        elapsed_time += traj_segment_time

    # If t is beyond the total trajectory duration, return None
    # NOTE: This could happen at the boundary due to numerical precision issues
    return None

def get_motion_states_at_local_t(start_state, traj_segment_param, t, n_dim):
    """
    Compute the state (position, velocity) within a segment at time t.

    This function calculates the position and velocity at a specific time `t` within a trajectory 
    segment based on the segment's parameters and the starting state.

    Input:
    - start_state: A tuple or list where:
        - start_state[0] is a list/array of the starting positions for each dimension.
        - start_state[1] is a list/array of the starting velocities for each dimension.
    - traj_segment_param: A list of parameters for each trajectory segment, where each entry corresponds
      to a dimension and contains:
        - (amax, switch_time1, switch_time2): The maximum acceleration and two switch times that 
          define the motion profile for this segment.
        - traj_type: The type of trajectory (e.g., acceleration-deceleration profile).
    - t: The relative time within the trajectory segment at which the position and velocity are calculated.
    - n_dim: The number of dimensions in the trajectory (e.g., 3 for 3D motion).

    Return:
    - position: A numpy array of positions at time `t` for each dimension (size: n_dim).
    - velocity: A numpy array of velocities at time `t` for each dimension (size: n_dim).
    """
    position = np.zeros(n_dim)
    velocity = np.zeros(n_dim)

    # Loop over each dimension to calculate position and velocity
    for dim in range(n_dim):
        # Unpack trajectory segment parameters
        (amax, switch_time1, switch_time2), traj_type = traj_segment_param[dim]
        
        # Get starting position and velocity for this dimension
        start_pos, start_vel = start_state[0][dim], start_state[1][dim]

        # Calculate the position, velocity, and acceleration at time t for this dimension
        pos, vel, acc = get_motion_state_at_local_t(
            t=t, traj_type=traj_type, start_pos=start_pos, start_vel=start_vel, 
            amax=amax, switch_time1=switch_time1, switch_time2=switch_time2
        )

        # Store the results for the current dimension
        position[dim], velocity[dim] = pos, vel

    return position, velocity


def get_motion_state_at_local_t(t, traj_type, start_pos, start_vel, amax, switch_time1, switch_time2):
    """
    Return the position, velocity, and acceleration at local time t for the given trajectory type.
    
    Inputs:
    - t: local time within the trajectory [0, total_time].
    - traj_type: one of 'P+P-', 'P-P+', 'P+L+P-', 'P-L-P+'.
    - start_pos, start_vel: initial position and velocity.
    - amax: maximum acceleration.
    - switch_time1: first switching time (end of the first phase)
    - switch_time2: second switching time (for trajectories with cruise phase, otherwise None)
    
    Returns:
    - pos, vel, acc: position, velocity, and acceleration at time t.
    """
    if traj_type == "P+P-":
        # Two phases: acceleration then deceleration.
        if t <= switch_time1:
            # Acceleration phase
            pos = start_pos + start_vel * t + 0.5 * amax * t**2
            vel = start_vel + amax * t
            acc = amax
        else:
            # Deceleration phase
            t_dec = t - switch_time1
            # Compute state at the switching point
            pos_switch = start_pos + start_vel * switch_time1 + 0.5 * amax * switch_time1**2
            vel_switch = start_vel + amax * switch_time1
            pos = pos_switch + vel_switch * t_dec - 0.5 * amax * t_dec**2
            vel = vel_switch - amax * t_dec
            acc = -amax
        return pos, vel, acc

    elif traj_type == "P-P+":
        # Two phases: deceleration then acceleration.
        if t <= switch_time1:
            # Deceleration phase
            pos = start_pos + start_vel * t - 0.5 * amax * t**2
            vel = start_vel - amax * t
            acc = -amax
        else:
            # Acceleration phase
            t_acc = t - switch_time1
            pos_switch = start_pos + start_vel * switch_time1 - 0.5 * amax * switch_time1**2
            vel_switch = start_vel - amax * switch_time1
            pos = pos_switch + vel_switch * t_acc + 0.5 * amax * t_acc**2
            vel = vel_switch + amax * t_acc
            acc = amax
        return pos, vel, acc

    elif traj_type == "P+L+P-":
        # Three phases: acceleration, constant velocity, deceleration.
        if t <= switch_time1:
            # Acceleration phase
            pos = start_pos + start_vel * t + 0.5 * amax * t**2
            vel = start_vel + amax * t
            acc = amax
        elif t <= switch_time2:
            # Constant velocity phase
            t_const = t - switch_time1
            # State at end of acceleration phase (should reach vmax)
            pos_switch = start_pos + start_vel * switch_time1 + 0.5 * amax * switch_time1**2
            vel_const = start_vel + amax * switch_time1
            pos = pos_switch + vel_const * t_const
            vel = vel_const
            acc = 0
        else:
            # Deceleration phase
            t_dec = t - switch_time2
            # State at beginning of deceleration phase
            pos_switch2 = (start_pos + start_vel * switch_time1 + 0.5 * amax * switch_time1**2 +
                           (start_vel + amax * switch_time1) * (switch_time2 - switch_time1))
            vel_switch2 = start_vel + amax * switch_time1  # equals vmax
            pos = pos_switch2 + vel_switch2 * t_dec - 0.5 * amax * t_dec**2
            vel = vel_switch2 - amax * t_dec
            acc = -amax
        return pos, vel, acc

    elif traj_type == "P-L-P+":
        # Three phases: deceleration, constant velocity, acceleration.
        if t <= switch_time1:
            # Deceleration phase
            pos = start_pos + start_vel * t - 0.5 * amax * t**2
            vel = start_vel - amax * t
            acc = -amax
        elif t <= switch_time2:
            # Constant velocity phase
            t_const = t - switch_time1
            pos_switch = start_pos + start_vel * switch_time1 - 0.5 * amax * switch_time1**2
            vel_const = start_vel - amax * switch_time1  # equals -vmax
            pos = pos_switch + vel_const * t_const
            vel = vel_const
            acc = 0
        else:
            # Acceleration phase
            t_acc = t - switch_time2
            pos_switch2 = (start_pos + start_vel * switch_time1 - 0.5 * amax * switch_time1**2 +
                           (start_vel - amax * switch_time1) * (switch_time2 - switch_time1))
            vel_switch2 = start_vel - amax * switch_time1  # equals -vmax
            pos = pos_switch2 + vel_switch2 * t_acc + 0.5 * amax * t_acc**2
            vel = vel_switch2 + amax * t_acc
            acc = amax
        return pos, vel, acc

    else:
        raise ValueError("Unknown trajectory type")
