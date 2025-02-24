def get_motion_state_at_local_t(t, traj_type, start_pos, start_vel, end_vel, vmax, amax, switch_time1, switch_time2, total_time):
    """
    Return the position, velocity, and acceleration at local time t for the given trajectory type.
    
    Inputs:
    - t: local time within the trajectory [0, total_time].
    - traj_type: one of 'P+P-', 'P-P+', 'P+L+P-', 'P-L-P+'.
    - start_pos, start_vel: initial position and velocity.
    - end_vel: final velocity (used for consistency; some cases might not need it directly).
    - vmax, amax: maximum velocity and acceleration.
    - switch_time1: first switching time (end of the first phase)
    - switch_time2: second switching time (for trajectories with cruise phase, otherwise None)
    - total_time: total duration of the trajectory.
    
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
