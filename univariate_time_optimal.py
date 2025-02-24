import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from solve_quadratic import solve_quadratic
from get_motion_state import get_motion_state_at_local_t

def univariate_time_optimal_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, amax):
    """
    Compute the execution times and switching times for all trajectory classes for univariate motion.
    
    Inputs:
    - start_pos, end_pos: Initial and final positions.
    - start_vel, end_vel: Initial and final velocities.
    - vmax: Maximum velocity.
    - amax: Maximum acceleration.
    
    Returns:
    - trajectories: A dictionary mapping each trajectory class to a tuple 
                    (execution_time, switch_time1, switch_time2) if valid, or None if invalid.
                    The classes are:
                        'P+P-'  : Accelerate then decelerate (only one switch time; second is None).
                        'P-P+'  : Decelerate then accelerate (only one switch time; second is None).
                        'P+L+P-': Accelerate to vmax, cruise, then decelerate.
                        'P-L-P+': Decelerate to -vmax, cruise, then accelerate.
    - optimal_label: The label of the trajectory with minimal execution time among valid trajectories,
                     or None if no valid trajectory exists.
    """
    assert vmax > 1e-6, "vmax must be greater than 1e-6 for numerical stability."
    assert amax > 1e-6, "amax must be greater than 1e-6 for numerical stability."

    x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    # Trajectory class P+P-: Accelerate then decelerate.
    def compute_p_plus_p_minus():
        coeffs = [amax, 2 * v1, (v1**2 - v2**2) / (2 * amax) + x1 - x2]
        sols = solve_quadratic(*coeffs)
        valid_ts = [t for t in sols if max((v2 - v1) / amax, 0) <= t <= (vmax - v1) / amax]
        if not valid_ts:
            return None
        t_p = valid_ts[0]
        T = 2 * t_p + (v1 - v2) / amax
        # Only one switch time (end of acceleration phase)
        return (T, t_p, None)

    # Trajectory class P-P+: Decelerate then accelerate.
    def compute_p_minus_p_plus():
        coeffs = [amax, -2 * v1, (v1**2 - v2**2) / (2 * amax) + x2 - x1]
        sols = solve_quadratic(*coeffs)
        valid_ts = [t for t in sols if max((v1 - v2) / amax, 0) <= t <= (vmax + v1) / amax]
        if not valid_ts:
            return None
        t_p = valid_ts[0]
        T = 2 * t_p + (v2 - v1) / amax
        # Only one switch time (end of deceleration phase)
        return (T, t_p, None)

    # Trajectory class P+L+P-: Accelerate to vmax, cruise, then decelerate.
    def compute_p_plus_l_plus_p_minus():
        t_p1 = (vmax - v1) / amax
        t_p2 = (vmax - v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        T = t_p1 + t_l + t_p2
        # Two switch times: first at end of acceleration, second at end of cruise
        return (T, t_p1, t_p1 + t_l)

    # Trajectory class P-L-P+: Decelerate to -vmax, cruise, then accelerate.
    def compute_p_minus_l_plus_p_plus():
        t_p1 = (vmax + v1) / amax
        t_p2 = (vmax + v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) - (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        T = t_p1 + t_l + t_p2
        # Two switch times: first at end of deceleration, second at end of cruise
        return (T, t_p1, t_p1 + t_l)

    traj_funcs = {
        'P+P-': compute_p_plus_p_minus,
        'P-P+': compute_p_minus_p_plus,
        'P+L+P-': compute_p_plus_l_plus_p_minus,
        'P-L-P+': compute_p_minus_l_plus_p_plus,
    }

    trajectories = {}
    for label, func in traj_funcs.items():
        trajectories[label] = func()

    # Choose the trajectory with the minimum execution time among valid ones
    valid_trajs = {k: v for k, v in trajectories.items() if v is not None}
    if valid_trajs:
        optimal_label = min(valid_trajs, key=lambda k: valid_trajs[k][0])
    else:
        optimal_label = None

    return trajectories, optimal_label

def plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, amax, num_points=100):
    """
    Plot candidate trajectories using get_motion_state_at_local_t.
    The figure is divided into 4 outer subplots (one per candidate trajectory),
    and each outer subplot is subdivided into 2 inner subplots:
    the top for position vs. time and the bottom for velocity vs. time.
    
    Inputs:
    - trajectories: dict mapping candidate labels to (total_time, switch_time1, switch_time2),
                    or None if the candidate trajectory is invalid.
    - start_pos, end_pos: initial and final positions.
    - start_vel, end_vel: initial and final velocities.
    - vmax, amax: maximum velocity and acceleration.
    - num_points: number of sampling points for plotting.
    """
    candidate_order = ['P+P-', 'P-P+', 'P+L+P-', 'P-L-P+']
    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)
    
    for i, candidate in enumerate(candidate_order):
        # Create two inner subplots within the outer cell:
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.4)
        ax_pos = plt.Subplot(fig, inner[0])
        ax_vel = plt.Subplot(fig, inner[1])
        
        # Check if candidate trajectory is valid (i.e., has computed times)
        if trajectories.get(candidate) is not None:
            total_time, switch_time1, switch_time2 = trajectories[candidate]
            # Sample time points from 0 to total_time
            t_samples = np.linspace(0, total_time, num_points)
            # Compute states (position, velocity, acceleration) at each time using get_motion_state_at_local_t
            states = [get_motion_state_at_local_t(t, candidate, start_pos, start_vel, end_vel,
                                                    vmax, amax, switch_time1, switch_time2, total_time)
                      for t in t_samples]
            pos_samples = np.array([s[0] for s in states])
            vel_samples = np.array([s[1] for s in states])
            
            ax_pos.plot(t_samples, pos_samples, 'b-')
            ax_vel.plot(t_samples, vel_samples, 'r-')
            ax_pos.set_title(f"{candidate} (Total Time: {list(total_time)[0]:.2f} s)")
        else:
            # If candidate trajectory is invalid, display a message.
            ax_pos.set_title(f"{candidate} (None)")
        
        # Set common labels and grid for both subplots
        for ax in (ax_pos, ax_vel):
            ax.set_xlabel('Time [s]')
            ax.grid(True)
        ax_pos.set_ylabel('Position')
        ax_vel.set_ylabel('Velocity')
        
        # Add the subplots to the figure
        fig.add_subplot(ax_pos)
        fig.add_subplot(ax_vel)
    
    plt.show()

# ------------------ Testing and Plotting Code ------------------
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    
    # Sample random boundary conditions
    start_pos = np.random.uniform(-10, 10)
    end_pos = np.random.uniform(-10, 10)
    start_vel = np.random.uniform(-2, 2)
    end_vel = np.random.uniform(-2, 2)
    vmax = np.random.uniform(1, 3)
    amax = np.random.uniform(1, 3)

    # Examples
    vmax, amax = np.array([1.0]), np.array([1.0])
    # Examples 1, 2, 3, 4: Corresponding to Figure 5 in the original paper
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0])
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([3.0]), np.array([0.0]), np.array([0.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([-0.5]), np.array([1.0]), np.array([1.0]) 
    
    # More examples
    # Example 5: This example illustrates that for the P-L+P+ trajectory, just before accelerating with amax, 
    # the velocity must have reached -vmax
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]) 

    # Example 6, 7, 8, 9:
    # Examples 6 and 9 demonstrate a scenario where, if the distance is insufficient for acceleration, 
    # it must first decelerate backward before accelerating forward
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.6]), np.array([0.0]), np.array([1.0]) 
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.8])
    # start_pos, end_pos, start_vel, end_vel = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.2])

    # Compute candidate trajectories and select the optimal one using the previously defined function.
    trajectories, optimal_label = univariate_time_optimal_interpolants(
        start_pos, end_pos, start_vel, end_vel, vmax, amax
    )
    
    # Plot the optimal trajectory.
    plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, amax)
