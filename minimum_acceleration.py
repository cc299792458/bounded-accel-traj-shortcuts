import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from solve_quadratic import solve_quadratic
from get_motion_state import get_motion_state_at_local_t

def minimum_acceleration_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold, 
                                      t_margin=1e-5, a_margin=1e-6):
    """
    Compute the minimum-acceleration trajectory for a fixed end time T.
    
    Returns:
    - trajectories: A dict mapping each candidate to a tuple 
                    (acceleration, switch_time1, switch_time2) if valid, or None if invalid.
                    The candidates are:
                        'P+P-': (a, t_s, None) where t_s is the switching time.
                        'P-P+': (a, t_s, None)
                        'P+L+P-': (a, switch_time1, switch_time2)
                        'P-L-P+': (a, switch_time1, switch_time2)
    - optimal_label: Name of the candidate with minimal acceleration among valid trajectories.
    """
    x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel
    
    # Candidate: P+P-
    def compute_p_plus_p_minus():
        coeffs = [T**2, 2 * T * (v1 + v2) + 4 * (x1 - x2), -(v2 - v1)**2]
        sols = solve_quadratic(*coeffs)
        valid_candidates = []
        for a in sols:
            if a <= 0:
                continue
            t_s = 0.5 * (T + (v2 - v1) / a)
            # Check feasibility and velocity limit at the switching time.
            if 0 < t_s < T + t_margin and abs(v1 + a * t_s) <= vmax:
                valid_candidates.append((a, t_s, None))
        return min(valid_candidates, key=lambda x: x[0]) if valid_candidates else None

    # Candidate: P-P+
    def compute_p_minus_p_plus():
        coeffs = [T**2, -2 * T * (v1 + v2) - 4 * (x1 - x2), -(v2 - v1)**2]
        sols = solve_quadratic(*coeffs)
        valid_candidates = []
        for a in sols:
            if a <= 0:
                continue
            t_s = 0.5 * (T + (v1 - v2) / a)
            if 0 < t_s < T + t_margin and abs(v1 - a * t_s) <= vmax:
                valid_candidates.append((a, t_s, None))
        return min(valid_candidates, key=lambda x: x[0]) if valid_candidates else None

    # Candidate: P+L+P-
    def compute_p_plus_l_plus_p_minus():
        a = (vmax**2 - vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax - (x2 - x1))
        if a <= 0:
            return None
        t_p1 = (vmax - v1) / a
        t_p2 = (vmax - v2) / a
        t_l = T - t_p1 - t_p2
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return (a, t_p1, t_p1 + t_l)

    # Candidate: P-L-P+
    def compute_p_minus_l_plus_p_plus():
        a = (vmax**2 + vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax + (x2 - x1))
        if a <= 0:
            return None
        t_p1 = (vmax + v1) / a
        t_p2 = (vmax + v2) / a
        t_l = T - t_p1 - t_p2
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return (a, t_p1, t_p1 + t_l)

    trajectories = {
        'P+P-': compute_p_plus_p_minus(),
        'P-P+': compute_p_minus_p_plus(),
        'P+L+P-': compute_p_plus_l_plus_p_minus(),
        'P-L-P+': compute_p_minus_l_plus_p_plus()
    }
    
    valid_trajectories = {k: v for k, v in trajectories.items() if v is not None}
    if not valid_trajectories:
        # NOTE: This problem is possible to have no solution.
        return None

    optimal_label = min(valid_trajectories, key=lambda k: valid_trajectories[k][0])
    a_min = valid_trajectories[optimal_label][0]
    
    if a_min <= a_threshold + a_margin:
        # Optionally clip acceleration to a_threshold if within the margin.
        a_min = np.clip(a_min, 0, a_threshold)
    else:
        raise ValueError("Required acceleration exceeds the rated maximum.")
    
    return trajectories, optimal_label

def plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, T, num_points=100):
    """
    Plot all four candidate motion primitives in a 2x2 grid.
    Each candidate subplot has two rows:
      - Top: position vs. time
      - Bottom: velocity vs. time.
      
    Uses get_motion_state_at_local_t to compute the state at each time sample.
    
    Inputs:
      - trajectories: dict mapping candidate labels to a tuple 
                 (acceleration, switch_time1, switch_time2) if valid, or None if invalid.
      - start_pos, end_pos: initial and final positions.
      - start_vel, end_vel: initial and final velocities.
      - vmax: maximum velocity.
      - T: total trajectory time.
      - num_points: number of samples for plotting.
    """
    candidate_order = ['P+P-', 'P-P+', 'P+L+P-', 'P-L-P+']
    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)
    
    for i, candidate in enumerate(candidate_order):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.4)
        ax_pos = plt.Subplot(fig, inner[0])
        ax_vel = plt.Subplot(fig, inner[1])
        
        candidate_result = trajectories.get(candidate, None)
        if candidate_result is not None:
            # Unpack the candidate solution: acceleration, switch_time1, and switch_time2
            acc, sw1, sw2 = candidate_result
            # Sample time from 0 to T
            t_samples = np.linspace(0, T, num_points)
            # Compute states using get_motion_state_at_local_t for each time sample
            states = [get_motion_state_at_local_t(t, candidate, start_pos, start_vel, end_vel, vmax, acc, sw1, sw2, T)
                      for t in t_samples]
            pos_samples = np.array([s[0] for s in states])
            vel_samples = np.array([s[1] for s in states])
            
            ax_pos.plot(t_samples, pos_samples, 'b-')
            ax_vel.plot(t_samples, vel_samples, 'r-')
            ax_pos.set_title(f"{candidate} (Min Accel: {list(acc)[0]:.3f} m/s^2)")
        else:
            ax_pos.text(0.5, 0.5, "No solution", ha='center', va='center')
            ax_pos.set_title(f"{candidate} (None)")
        
        # Set common labels and grid
        for ax in (ax_pos, ax_vel):
            ax.set_xlabel('Time [s]')
            ax.grid(True)
        ax_pos.set_ylabel('Position')
        ax_vel.set_ylabel('Velocity')
        
        fig.add_subplot(ax_pos)
        fig.add_subplot(ax_vel)
    
    plt.show()

# ------------------ Testing and Main Code ------------------
if __name__ == '__main__':
    np.random.seed(42)
    
    # Generate random boundary conditions.
    start_pos = np.random.uniform(-10, 10)
    end_pos = np.random.uniform(-10, 10)
    start_vel = np.random.uniform(-2, 2)
    end_vel = np.random.uniform(-2, 2)
    vmax = np.random.uniform(1, 3)
    T = np.random.uniform(1, 10)

    # Examples
    # Examples 1, 2, 3, 4: Corresponding to Figure 5 in the original paper
    vmax, a_threshold = np.array([1.0]), np.array([100.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([2.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([3.0]), np.array([0.0]), np.array([0.0]), np.array([4.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([2.41])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([-0.5]), np.array([1.0]), np.array([1.0]), np.array([4.5])
    
    # More examples
    # Example 5: This example illustrates that for the P-L+P+ trajectory, just before accelerating with amax, 
    # the velocity must have reached -vmax
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]), np.array([1.0]) 

    # Example 6, 7, 8, 9:
    # Examples 6 and 9 demonstrate a scenario where, if the distance is insufficient for acceleration, 
    # it must first decelerate backward before accelerating forward
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]), np.array([1.63])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.6]), np.array([0.0]), np.array([1.0]), np.array([1.1]) 
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.8]), np.array([1.01])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.2]), np.array([2.14])

    # Compute candidate trajectories, minimal acceleration, and selected candidate.
    trajectories, optimal_label = minimum_acceleration_interpolants(
        start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold
    )
    
    # Plot the trajectory based on the selected candidate.
    plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, T)
