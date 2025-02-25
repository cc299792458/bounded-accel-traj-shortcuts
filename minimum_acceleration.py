import numpy as np

from plot import plot_trajectory
from solve_quadratic import solve_quadratic

def minimum_acceleration_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold):
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
            if 0 <= t_s <= T and abs(v1 + a * t_s) <= vmax: # NOTE: t_margin has been removed here.
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
            if 0 <= t_s <= T and abs(v1 - a * t_s) <= vmax: # NOTE: t_margin has been removed here.
                valid_candidates.append((a, t_s, None))
        return min(valid_candidates, key=lambda x: x[0]) if valid_candidates else None

    # Candidate: P+L+P-
    def compute_p_plus_l_plus_p_minus():
        if T * vmax - (x2 - x1) == 0:
            return (np.array([0.0]), np.array([0.0]), T) if v1 == v2 == vmax else None
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
        if T * vmax + (x2 - x1) == 0:
            return (np.array([0.0]), np.array([0.0]), T) if v1 == v2 == -vmax else None
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
        return None, None

    optimal_label = min(valid_trajectories, key=lambda k: valid_trajectories[k][0])
    amin = valid_trajectories[optimal_label][0]
    
    if amin * 0.999 <= a_threshold:
        # Optionally clip acceleration to a_threshold if within the margin.
        amin = np.clip(amin, 0, a_threshold)
        trajectories[optimal_label] = (amin, trajectories[optimal_label][1], trajectories[optimal_label][2])
    else:
        # NOTE: The minimum required acceleration can exceed the maximum available acceleration.
        return None, None
        
    return trajectories, optimal_label

# ------------------ Testing and Main Code ------------------
if __name__ == '__main__':
    np.random.seed(42)
    
    a_threshold = np.array([100.0])

    # Generate random boundary conditions.
    start_pos = np.random.uniform(-10, 10)
    end_pos = np.random.uniform(-10, 10)
    start_vel = np.random.uniform(-2, 2)
    end_vel = np.random.uniform(-2, 2)
    vmax = np.random.uniform(2, 4)
    T = np.random.uniform(1, 10)

    # Examples
    # Examples 1, 2, 3, 4: Corresponding to Figure 5 in the original paper
    vmax = np.array([1.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([2.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([3.0]), np.array([0.0]), np.array([0.0]), np.array([4.0])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([2.41])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([-0.5]), np.array([1.0]), np.array([1.0]), np.array([4.5])
    
    # More examples
    # Example 5: This example illustrates that for the P-L+P+ trajectory, just before accelerating with amax, 
    # the velocity must have reached -vmax
    start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.0]), np.array([1.0]) 

    # Example 6, 7, 8, 9:
    # Examples 6 and 9 demonstrate a scenario where, if the distance is insufficient for acceleration, 
    # it must first decelerate backward before accelerating forward
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]), np.array([1.63])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.6]), np.array([0.0]), np.array([1.0]), np.array([1.1]) 
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([0.8]), np.array([1.01])
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([0.5]), np.array([0.0]), np.array([1.2]), np.array([2.14])

    # Example 10:
    # This is a corner case.
    # start_pos, end_pos, start_vel, end_vel, T = np.array([0.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])

    # Compute candidate trajectories, minimal acceleration, and selected candidate.
    trajectories, optimal_label = minimum_acceleration_interpolants(
        start_pos, end_pos, start_vel, end_vel, vmax, T, a_threshold
    )
    
    # Plot the trajectory based on the selected candidate.
    plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, T=T, solution_type='min_accel')
