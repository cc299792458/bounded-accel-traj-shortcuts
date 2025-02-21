import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def univariate_time_optimal_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, amax):
    """
    Compute the execution times for all trajectory classes for univariate motion,
    and return the optimal trajectory.

    Inputs:
    - start_pos, end_pos: Initial and final positions.
    - start_vel, end_vel: Initial and final velocities.
    - vmax: Maximum velocity.
    - amax: Maximum acceleration.

    Returns:
    - trajectories: A dictionary mapping each trajectory class to its execution time (or None if invalid).
      The classes are:
         'P+P-'  : Accelerate then decelerate.
         'P-P+'  : Decelerate then accelerate.
         'P+L+P-': Accelerate to vmax, cruise, then decelerate.
         'P-L-P+': Decelerate to -vmax, cruise, then accelerate.
    - optimal_label: The label of the trajectory with minimal execution time (if any valid trajectory exists).
    - optimal_time: The minimal execution time among valid trajectories.
    """
    x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
    
    # Trajectory class P+P-
    def compute_p_plus_p_minus():
        coeffs = [amax, 2 * v1, (v1**2 - v2**2) / (2 * amax) + x1 - x2]
        sols = solve_quadratic(*coeffs)
        valid_ts = [t for t in sols if max((v2 - v1) / amax, 0) <= t <= (vmax - v1) / amax]
        if not valid_ts:
            return None
        t_p = valid_ts[0]
        return 2 * t_p + (v1 - v2) / amax

    # Trajectory class P-P+
    def compute_p_minus_p_plus():
        coeffs = [amax, -2 * v1, (v1**2 - v2**2) / (2 * amax) + x2 - x1]
        sols = solve_quadratic(*coeffs)
        valid_ts = [t for t in sols if max((v1 - v2) / amax, 0) <= t <= (vmax + v1) / amax]
        if not valid_ts:
            return None
        t_p = valid_ts[0]
        return 2 * t_p + (v2 - v1) / amax

    # Trajectory class P+L+P-
    def compute_p_plus_l_plus_p_minus():
        t_p1 = (vmax - v1) / amax
        t_p2 = (vmax - v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return t_p1 + t_l + t_p2

    # Trajectory class P-L-P+
    def compute_p_minus_l_plus_p_plus():
        t_p1 = (vmax + v1) / amax
        t_p2 = (vmax + v2) / amax
        t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) - (x2 - x1) / vmax
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return t_p1 + t_l + t_p2

    trajectories = {}
    trajectories['P+P-'] = compute_p_plus_p_minus()
    trajectories['P-P+'] = compute_p_minus_p_plus()
    trajectories['P+L+P-'] = compute_p_plus_l_plus_p_minus()
    trajectories['P-L-P+'] = compute_p_minus_l_plus_p_plus()

    # Determine the optimal (minimal) execution time among valid trajectories
    valid_trajs = {k: v for k, v in trajectories.items() if v is not None}
    if valid_trajs:
        optimal_label = min(valid_trajs, key=valid_trajs.get)
        optimal_time = valid_trajs[optimal_label]
    else:
        raise ValueError

    return trajectories, optimal_label, optimal_time

def plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, amax, num_points=100):
    """
    Plot candidate trajectories in one figure.
    The figure is divided into 4 outer subplots (one per candidate trajectory),
    and each outer subplot is subdivided into 2 inner subplots: 
    the top for position vs. time and the bottom for velocity vs. time.
    """
    def solve_quadratic(a, b, c):
        disc = b**2 - 4*a*c
        if disc < 0:
            return []
        sqrt_disc = np.sqrt(disc)
        return [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]
    
    def compute_candidate_segments(candidate_label):
        if candidate_label == 'P+P-':
            coeffs = (amax, 2 * start_vel, (start_vel**2 - end_vel**2) / (2 * amax) + start_pos - end_pos)
            sols = solve_quadratic(*coeffs)
            valid_ts = [t for t in sols if t >= max((end_vel - start_vel) / amax, 0) and t <= (vmax - start_vel) / amax]
            if not valid_ts:
                raise ValueError("No valid solution for P+P- trajectory")
            t_p = valid_ts[0]
            T1 = t_p  # Acceleration phase
            T2 = t_p + (start_vel - end_vel) / amax  # Deceleration phase
            t1 = np.linspace(0, T1, num_points)
            pos1 = start_pos + start_vel*t1 + 0.5*amax*t1**2
            vel1 = start_vel + amax*t1
            t2 = np.linspace(0, T2, num_points)
            pos0 = start_pos + start_vel*T1 + 0.5*amax*T1**2
            v0 = start_vel + amax*T1
            pos2 = pos0 + v0*t2 - 0.5*amax*t2**2
            vel2 = v0 - amax*t2
            t_seg2 = t2 + T1
            t_total = np.concatenate((t1, t_seg2))
            pos_total = np.concatenate((pos1, pos2))
            vel_total = np.concatenate((vel1, vel2))
            total_time = T1 + T2
            return t_total, pos_total, vel_total, total_time
        
        elif candidate_label == 'P-P+':
            coeffs = (amax, -2 * start_vel, (start_vel**2 - end_vel**2) / (2 * amax) + end_pos - start_pos)
            sols = solve_quadratic(*coeffs)
            valid_ts = [t for t in sols if t >= max((start_vel - end_vel) / amax, 0) and t <= (vmax + start_vel) / amax]
            if not valid_ts:
                raise ValueError("No valid solution for P-P+ trajectory")
            t_p = valid_ts[0]
            T1 = t_p  # Deceleration phase
            T2 = t_p + (end_vel - start_vel) / amax  # Acceleration phase
            t1 = np.linspace(0, T1, num_points)
            pos1 = start_pos + start_vel*t1 - 0.5*amax*t1**2
            vel1 = start_vel - amax*t1
            t2 = np.linspace(0, T2, num_points)
            pos0 = start_pos + start_vel*T1 - 0.5*amax*T1**2
            v0 = start_vel - amax*T1
            pos2 = pos0 + v0*t2 + 0.5*amax*t2**2
            vel2 = v0 + amax*t2
            t_seg2 = t2 + T1
            t_total = np.concatenate((t1, t_seg2))
            pos_total = np.concatenate((pos1, pos2))
            vel_total = np.concatenate((vel1, vel2))
            total_time = T1 + T2
            return t_total, pos_total, vel_total, total_time
        
        elif candidate_label == 'P+L+P-':
            t_p1 = (vmax - start_vel) / amax
            t_p2 = (vmax - end_vel) / amax
            t_l = (end_vel**2 + start_vel**2 - 2*vmax**2) / (2*vmax*amax) + (end_pos - start_pos) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                raise ValueError("Invalid time segments for P+L+P- trajectory")
            t1 = np.linspace(0, t_p1, num_points)
            pos1 = start_pos + start_vel*t1 + 0.5*amax*t1**2
            vel1 = start_vel + amax*t1
            t2 = np.linspace(0, t_l, num_points)
            pos0 = start_pos + start_vel*t_p1 + 0.5*amax*t_p1**2
            pos2 = pos0 + vmax*t2
            vel2 = np.full_like(t2, vmax)
            t3 = np.linspace(0, t_p2, num_points)
            pos_mid = pos0 + vmax*t_l
            v_mid = vmax
            pos3 = pos_mid + v_mid*t3 - 0.5*amax*t3**2
            vel3 = v_mid - amax*t3
            t_seg2 = t2 + t_p1
            t_seg3 = t3 + t_p1 + t_l
            t_total = np.concatenate((t1, t_seg2, t_seg3))
            pos_total = np.concatenate((pos1, pos2, pos3))
            vel_total = np.concatenate((vel1, vel2, vel3))
            total_time = t_p1 + t_l + t_p2
            return t_total, pos_total, vel_total, total_time
        
        elif candidate_label == 'P-L-P+':
            t_p1 = (vmax + start_vel) / amax
            t_p2 = (vmax + end_vel) / amax
            t_l = (end_vel**2 + start_vel**2 - 2*vmax**2) / (2*vmax*amax) - (end_pos - start_pos) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                raise ValueError("Invalid time segments for P-L-P+ trajectory")
            t1 = np.linspace(0, t_p1, num_points)
            pos1 = start_pos + start_vel*t1 - 0.5*amax*t1**2
            vel1 = start_vel - amax*t1
            t2 = np.linspace(0, t_l, num_points)
            pos0 = start_pos + start_vel*t_p1 - 0.5*amax*t_p1**2
            pos2 = pos0 - vmax*t2
            vel2 = np.full_like(t2, -vmax)
            t3 = np.linspace(0, t_p2, num_points)
            pos_mid = pos0 - vmax*t_l
            v_mid = -vmax
            pos3 = pos_mid + v_mid*t3 + 0.5*amax*t3**2
            vel3 = v_mid + amax*t3
            t_seg2 = t2 + t_p1
            t_seg3 = t3 + t_p1 + t_l
            t_total = np.concatenate((t1, t_seg2, t_seg3))
            pos_total = np.concatenate((pos1, pos2, pos3))
            vel_total = np.concatenate((vel1, vel2, vel3))
            total_time = t_p1 + t_l + t_p2
            return t_total, pos_total, vel_total, total_time
        
        else:
            raise ValueError("Unknown trajectory type")
    
    candidate_order = ['P+P-', 'P-P+', 'P+L+P-', 'P-L-P+']
    
    # Create a figure with a larger size and more space between subplots.
    fig = plt.figure(figsize=(16, 10))  # Widen the figure, for example
    outer = gridspec.GridSpec(
        2, 2, 
        wspace=0.4,  # Increase horizontal space
        hspace=0.4   # Increase vertical space
    )
    
    for i, candidate in enumerate(candidate_order):
        # Subdivide each outer cell vertically into 2 inner subplots.
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, 
            subplot_spec=outer[i], 
            hspace=0.4  # Increase space between position and velocity subplots
        )
        ax_pos = plt.Subplot(fig, inner[0])
        ax_vel = plt.Subplot(fig, inner[1])
        
        if trajectories.get(candidate) is not None:
            try:
                t_total, pos_total, vel_total, total_time = compute_candidate_segments(candidate)
                
                # Top subplot: position vs time
                ax_pos.plot(t_total, pos_total, 'b-')
                ax_pos.set_ylabel('Position')
                title_text = f"{candidate} (Total Time: {float(total_time):.2f} s)"
                ax_pos.set_title(title_text)
                
                # Bottom subplot: velocity vs time
                ax_vel.plot(t_total, vel_total, 'r-')
                ax_vel.set_ylabel('Velocity')
            except Exception as e:
                ax_pos.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax_pos.set_title(candidate)
                ax_vel.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax_vel.set_title(candidate)
        else:
            # If None, just show empty subplots with a title
            ax_pos.set_title(f"{candidate} (None)")
            ax_pos.set_ylabel('Position')
            ax_vel.set_ylabel('Velocity')
        
        # Common settings for both subplots
        ax_pos.set_xlabel('Time [s]')
        ax_pos.grid(True)
        ax_vel.set_xlabel('Time [s]')
        ax_vel.grid(True)
        
        # Add the two subplots to the figure
        fig.add_subplot(ax_pos)
        fig.add_subplot(ax_vel)
    
    plt.tight_layout()
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
    # vmax, amax = np.array([1.0]), np.array([1.0])
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
    trajectories, optimal_label, optimal_time = univariate_time_optimal_interpolants(
        start_pos, end_pos, start_vel, end_vel, vmax, amax
    )
    
    # Plot the optimal trajectory.
    plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, amax)
