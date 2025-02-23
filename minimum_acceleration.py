import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from solve_quadratic import solve_quadratic

def minimum_acceleration_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, T, amax, 
                                      t_margin=1e-5, a_margin=1e-6):
    """
    Compute the minimum-acceleration trajectory for a fixed end time T.

    Returns:
    - results: A dict mapping each candidate to its computed acceleration or None if invalid.
    - a_min: The minimal acceleration among valid candidates.
    - selected_primitive: Name of the candidate with minimal acceleration.
    """
    x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel
    
    # Candidate: P+P-
    def compute_p_plus_p_minus():
        coeffs = [T**2, 2*T*(v1+v2) + 4*(x1 - x2), -(v2 - v1)**2]
        sols = solve_quadratic(*coeffs)
        valid_a = []
        for a in sols:
            if a <= 0:
                continue
            t_s = 0.5*(T + (v2 - v1)/a)
            # Check if t_s is feasible and the velocity at t_s remains <= vmax
            if 0 < t_s < T + t_margin and abs(v1 + a*t_s) <= vmax:
                valid_a.append(a)
        return min(valid_a) if valid_a else None

    # Candidate: P-P+
    def compute_p_minus_p_plus():
        coeffs = [T**2, -2*T*(v1+v2) - 4*(x1 - x2), -(v2 - v1)**2]
        sols = solve_quadratic(*coeffs)
        valid_a = []
        for a in sols:
            if a <= 0:
                continue
            t_s = 0.5*(T + (v1 - v2)/a)
            if 0 < t_s < T + t_margin and abs(v1 - a*t_s) <= vmax:
                valid_a.append(a)
        return min(valid_a) if valid_a else None

    # Candidate: P+L+P-
    def compute_p_plus_l_plus_p_minus():
        a = (vmax**2 - vmax*(v1+v2) + 0.5*(v1**2 + v2**2)) / (T*vmax - (x2 - x1))
        if a <= 0:
            return None
        t_p1 = (vmax - v1)/a
        t_p2 = (vmax - v2)/a
        t_l = T - t_p1 - t_p2
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return a

    # Candidate: P-L-P+
    def compute_p_minus_l_minus_p_plus():
        a = (vmax**2 + vmax*(v1+v2) + 0.5*(v1**2 + v2**2)) / (T*vmax + (x2 - x1))
        if a <= 0:
            return None
        t_p1 = (vmax + v1)/a
        t_p2 = (vmax + v2)/a
        t_l = T - t_p1 - t_p2
        if t_p1 < 0 or t_p2 < 0 or t_l < 0:
            return None
        return a

    # Build a dictionary of results for each candidate
    results = {
        'P+P-': compute_p_plus_p_minus(),
        'P-P+': compute_p_minus_p_plus(),
        'P+L+P-': compute_p_plus_l_plus_p_minus(),
        'P-L-P+': compute_p_minus_l_minus_p_plus()
    }

    # Filter valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        # NOTE: This problem is possible to have no solution.
        return None

    # Select the candidate with minimal acceleration
    selected_primitive = min(valid_results, key=valid_results.get)
    a_min = valid_results[selected_primitive]

    # Check rated acceleration limit
    if a_min <= amax + a_margin:
        a_min = np.clip(a_min, 0, amax)
    else:
        raise ValueError("Required acceleration exceeds the rated maximum.")
    
    return results, a_min, selected_primitive

def plot_trajectory(results, start_pos, end_pos, start_vel, end_vel, vmax, T, num_points=100):
    """
    Plot all four candidate motion primitives in a 2x2 grid. 
    Each outer subplot corresponds to one candidate. 
    Each candidate subplot has two rows:
      - Top: position vs. time
      - Bottom: velocity vs. time.

    If a candidate is None in 'results', we display "No solution".
    Otherwise, we compute the trajectory with that candidate's acceleration.
    """
    x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

    # Helper: given a candidate label and an acceleration 'a', sample the trajectory
    def compute_candidate_segments(candidate_label, a):
        if candidate_label == 'P+P-':
            t_s = 0.5 * (T + (v2 - v1)/a)
            t1 = np.linspace(0, t_s, num_points)
            pos1 = x1 + v1*t1 + 0.5*a*t1**2
            vel1 = v1 + a*t1
            t2 = np.linspace(0, T - t_s, num_points)
            pos0 = x1 + v1*t_s + 0.5*a*t_s**2
            v_mid = v1 + a*t_s
            pos2 = pos0 + v_mid*t2 - 0.5*a*t2**2
            vel2 = v_mid - a*t2
            t_total = np.concatenate((t1, t2 + t_s))
            pos_total = np.concatenate((pos1, pos2))
            vel_total = np.concatenate((vel1, vel2))
            return t_total, pos_total, vel_total

        elif candidate_label == 'P-P+':
            t_s = 0.5 * (T + (v1 - v2)/a)
            t1 = np.linspace(0, t_s, num_points)
            pos1 = x1 + v1*t1 - 0.5*a*t1**2
            vel1 = v1 - a*t1
            t2 = np.linspace(0, T - t_s, num_points)
            pos0 = x1 + v1*t_s - 0.5*a*t_s**2
            v_mid = v1 - a*t_s
            pos2 = pos0 + v_mid*t2 + 0.5*a*t2**2
            vel2 = v_mid + a*t2
            t_total = np.concatenate((t1, t2 + t_s))
            pos_total = np.concatenate((pos1, pos2))
            vel_total = np.concatenate((vel1, vel2))
            return t_total, pos_total, vel_total

        elif candidate_label == 'P+L+P-':
            a = results['P+L+P-']  # we only call this if it's not None
            t_p1 = (vmax - v1)/a
            t_p2 = (vmax - v2)/a
            t_l = T - t_p1 - t_p2
            t1 = np.linspace(0, t_p1, num_points)
            pos1 = x1 + v1*t1 + 0.5*a*t1**2
            vel1 = v1 + a*t1
            t2 = np.linspace(0, t_l, num_points)
            pos0 = x1 + v1*t_p1 + 0.5*a*t_p1**2
            pos2 = pos0 + vmax*t2
            vel2 = np.full_like(t2, vmax)
            t3 = np.linspace(0, t_p2, num_points)
            pos_mid = pos0 + vmax*t_l
            pos3 = pos_mid + vmax*t3 - 0.5*a*t3**2
            vel3 = vmax - a*t3
            t_total = np.concatenate((t1, t2 + t_p1, t3 + t_p1 + t_l))
            pos_total = np.concatenate((pos1, pos2, pos3))
            vel_total = np.concatenate((vel1, vel2, vel3))
            return t_total, pos_total, vel_total

        elif candidate_label == 'P-L-P+':
            a = results['P-L-P+']
            t_p1 = (vmax + v1)/a
            t_p2 = (vmax + v2)/a
            t_l = T - t_p1 - t_p2
            t1 = np.linspace(0, t_p1, num_points)
            pos1 = x1 + v1*t1 - 0.5*a*t1**2
            vel1 = v1 - a*t1
            t2 = np.linspace(0, t_l, num_points)
            pos0 = x1 + v1*t_p1 - 0.5*a*t_p1**2
            pos2 = pos0 - vmax*t2
            vel2 = np.full_like(t2, -vmax)
            t3 = np.linspace(0, t_p2, num_points)
            pos_mid = pos0 - vmax*t_l
            pos3 = pos_mid - vmax*t3 + 0.5*a*t3**2
            vel3 = -vmax + a*t3
            t_total = np.concatenate((t1, t2 + t_p1, t3 + t_p1 + t_l))
            pos_total = np.concatenate((pos1, pos2, pos3))
            vel_total = np.concatenate((vel1, vel2, vel3))
            return t_total, pos_total, vel_total

        else:
            raise ValueError(f"Unknown candidate type: {candidate_label}")

    # Define the 4 candidates in a specific order
    candidate_order = ['P+P-', 'P-P+', 'P+L+P-', 'P-L-P+']

    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)

    for i, candidate in enumerate(candidate_order):
        # Subdivide each outer cell vertically into 2 inner subplots.
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.4)
        ax_pos = plt.Subplot(fig, inner[0])
        ax_vel = plt.Subplot(fig, inner[1])

        a_candidate = results.get(candidate, None)
        
        if a_candidate is not None:
            try:
                t_total, pos_total, vel_total = compute_candidate_segments(candidate, a_candidate)
                # Plot position
                ax_pos.plot(t_total, pos_total, 'b-')
                ax_pos.set_ylabel('Position')
                ax_pos.set_title(f"{candidate} (Min Accel: {a_candidate:.3f}m/s^2)")

                # Plot velocity
                ax_vel.plot(t_total, vel_total, 'r-')
                ax_vel.set_ylabel('Velocity')
            except Exception as e:
                # If an error arises during trajectory sampling
                ax_pos.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax_pos.set_title(candidate)
                ax_vel.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax_vel.set_title(candidate)
        else:
            ax_pos.set_title(f"{candidate} (None)")

        # Common settings
        ax_pos.set_xlabel('Time [s]')
        ax_pos.grid(True)
        ax_vel.set_xlabel('Time [s]')
        ax_vel.grid(True)

        # Add subplots to the figure
        fig.add_subplot(ax_pos)
        fig.add_subplot(ax_vel)

    plt.tight_layout()
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
    T = 9.070201678129566
    amax = 100.0

    # Compute candidate results, minimal acceleration, and selected candidate.
    results, a_min, selected_primitive = minimum_acceleration_interpolants(
        start_pos, end_pos, start_vel, end_vel, vmax, T, amax
    )
    
    print("Candidate Results:", results)
    print("Selected Primitive:", selected_primitive, "with a_min =", a_min)
    
    # Plot the trajectory based on the selected candidate.
    plot_trajectory(results, start_pos, end_pos, start_vel, end_vel, vmax, T)
