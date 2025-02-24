import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
from get_motion_state import get_motion_state_at_local_t

def plot_trajectory(trajectories, start_pos, end_pos, start_vel, end_vel, vmax, amax=None, T=None, num_points=100, solution_type='time_optimal'):
    """
    Plot candidate motion primitives in a 2x2 grid.
    Each candidate subplot is subdivided into two rows:
      - Top: position vs. time
      - Bottom: velocity vs. time.
      
    Uses get_motion_state_at_local_t to compute the state at each time sample.
    
    Parameters:
      - trajectories: dict mapping candidate labels to a tuple:
          * For solution_type 'time_optimal': (total_time, switch_time1, switch_time2)
          * For solution_type 'min_accel': (acceleration, switch_time1, switch_time2)
        If a candidate is invalid, its value is None.
      - start_pos, end_pos: initial and final positions.
      - start_vel, end_vel: initial and final velocities.
      - vmax: maximum velocity.
      - amax: maximum acceleration.
      - T: overall trajectory time when using 'min_accel' solution; for 'time_optimal', the total time is taken from the candidate.
      - num_points: number of sampling points for plotting.
      - solution_type: either 'time_optimal' or 'min_accel' to indicate the type of candidate solution.
    """

    candidate_order = ['P+P-', 'P-P+', 'P+L+P-', 'P-L-P+']
    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.4)
    
    for i, candidate in enumerate(candidate_order):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i], hspace=0.4)
        ax_pos = plt.Subplot(fig, inner[0])
        ax_vel = plt.Subplot(fig, inner[1])
        
        if trajectories.get(candidate) is not None:
            if solution_type == 'time_optimal':
                # Candidate solution: (total_time, switch_time1, switch_time2)
                total_time, switch_time1, switch_time2 = trajectories[candidate]
                t_samples = np.linspace(0, total_time, num_points)
                # Use amax as the acceleration parameter in get_motion_state_at_local_t
                states = [get_motion_state_at_local_t(t, candidate, start_pos, start_vel, end_vel,
                                                        vmax, amax, switch_time1, switch_time2, total_time)
                          for t in t_samples]
                title_text = f"{candidate} (Total Time: {list(total_time)[0]:.2f} s)"
            elif solution_type == 'min_accel':
                # Candidate solution: (acceleration, switch_time1, switch_time2)
                acc, switch_time1, switch_time2 = trajectories[candidate]
                t_samples = np.linspace(0, T, num_points)
                # Use candidate's acceleration as the effective acceleration in the state function, total time is T
                states = [get_motion_state_at_local_t(t, candidate, start_pos, start_vel, end_vel,
                                                        vmax, acc, switch_time1, switch_time2, T)
                          for t in t_samples]
                title_text = f"{candidate} (Min Accel: {list(acc)[0]:.3f} m/s^2)"
            else:
                raise ValueError("Unknown solution_type. Use 'time_optimal' or 'min_accel'.")
            
            pos_samples = np.array([s[0] for s in states])
            vel_samples = np.array([s[1] for s in states])
            ax_pos.plot(t_samples, pos_samples, 'b-')
            ax_vel.plot(t_samples, vel_samples, 'r-')
            ax_pos.set_title(title_text)
        else:
            ax_pos.set_title(f"{candidate} (None)")
        
        # Set common labels and grid for both subplots
        for ax in (ax_pos, ax_vel):
            ax.set_xlabel('Time [s]')
            ax.grid(True)
        ax_pos.set_ylabel('Position')
        ax_vel.set_ylabel('Velocity')
        
        fig.add_subplot(ax_pos)
        fig.add_subplot(ax_vel)
    
    plt.show()
