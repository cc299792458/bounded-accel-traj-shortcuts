"""
This implementation is based on the method presented in the paper:
"Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts"
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse, Rectangle
from get_motion_state import get_motion_state_at_local_t
from minimum_acceleration import minimum_acceleration_interpolants
from univariate_time_optimal import univariate_time_optimal_interpolants

class Smoother:
    """
    Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts.
    This class implements a smoothing algorithm for manipulator trajectories with bounded velocity 
    and acceleration, using optimal shortcuts for improved performance and natural-looking motion.
    """
    
    def __init__(self, path, vmax, amax, collision_checker, max_iterations=100, obstacles=None):
        """
        Initialize the Smoother class.

        Parameters:
        - path: List of waypoints np.array([position, velocity])
        - vmax: Maximum velocity for each dimension.
        - amax: Maximum acceleration for each dimension.
        - collision_checker: Function to check for collisions.
        - max_iterations: Maximum number of shortcut iterations.
        """
        self.path = path
        self.vmax = vmax
        self.amax = amax
        self.dimension = self.vmax.shape[0]
        self.collision_checker = collision_checker
        self.max_iterations = max_iterations
        self.traj_segment_times = np.array([])  # Array of time durations for each segment
        self.traj_segment_params = []  # List of trajectories parameters for each segment
        self.obstacles = obstacles
        self.total_time = []

    def smooth_path(self, plot_traj=False):
        """
        Smooth the traj using time-optimal segments and shortcuts.

        Returns:
        - Updated path as a numpy array of waypoints [(position, velocity)].
        """
        # The algorithm fails if the initial step fails
        if not self.generate_initial_traj():
            return None, None, None

        for iteration in range(self.max_iterations):
            total_time = np.sum(self.traj_segment_times)
            self.total_time.append(total_time)
            t1, t2 = self.select_random_times(total_time)
            shortcut_start = self.get_motion_states_at_global_t(t1)
            shortcut_end = self.get_motion_states_at_global_t(t2)
            shortcut_traj_time, shortcut_traj_param = self.compute_traj_segment(shortcut_start, shortcut_end)
            if plot_traj:
                self.plot_traj(iteration, shortcut_start=shortcut_start, shortcut_end=shortcut_end, 
                                   candidate_shortcut_time=shortcut_traj_time, candidate_shortcut_param=shortcut_traj_param)
            # Only update if the traj exists
            if shortcut_traj_param is not None:
                # Plot again if update segment data successfully
                if self.update_segment_data(shortcut_start, shortcut_end, t1, t2, shortcut_traj_time, shortcut_traj_param):
                    if plot_traj:
                        self.plot_traj(iteration, shortcut_start=shortcut_start, shortcut_end=shortcut_end)
            # Plot the final trajectory
            if plot_traj:
                self.plot_traj(iteration, shortcut_start=shortcut_start, shortcut_end=shortcut_end) 

        return self.path, self.traj_segment_times, self.traj_segment_params
    
    def generate_initial_traj(self):
        """Generate the initial time-optimal traj for all segments."""
        self.traj_segment_times = []
        self.traj_segment_params = []
        
        for i in range(self.path.shape[0] - 1):
            start_state, end_state = self.path[i], self.path[i + 1]
            traj_segment_time, traj_segment_param = self.compute_traj_segment(start_state, end_state)
            if traj_segment_param is None:
                return False
            self.traj_segment_times.append(traj_segment_time)
            self.traj_segment_params.append(traj_segment_param)

        self.traj_segment_times = np.array(self.traj_segment_times)

        return True
    
    def compute_traj_segment(self, start_state, end_state):
        traj_segment_time = self.compute_traj_segment_time(start_state, end_state)
        traj_segment_param = self.compute_traj_segment_param(start_state, end_state, traj_segment_time)
        
        return traj_segment_time, traj_segment_param

    def compute_traj_segment_time(self, start_state, end_state):
        """
        Calculate the maximum time required to traverse a segment across all dimensions,
        considering vmax and amax constraints.
        """
        t_requireds = []
        for dim in range(self.dimension):
            trajectories, optimal_label = univariate_time_optimal_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                amax=self.amax[dim]
            )
            t_requireds.append(trajectories[optimal_label][0])

        return max(t_requireds)

    def compute_traj_segment_param(self, start_state, end_state, traj_segment_time):
        """
        Calculate the traj for a single segment using minimum acceleration interpolants.
        """
        # Vectorized calculation for all dimensions
        traj_params = []
        for dim in range(self.dimension):
            trajectories, optimal_label = minimum_acceleration_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                T=traj_segment_time,
                a_threshold=self.amax[dim]
            )
            if trajectories is None:
                # NOTE: This is actually possible. See consistency_validation.py for an example.
                return None
            traj_params.append((trajectories[optimal_label], optimal_label))

        return np.array(traj_params, dtype=object)
    
    def select_random_times(self, total_time, min_time_interval=0.01):
        """
        Select two random times within the total traj duration.

        Input:
        - total_time: The total duration of the traj
        - min_time_interval: Minimal time interval between t1 and t2

        Return:
        - t1, t2: Two random times within the total traj duration.
        """
        t1 = t2 = 0
        while t2 - t1 < min_time_interval:
            random_times = np.random.uniform(0, total_time, 2)
            t1, t2 = np.sort(random_times)
        return t1, t2

    def get_motion_states_at_global_t(self, t):
        """
        Find the interpolated state (position, velocity) at a specific time t within the traj.

        Input:
        - t: The target time

        Return:
        - position: Numpy array of positions at time t.
        - velocity: Numpy array of velocities at time t.
        """
        elapsed_time = 0

        for i in range(self.traj_segment_times.shape[0]):  # Use shape[0] since segment_time is a NumPy array
            traj_segment_time = self.traj_segment_times[i]
            if elapsed_time + traj_segment_time >= t:
                # Relative time within the current segment
                relative_t = t - elapsed_time
                start_state = self.path[i]

                # Compute and return the interpolated state
                return self.get_motion_states_at_local_t(start_state=start_state, traj_segment_param=self.traj_segment_params[i], t=relative_t)

            elapsed_time += traj_segment_time

        # If t is beyond the total traj duration. # NOTE: This could happen at the boundary due to the numerical precision issues
        return None

    def get_motion_states_at_local_t(self, start_state, traj_segment_param, t):
        """
        Compute the state (position, velocity) within a segment at time t.

        Input:
        - start_state: The starting state of the segment.
        - traj_segment_param: traj parameters for each dimension.
        - t: Relative time within the segment.

        Return:
        - position: Numpy array of positions at time t.
        - velocity: Numpy array of velocities at time t.
        """
        position = np.zeros(self.dimension)
        velocity = np.zeros(self.dimension)

        for dim in range(self.dimension):
            (amax, switch_time1, switch_time2), traj_type = traj_segment_param[dim]
            start_pos, start_vel = start_state[0][dim], start_state[1][dim]

            pos, vel, acc = get_motion_state_at_local_t(
                t=t, traj_type=traj_type, start_pos=start_pos, start_vel=start_vel, 
                amax=amax, switch_time1=switch_time1, switch_time2=switch_time2
            )
            position[dim], velocity[dim] = pos, vel

        return position, velocity

    def update_segment_data(self, start_state, end_state, t1, t2, shortcut_traj_time, shortcut_traj_param):
        """
        Update the traj and path by replacing the section between t1 and t2 with a shortcut.
        Also inserts connecting segments to ensure continuity.
        Collision checking and time reduction checking are performed at this step 
        to ensure the validity and efficiency of the connection traj.

        Input:
        - start_state, end_state: States at t1 and t2 respectively.
        - t1, t2: Start and end times of the shortcut in the original traj.
        - shortcut_traj_time: Duration of the shortcut segment.
        - shortcut_traj_param: Trajectory params for the shortcut segment.
        """
        # Find the indices of segments affected by t1 and t2
        elapsed_time = 0
        start_index, end_index = None, None
        for i, traj_segment_time in enumerate(self.traj_segment_times):
            if elapsed_time <= t1 < elapsed_time + traj_segment_time:
                start_index = i
            if elapsed_time <= t2 <= elapsed_time + traj_segment_time:
                end_index = i
            elapsed_time += traj_segment_time

        if start_index is None or end_index is None:
            raise ValueError("Invalid times t1 or t2, cannot find affected segments.")

        # Get total time of affected segments between start_index and end_index
        middle_segment_times = self.traj_segment_times[start_index:end_index+1]
        total_middle_time = np.sum(middle_segment_times)

        # Locate the start and end nodes for connection
        prev_state = self.path[start_index]  # Previous node before t1
        connect_time_before, connect_param_before = self.compute_traj_segment(prev_state, start_state)

        next_state = self.path[end_index + 1]  # Next node after t2
        connect_time_after, connect_param_after = self.compute_traj_segment(end_state, next_state)

        # Cancel update if the connection traj is invalid, does not reduce total time, or is not collision-free
        if connect_param_before is None or connect_param_after is None:
            return False
        if total_middle_time < connect_time_before + shortcut_traj_time + connect_time_after:
            return False
        if not self.is_segment_collision_free(start_state, shortcut_traj_time, shortcut_traj_param):
            return False
        if not self.is_segment_collision_free(prev_state, connect_time_before, connect_param_before):
            return False
        if not self.is_segment_collision_free(end_state, connect_time_after, connect_param_after):
            return False

        # Update path, segment_time and segment_traj using np.concatenate
        self.path = np.concatenate([
            self.path[:start_index + 1],
            [start_state, end_state],
            self.path[end_index + 1:]
        ])

        before_time = self.traj_segment_times[:start_index]
        after_time = self.traj_segment_times[end_index + 1:]
        self.traj_segment_times = np.concatenate([
            before_time,
            [connect_time_before, shortcut_traj_time, connect_time_after],
            after_time
        ])

        before_traj = self.traj_segment_params[:start_index]
        after_traj = self.traj_segment_params[end_index + 1:]
        self.traj_segment_params = before_traj + [
            connect_param_before,
            shortcut_traj_param,
            connect_param_after
        ] + after_traj

        return True
    
    def is_segment_collision_free(self, start_state, traj_segment_time, traj_segment_param, time_step=0.01):
        # Generate time points to sample along the traj
        num_samples = int(traj_segment_time / time_step) + 1
        sampled_times = np.linspace(0, traj_segment_time, num_samples)

        for time in sampled_times:
            state = self.get_motion_states_at_local_t(start_state=start_state, traj_segment_param=traj_segment_param, t=time)
            if not self.collision_checker(state):
                return False
        return True
    
    def plot_traj(self, iteration: int, shortcut_start: tuple, shortcut_end: tuple, 
                  candidate_shortcut_time=None, candidate_shortcut_param=None):
        """
        Plot the current trajectory of the smoother object in 2D with animation during smoothing.

        This method dynamically updates the trajectory plot for visualization, without creating new windows. 
        It provides a real-time view of the smoothing process. Optionally, it displays the current iteration number 
        and visualizes shortcut points (including candidate shortcuts) if provided.

        Args:
            iteration (int): The current iteration number to display on the plot.
            shortcut_start (tuple): A tuple representing the start point of the shortcut (x, y).
            shortcut_end (tuple): A tuple representing the end point of the shortcut (x, y).
            candidate_shortcut_time (float, optional): The time duration for the candidate shortcut to be visualized.
            candidate_shortcut_param (optional): Parameters defining the candidate shortcut, if applicable.

        Raises:
            ValueError: If the dimension of the trajectory is not 2D.
        """
        if self.dimension != 2:
            raise ValueError("This plotting function only supports 2D trajectories.")

        # Initialize plot only on the first call
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(8, 6))
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")
            self._ax.set_title("2D traj Smoothing")
            self._ax.grid(True)
            self._ax.axis("equal")

            # Add obstacles if exists
            if self.obstacles:
                self._obstacle_label_added = getattr(self, "_obstacle_label_added", False)
                for obs in self.obstacles:
                    if obs[0] == "ellipse":
                        _, center, rx, ry = obs
                        self._ax.add_patch(Ellipse(xy=center, width=2*rx, height=2*ry,
                                                edgecolor='r', facecolor='gray', alpha=0.5, 
                                                label="Obstacle" if not self._obstacle_label_added else ""))
                    elif obs[0] == "rectangle":
                        _, center, width, height = obs
                        self._ax.add_patch(Rectangle((center[0]-width/2, center[1]-height/2), width, height,
                                                    edgecolor='b', facecolor='lightblue', alpha=0.5, 
                                                    label="Obstacle" if not self._obstacle_label_added else ""))
                    self._obstacle_label_added = True

            # Plot initial and smoothed traj
            initial_positions = np.array([state[0] for state in self.path])
            self._ax.plot(initial_positions[:, 0], initial_positions[:, 1], 'y--', label='Initial traj')
            self._traj_line, = self._ax.plot([], [], '-o', markersize=2, label='Smoothed traj')

            # Plot milestones and shortcut points and optinally candidate shortcut
            self._milestones, = self._ax.plot([], [], 'ro', markersize=8, label='Milestones')
            self._shortcut_points, = self._ax.plot([], [], 'yo', markersize=4, label='Shortcut Points')
            if candidate_shortcut_time is not None and candidate_shortcut_param is not None:
                self._candidate_shortcut, = self._ax.plot([], [], '-yo', markersize=1, label='Candidate Shortcut')    

            # Add legend and iteration text
            self._ax.legend(loc="upper left")
            self._iteration_text = self._ax.text(0.95, 0.95, "", transform=self._ax.transAxes, 
                                                fontsize=12, color="blue", ha="right", va="top")

        # Update plot
        times = np.linspace(0, np.sum(self.traj_segment_times), 500)
        positions = np.array([self.get_motion_states_at_global_t(t)[0] for t in times if self.get_motion_states_at_global_t(t) is not None])
        self._traj_line.set_data(positions[:, 0], positions[:, 1])

        # Update milestones and shortcut points and optinally candidate shortcut
        initial_positions = np.array([state[0] for state in self.path])
        self._milestones.set_data(initial_positions[:, 0], initial_positions[:, 1])
        self._shortcut_points.set_data([shortcut_start[0][0], shortcut_end[0][0]], [shortcut_start[0][1], shortcut_end[0][1]])
        
        if candidate_shortcut_time is not None and candidate_shortcut_param is not None:
            times = np.linspace(0, candidate_shortcut_time, 20)
            positions = np.array([self.get_motion_states_at_local_t(shortcut_start, candidate_shortcut_param, t)[0] for t in times 
                                  if self.get_motion_states_at_local_t(shortcut_start, candidate_shortcut_param, t)[0] is not None])
            self._candidate_shortcut.set_data(positions[:, 0], positions[:, 1])   
        else:
            self._candidate_shortcut.set_data([], [])

        # Update iteration text
        if iteration is not None:
            self._iteration_text.set_text(f"Iteration: {iteration}")

        # Redraw the plot
        self._fig.canvas.draw()
        plt.pause(0.1)
