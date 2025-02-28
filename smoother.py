"""
This implementation is based on the method presented in the paper:
"Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from compute_traj_segment import compute_traj_segment
from get_motion_state import get_motion_states_at_global_t, get_motion_states_at_local_t

class Smoother:
    """
    Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts.
    This class implements a smoothing algorithm for manipulator trajectories with bounded velocity 
    and acceleration, using optimal shortcuts for improved performance and natural-looking motion.
    """
    
    def __init__(self, path, bounds, vmax, amax, collision_checker, max_iterations=100, obstacles=None):
        """
        Initialize the Smoother class.

        Parameters:
        - path: List of waypoints np.array([position, velocity])
        - bounds: Position boundary limits, a N*2 array, where N is the number of dimensions.
        - vmax: Maximum velocity for each dimension.
        - amax: Maximum acceleration for each dimension.
        - collision_checker: Function to check for collisions.
        - max_iterations: Maximum number of shortcut iterations.
        - obstacles: A list of obstacles to be included in the plot.
        """
        self.path = path
        self.bounds = bounds
        self.vmax = vmax
        self.amax = amax
        self.dimension = self.vmax.shape[0]
        self.collision_checker = collision_checker
        self.max_iterations = max_iterations
        self.traj_segment_times = np.array([])  # Array of time durations for each segment
        self.traj_segment_params = []  # List of trajectories parameters for each segment
        self.obstacles = obstacles
        self.total_time = []

    def smooth_path(self, plot_traj=False, save_frames=False, save_path="smoothing_frames"):
        """
        Smooth the traj using time-optimal segments and shortcuts.

        Returns:
        - Smoothed path as a numpy array of waypoints [(position, velocity)].
        - traj_segment_times: A numpy array of time durations for each segment. Each element represents the duration 
            of the corresponding trajectory segment.
        - traj_segment_params: A list of parameters for each trajectory segment, containing:
            - For each segment, the parameters like acceleration limits, switch times, and trajectory type.
        """
        # The algorithm fails if the initial step fails
        if not self.generate_initial_traj():
            return self.path, None, None

        for iteration in range(self.max_iterations):
            total_time = np.sum(self.traj_segment_times)
            self.total_time.append(total_time)
            t1, t2 = self.select_random_times(total_time)
            shortcut_start = get_motion_states_at_global_t(self.path, self.traj_segment_times, self.traj_segment_params, t1, n_dim=self.dimension)
            shortcut_end = get_motion_states_at_global_t(self.path, self.traj_segment_times, self.traj_segment_params, t2, n_dim=self.dimension)
            shortcut_traj_time, shortcut_traj_param, traj_feasibility = compute_traj_segment(shortcut_start, shortcut_end, self.vmax, self.amax, 
                                                                           collision_checker=self.collision_checker, bounds=self.bounds, n_dim=self.dimension)
            if plot_traj:
                self.plot_traj(iteration, shortcut_start=shortcut_start, shortcut_end=shortcut_end, 
                                   candidate_shortcut_time=shortcut_traj_time, candidate_shortcut_param=shortcut_traj_param,
                                   save_frames=save_frames, save_path=save_path)
            # Only update if the traj exists and feasible
            if traj_feasibility:
                # Plot again if update segment data successfully
                if self.update_traj(shortcut_start, shortcut_end, t1, t2, shortcut_traj_time, shortcut_traj_param):
                    if plot_traj:
                        self.plot_traj(iteration, shortcut_start=shortcut_start, shortcut_end=shortcut_end, save_frames=save_frames, save_path=save_path)
        
        # Plot the final trajectory
        if plot_traj:
            self.plot_traj(iteration, shortcut_start=shortcut_start, shortcut_end=shortcut_end, save_frames=save_frames, save_path=save_path) 

        return self.path, self.traj_segment_times, self.traj_segment_params
    
    def generate_initial_traj(self):
        """Generate the initial time-optimal traj for all segments."""
        self.traj_segment_times = []
        self.traj_segment_params = []
        
        for i in range(self.path.shape[0] - 1):
            start_state, end_state = self.path[i], self.path[i + 1]
            traj_segment_time, traj_segment_param, traj_feasibility = compute_traj_segment(start_state, end_state, self.vmax, self.amax, 
                                                                         collision_checker=self.collision_checker, bounds=self.bounds, n_dim=self.dimension)
            if traj_segment_param is None or not traj_feasibility:
                return False
            self.traj_segment_times.append(traj_segment_time)
            self.traj_segment_params.append(traj_segment_param)

        self.traj_segment_times = np.array(self.traj_segment_times)

        return True
    
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

    def update_traj(self, start_state, end_state, t1, t2, shortcut_traj_time, shortcut_traj_param):
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

        # Calculate connection before
        prev_state = self.path[start_index]
        connect_time_before, connect_param_before, feasibility_before = compute_traj_segment(
            prev_state, start_state, self.vmax, self.amax, 
            collision_checker=self.collision_checker, bounds=self.bounds, n_dim=self.dimension)
        if not feasibility_before:
            return False

        # Calculate connection after
        next_state = self.path[end_index + 1]
        connect_time_after, connect_param_after, feasibility_after = compute_traj_segment(
            end_state, next_state, self.vmax, self.amax, 
            collision_checker=self.collision_checker, bounds=self.bounds, n_dim=self.dimension)
        if not feasibility_after:
            return False

        # Cancel update if the connection traj does not reduce total time
        if total_middle_time < connect_time_before + shortcut_traj_time + connect_time_after:
            return False    # NOTE: The issue is unlikely to occur, but it's better to be safe than sorry
        
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
    
    def plot_traj(self, iteration: int, shortcut_start: tuple, shortcut_end: tuple, 
                  candidate_shortcut_time=None, candidate_shortcut_param=None, save_frames=False, save_path="smoothing_frames"):
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
            save_frames (bool): Whether to save frames for gif.
            save_path (str): The folder where the frames will be saved.
        """
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
                                                    edgecolor='r', facecolor='gray', alpha=0.5,
                                                    label="Obstacle" if not self._obstacle_label_added else ""))
                    self._obstacle_label_added = True

            # Plot initial and smoothed traj
            initial_positions = np.array([state[0] for state in self.path])
            self._ax.plot(initial_positions[:, 0], initial_positions[:, 1], 'y--', label='Initial traj')
            self._traj_line, = self._ax.plot([], [], '-o', markersize=2, label='Smoothed traj')

            # Plot milestones and shortcut points and optionally candidate shortcut
            self._milestones, = self._ax.plot([], [], 'ro', markersize=8, label='Milestones')
            self._shortcut_points, = self._ax.plot([], [], 'yo', markersize=4, label='Shortcut Points')
            self._candidate_shortcut, = self._ax.plot([], [], '-yo', markersize=1, label='Candidate Shortcut')    

            # Add legend and iteration text
            self._ax.legend(loc="lower left")
            self._iteration_text = self._ax.text(0.95, 0.95, "", transform=self._ax.transAxes, 
                                                fontsize=12, color="blue", ha="right", va="top")

        # Update plot
        times = np.linspace(0, np.sum(self.traj_segment_times), 500)
        positions = np.array([get_motion_states_at_global_t(self.path, self.traj_segment_times, self.traj_segment_params, t, n_dim=self.dimension)[0] 
                              for t in times if get_motion_states_at_global_t(self.path, self.traj_segment_times, self.traj_segment_params, t, n_dim=self.dimension) is not None])
        self._traj_line.set_data(positions[:, 0], positions[:, 1])

        # Update milestones and shortcut points and optionally candidate shortcut
        initial_positions = np.array([state[0] for state in self.path])
        self._milestones.set_data(initial_positions[:, 0], initial_positions[:, 1])
        self._shortcut_points.set_data([shortcut_start[0][0], shortcut_end[0][0]], [shortcut_start[0][1], shortcut_end[0][1]])
        
        if candidate_shortcut_time is not None and candidate_shortcut_param is not None:
            times = np.linspace(0, candidate_shortcut_time, 20)
            positions = np.array([get_motion_states_at_local_t(shortcut_start, candidate_shortcut_param, t, n_dim=self.dimension)[0] for t in times 
                                  if get_motion_states_at_local_t(shortcut_start, candidate_shortcut_param, t, n_dim=self.dimension)[0] is not None])
            self._candidate_shortcut.set_data(positions[:, 0], positions[:, 1])   
        else:
            self._candidate_shortcut.set_data([], [])

        # Update iteration text
        if iteration is not None:
            self._iteration_text.set_text(f"Iteration: {iteration}")

        # Save the frame if requested
        if save_frames:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not hasattr(self, 'frame_index'):
                self.frame_index = 0
            self._fig.savefig(f"{save_path}/frame_{self.frame_index:03d}.png")
            self.frame_index += 1

        # Redraw the plot
        self._fig.canvas.draw()
        plt.pause(0.1)

    def interpolate_control_trajectory(self, control_frequency=60):
        """
        Interpolates the trajectory at a given control frequency.

        This function samples the trajectory at uniform time intervals based on 
        the specified control frequency and returns the interpolated trajectory.

        Parameters:
        - control_frequency: The frequency (Hz) at which to sample the trajectory.

        Returns:
        - A numpy array of interpolated trajectory states [(position, velocity)].
        """
        total_time = np.sum(self.traj_segment_times)
        num_points = int(total_time * control_frequency) + 1
        times = np.linspace(0, total_time, num_points)
        interpolated_trajectory = []
        for t in times:
            state = get_motion_states_at_global_t(self.path, self.traj_segment_times, self.traj_segment_params, t, n_dim=self.dimension)
            if state is not None:
                interpolated_trajectory.append(state)
        interpolated_trajectory = np.array(interpolated_trajectory)

        return interpolated_trajectory