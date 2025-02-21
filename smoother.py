"""
This implementation is based on the method presented in the paper:
"Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts"
"""

import numpy as np
import matplotlib.pyplot as plt

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
        self.vmax = np.array(vmax)
        self.amax = np.array(amax)
        self.dimension = self.vmax.shape[0]
        self.collision_checker = collision_checker
        self.max_iterations = max_iterations
        self.segment_time = np.array([])  # Array of time durations for each segment
        self.segment_trajectory = []  # List of trajectories for each segment
        self.obstacles = obstacles
        self.total_time = []

    def smooth_path(self, plot_trajectory=False):
        """
        Smooth the trajectory using time-optimal segments and shortcuts.

        Returns:
        - Updated path as a numpy array of waypoints [(position, velocity)].
        """
        # The algorithm fails if the initial step fails
        if not self._generate_initial_trajectory():
            return None

        for iteration in range(self.max_iterations):
            total_time = np.sum(self.segment_time)
            self.total_time.append(total_time)
            if plot_trajectory:
                self.plot_trajectory(iteration, obstacles=self.obstacles)
            t1, t2 = self._select_random_times(total_time)
            start_state = self._get_state_at_time(t1)
            end_state = self._get_state_at_time(t2)

            shortcut_time, shortcut_trajectory = self._compute_optimal_segment(start_state, end_state)
            
            # Only update if the trajectory exists
            if shortcut_trajectory is not None:
                self._update_segment_data(start_state, end_state, t1, t2, shortcut_time, shortcut_trajectory)

        return self.path, self.segment_time, self.segment_trajectory
    
    def _generate_initial_trajectory(self):
        """Generate the initial time-optimal trajectory for all segments."""
        self.segment_trajectory = []
        segment_times = []

        for i in range(self.path.shape[0] - 1):
            start_state, end_state = self.path[i], self.path[i + 1]
            segment_time, segment_trajectory = self._compute_optimal_segment(start_state, end_state)
            if segment_trajectory is None:
                return False
            segment_times.append(segment_time)
            self.segment_trajectory.append(segment_trajectory)

        self.segment_time = np.array(segment_times)

        return True

    def _calculate_segment_time(self, start_state, end_state, safe_margin=1e-15):
        """
        Calculate the maximum time required to traverse a segment across all dimensions,
        considering vmax and amax constraints.
        """
        t_requireds = np.array([
            self._univariate_time_optimal_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                amax=self.amax[dim]
            )
            for dim in range(self.dimension)
        ])
        if None in t_requireds:
            return None

        return np.max(t_requireds) + safe_margin

    def _calculate_segment_trajectory(self, start_state, end_state, segment_time):
        """
        Calculate the trajectory for a single segment using minimum acceleration interpolants.
        """
        # Vectorized calculation for all dimensions
        trajectory_data = [
            self._minimum_acceleration_interpolants(
                start_pos=start_state[0][dim],
                end_pos=end_state[0][dim],
                start_vel=start_state[1][dim],
                end_vel=end_state[1][dim],
                vmax=self.vmax[dim],
                T=segment_time,
                dim=dim,
            )
            for dim in range(self.dimension)
        ]

        # Return None if the trajectory doesn't exist
        if None in trajectory_data:
            return None

        return np.array(trajectory_data, dtype=object)
    
    def _select_random_times(self, total_time, min_time_interval=0.01):
        """
        Select two random times within the total trajectory duration.

        Input:
        - total_time: The total duration of the trajectory
        - min_time_interval: Minimal time interval between t1 and t2

        Return:
        - t1, t2: Two random times within the total trajectory duration.
        """
        t1 = t2 = 0
        while t2 - t1 < min_time_interval:
            random_times = np.random.uniform(0, total_time, 2)
            t1, t2 = np.sort(random_times)
        return t1, t2

    def _get_state_at_time(self, t):
        """
        Find the interpolated state (position, velocity) at a specific time t within the trajectory.

        Input:
        - t: The target time

        Return:
        - position: Numpy array of positions at time t.
        - velocity: Numpy array of velocities at time t.
        """
        elapsed_time = 0

        for i in range(self.segment_time.shape[0]):  # Use shape[0] since segment_time is a NumPy array
            segment_time = self.segment_time[i]
            if elapsed_time + segment_time >= t:
                # Relative time within the current segment
                relative_t = t - elapsed_time
                start_state = self.path[i]
                end_state = self.path[i + 1]

                # Compute and return the interpolated state
                return self._get_state_in_segment(
                    start_state=start_state,
                    end_state=end_state,
                    segment_time=segment_time,
                    segment_trajectory=self.segment_trajectory[i],
                    t=relative_t
                )

            elapsed_time += segment_time

        # If t is beyond the total trajectory duration
        return None

    def _get_state_in_segment(self, start_state, end_state, segment_time, segment_trajectory, t):
        """
        Compute the state (position, velocity) within a segment at time t.

        Input:
        - start_state: The starting state of the segment.
        - end_state: The ending state of the segment.
        - segment_time: The duration of the segment.
        - segment_trajectory: Trajectory parameters for each dimension.
        - t: Relative time within the segment.

        Return:
        - position: Numpy array of positions at time t.
        - velocity: Numpy array of velocities at time t.
        """
        position = np.zeros(self.dimension)
        velocity = np.zeros(self.dimension)

        for dim in range(self.dimension):
            acc, trajectory_type = segment_trajectory[dim]
            x1, x2 = start_state[0][dim], end_state[0][dim]
            v1, v2 = start_state[1][dim], end_state[1][dim]

            pos, vel = self._compute_trajectory_state(
                x1=x1, x2=x2, v1=v1, v2=v2, a=acc,
                vmax=self.vmax[dim], T=segment_time,
                trajectory_type=trajectory_type, t=t
            )
            position[dim] = pos
            velocity[dim] = vel

        return position, velocity
    
    def _compute_optimal_segment(self, start_state, end_state):
        segment_time = self._calculate_segment_time(start_state, end_state)
        segment_trajectory = self._calculate_segment_trajectory(
                start_state, end_state, segment_time
            ) if segment_time is not None else None
        
        return segment_time, segment_trajectory

    def _compute_trajectory_state(self, x1, x2, v1, v2, a, vmax, T, trajectory_type, t):
        """
        Compute the position x(t) and velocity v(t) for a given trajectory type.

        Input:
        - x1, x2: Initial and final positions.
        - v1, v2: Initial and final velocities.
        - a: Acceleration used in the trajectory.
        - vmax: Maximum velocity.
        - T: Total trajectory time.
        - trajectory_type: One of 'P+P-', 'P-P+', 'P+L+P-', 'P-L-P+'.
        - t: The time at which to compute the state.

        Return:
        - x_t: Position at time t.
        - v_t: Velocity at time t.
        """
        if trajectory_type == 'P+P-':
            # Compute switch time
            t_s = 0.5 * (T + (v2 - v1) / a)
            if t <= t_s:  # Acceleration phase
                v_t = v1 + a * t
                x_t = x1 + v1 * t + 0.5 * a * t**2
            else:  # Deceleration phase
                delta_t = t - t_s
                v_peak = v1 + a * t_s
                v_t = v_peak - a * delta_t
                x_t = (x1 + v1 * t_s + 0.5 * a * t_s**2 +
                    v_peak * delta_t - 0.5 * a * delta_t**2)

        elif trajectory_type == 'P-P+':
            # Compute switch time
            t_s = 0.5 * (T + (v1 - v2) / a)
            if t <= t_s:  # Deceleration phase
                v_t = v1 - a * t
                x_t = x1 + v1 * t - 0.5 * a * t**2
            else:  # Acceleration phase
                delta_t = t - t_s
                v_valley = v1 - a * t_s
                v_t = v_valley + a * delta_t
                x_t = (x1 + v1 * t_s - 0.5 * a * t_s**2 +
                    v_valley * delta_t + 0.5 * a * delta_t**2)

        elif trajectory_type == 'P+L+P-':
            # Compute durations
            t_p1 = (vmax - v1) / a
            t_p2 = (vmax - v2) / a
            t_l = T - t_p1 - t_p2
            if t <= t_p1:  # Acceleration phase
                v_t = v1 + a * t
                x_t = x1 + v1 * t + 0.5 * a * t**2
            elif t <= t_p1 + t_l:  # Constant velocity phase
                delta_t = t - t_p1
                v_t = vmax
                x_t = (x1 + v1 * t_p1 + 0.5 * a * t_p1**2 +
                    vmax * delta_t)
            else:  # Deceleration phase
                delta_t = t - t_p1 - t_l
                v_t = vmax - a * delta_t
                x_t = (x1 + v1 * t_p1 + 0.5 * a * t_p1**2 +
                    vmax * t_l +
                    vmax * delta_t - 0.5 * a * delta_t**2)

        elif trajectory_type == 'P-L-P+':
            # Compute durations
            t_p1 = (vmax + v1) / a
            t_p2 = (vmax + v2) / a
            t_l = T - t_p1 - t_p2
            if t <= t_p1:  # Deceleration phase
                v_t = v1 - a * t
                x_t = x1 + v1 * t - 0.5 * a * t**2
            elif t <= t_p1 + t_l:  # Constant negative velocity phase
                delta_t = t - t_p1
                v_t = -vmax
                x_t = (x1 + v1 * t_p1 - 0.5 * a * t_p1**2 +
                    (-vmax) * delta_t)
            else:  # Acceleration phase
                delta_t = t - t_p1 - t_l
                v_t = -vmax + a * delta_t
                x_t = (x1 + v1 * t_p1 - 0.5 * a * t_p1**2 +
                    (-vmax) * t_l +
                    (-vmax) * delta_t + 0.5 * a * delta_t**2)

        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        return x_t, v_t

    def _update_segment_data(self, start_state, end_state, t1, t2, shortcut_time, shortcut_trajectory):
        """
        Update the trajectory and path by replacing the section between t1 and t2 with a shortcut.
        Also inserts connecting segments to ensure continuity.
        Collision checking and time reduction checking are performed at this step 
        to ensure the validity and efficiency of the connection trajectory.

        Input:
        - start_state, end_state: States at t1 and t2 respectively.
        - t1, t2: Start and end times of the shortcut in the original trajectory.
        - shortcut_time: Duration of the shortcut segment.
        - shortcut_trajectory: Trajectory for the shortcut segment.
        """
        # Find the indices of segments affected by t1 and t2
        elapsed_time = 0
        start_index, end_index = None, None
        for i, segment_time in enumerate(self.segment_time):
            if elapsed_time <= t1 < elapsed_time + segment_time:
                start_index = i
            if elapsed_time <= t2 <= elapsed_time + segment_time:
                end_index = i
            elapsed_time += segment_time

        if start_index is None or end_index is None:
            raise ValueError("Invalid times t1 or t2, cannot find affected segments.")

        # Get total time of affected segments between start_index and end_index
        middle_segment_times = self.segment_time[start_index:end_index+1]
        total_middle_time = np.sum(middle_segment_times)

        # Locate the start and end nodes for connection
        prev_state = self.path[start_index]  # Previous node before t1
        connect_time_before, connect_trajectory_before = self._compute_optimal_segment(prev_state, start_state)

        next_state = self.path[end_index + 1]  # Next node after t2
        connect_time_after, connect_trajectory_after = self._compute_optimal_segment(end_state, next_state)

        # Cancel update if the connection trajectory is invalid, does not reduce total time, or is not collision-free
        if connect_time_before is None or connect_time_after is None or connect_trajectory_before is None or connect_trajectory_after is None:
            return None
        if total_middle_time < connect_time_before + shortcut_time + connect_time_after:
            return None
        if not self._is_segment_collision_free(start_state, end_state, shortcut_time, shortcut_trajectory):
            return None
        if not self._is_segment_collision_free(prev_state, start_state, connect_time_before, connect_trajectory_before):
            return None
        if not self._is_segment_collision_free(end_state, next_state, connect_time_after, connect_trajectory_after):
            return None

        # Update path, segment_time and segment_trajectory using np.concatenate
        self.path = np.concatenate([
            self.path[:start_index + 1],
            [start_state, end_state],
            self.path[end_index + 1:]
        ])

        before_time = self.segment_time[:start_index]
        after_time = self.segment_time[end_index + 1:]
        self.segment_time = np.concatenate([
            before_time,
            [connect_time_before, shortcut_time, connect_time_after],
            after_time
        ])

        before_trajectory = self.segment_trajectory[:start_index]
        after_trajectory = self.segment_trajectory[end_index + 1:]
        self.segment_trajectory = before_trajectory + [
            connect_trajectory_before,
            shortcut_trajectory,
            connect_trajectory_after
        ] + after_trajectory
    
    def _univariate_time_optimal_interpolants(self, start_pos, end_pos, start_vel, end_vel, vmax, amax):
        """
        Compute the time-optimal trajectory execution time for univariate motion.

        Input:
        - start_pos, end_pos: Initial and final positions.
        - start_vel, end_vel: Initial and final velocities.
        - vmax: Maximum velocity.
        - amax: Maximum acceleration.

        Return:
        - T: Minimal execution time for valid motion primitive combinations, or None if no valid combination exists.
        """
        x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

        def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]
        
        # Class P+P-
        def compute_p_plus_p_minus():
            coefficients = [amax, 2 * v1, (v1**2 - v2**2) / (2 * amax) + x1 - x2]
            solutions = solve_quadratic(*coefficients)
            valid_t = [t for t in solutions if max((v2 - v1) / amax, 0) <= t <= (vmax - v1) / amax]
            if not valid_t:
                return None
            t_p = valid_t[0]
            
            return np.array(2 * t_p + (v1 - v2) / amax)

        # Class P-P+
        def compute_p_minus_p_plus():
            coefficients = [amax, -2 * v1, (v1**2 - v2**2) / (2 * amax) + x2 - x1]
            solutions = solve_quadratic(*coefficients)
            valid_t = [t for t in solutions if max((v1 - v2) / amax, 0) <= t <= (vmax + v1) / amax]
            if not valid_t:
                return None
            t_p = valid_t[0]
            
            return np.array(2 * t_p + (v2 - v1) / amax)

        # Class P+L+P-
        def compute_p_plus_l_plus_p_minus():
            t_p1 = (vmax - v1) / amax
            t_p2 = (vmax - v2) / amax
            t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) + (x2 - x1) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return np.array(t_p1 + t_l + t_p2)

        # Class P-L-P+
        def compute_p_minus_l_plus_p_plus():
            t_p1 = (vmax + v1) / amax
            t_p2 = (vmax + v2) / amax
            t_l = (v2**2 + v1**2 - 2 * vmax**2) / (2 * vmax * amax) - (x2 - x1) / vmax
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return np.array(t_p1 + t_l + t_p2)

        # Evaluate all four classes in the specified order
        t_p_plus_p_minus = compute_p_plus_p_minus()
        t_p_minus_p_plus = compute_p_minus_p_plus()
        t_p_plus_l_plus_p_minus = compute_p_plus_l_plus_p_minus()
        t_p_minus_l_plus_p_plus = compute_p_minus_l_plus_p_plus()

        # Collect valid times and return the minimal one
        times = [t for t in [t_p_plus_p_minus, t_p_minus_p_plus, t_p_plus_l_plus_p_minus, t_p_minus_l_plus_p_plus] if t is not None]
        return np.min(times) if times else None
    
    def _minimum_acceleration_interpolants(self, start_pos, end_pos, start_vel, end_vel, vmax, T, dim, t_margin=1e-5, a_margin=1e-6):
        """
        Compute the minimum-acceleration trajectory for fixed end time T.

        Input:
        - start_pos, end_pos: Initial and final positions.
        - start_vel, end_vel: Initial and final velocities.
        - vmax: Maximum velocity.
        - T: Fixed end time.
        - dim: Current dimension.
        - t_margin: A small time margin to compensate for numerical precision errors
        - a_margin: A small acceleration margin to compensate for numerical precision errors

        Return:
        - a_min: Minimal acceleration for valid motion primitive combinations, or None if no valid combination exists.
        - selected_primitive: Name of the selected motion primitive.
        """
        x1, x2, v1, v2 = start_pos, end_pos, start_vel, end_vel

        def solve_quadratic(a, b, c):
            """Solve quadratic equation ax^2 + bx + c = 0 and return real solutions."""
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return []
            sqrt_discriminant = np.sqrt(discriminant)
            return [(-b + sqrt_discriminant) / (2 * a), (-b - sqrt_discriminant) / (2 * a)]

        # Class P+P-
        def compute_p_plus_p_minus():
            coefficients = [T**2, 2 * T * (v1 + v2) + 4 * (x1 - x2), -(v2 - v1)**2]
            solutions = solve_quadratic(*coefficients)
            valid_a = []
            for a in solutions:
                if a <= 0:
                    continue
                t_s = 0.5 * (T + (v2 - v1) / a)
                if 0 < t_s < T + t_margin and abs(v1 + a * t_s) <= vmax:
                    valid_a.append(a)
            return (min(valid_a), 'P+P-') if valid_a else None

        # Class P-P+
        def compute_p_minus_p_plus():
            coefficients = [T**2, -2 * T * (v1 + v2) - 4 * (x1 - x2), -(v2 - v1)**2]
            solutions = solve_quadratic(*coefficients)
            valid_a = []
            for a in solutions:
                if a <= 0:
                    continue
                t_s = 0.5 * (T + (v1 - v2) / a)
                if 0 < t_s < T + t_margin and abs(v1 - a * t_s) <= vmax:
                    valid_a.append(a)
            return (min(valid_a), 'P-P+') if valid_a else None

        # Class P+L+P-
        def compute_p_plus_l_plus_p_minus():
            a = (vmax**2 - vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax - (x2 - x1))
            if a <= 0:
                return None
            t_p1 = (vmax - v1) / a
            t_p2 = (vmax - v2) / a
            t_l = T - t_p1 - t_p2
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return (a, 'P+L+P-')

        # Class P-L-P+
        def compute_p_minus_l_minus_p_plus():
            a = (vmax**2 + vmax * (v1 + v2) + 0.5 * (v1**2 + v2**2)) / (T * vmax + (x2 - x1))
            if a <= 0:
                return None
            t_p1 = (vmax + v1) / a
            t_p2 = (vmax + v2) / a
            t_l = T - t_p1 - t_p2
            if t_p1 < 0 or t_p2 < 0 or t_l < 0:
                return None
            return (a, 'P-L-P+')

        # Evaluate all four classes independently
        results = [
            compute_p_plus_p_minus(),  # P+P-
            compute_p_minus_p_plus(),  # P-P+
            compute_p_plus_l_plus_p_minus(),  # P+L+P-
            compute_p_minus_l_minus_p_plus()  # P-L-P+
        ]
        valid_results = [result for result in results if result is not None]

        if not valid_results:
            # Return None if no valid result
            return None

        # Find the minimum acceleration and corresponding primitive
        a_min, selected_primitive = min(valid_results, key=lambda x: x[0])

        if a_min <= self.amax[dim] + a_margin:
            a_min = np.clip(a_min, 0, self.amax[dim])  
        else: 
            # Return None if the acceleration exceeds the limit
            return None

        return a_min, selected_primitive

    def _is_segment_collision_free(self, start_state, end_state, segment_time, segment_trajectory, time_step=0.01):
        # Generate time points to sample along the trajectory
        num_samples = int(segment_time / time_step) + 1
        sampled_times = np.linspace(0, segment_time, num_samples)

        for time in sampled_times:
            state = self._get_state_in_segment(
                start_state=start_state, end_state=end_state, 
                segment_time=segment_time, segment_trajectory=segment_trajectory,
                t=time
            )
            if not self.collision_checker(state):
                return False
        return True
    
    def plot_trajectory(self, iteration=None, obstacles=None):
        """
        Plot the current trajectory of the smoother object in 2D with animation during smoothing.

        This method dynamically updates the trajectory plot for visualization without creating new windows.
        Optionally, displays the current iteration number.

        Args:
            iteration (int, optional): The current iteration number to display on the plot.
            obstacles (list of tuple, optional): List of obstacles, where each obstacle is defined as a
                                                tuple (x, y, width, height).
        """
        if self.dimension != 2:
            raise ValueError("This plotting function only supports 2D trajectories.")

        # Initialize plot only on the first call
        if not hasattr(self, "_fig"):
            self._fig, self._ax = plt.subplots(figsize=(8, 6))
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")
            self._ax.set_title("2D Trajectory Smoothing")
            self._ax.grid(True)
            self._ax.axis("equal")

            # Add obstacles if provided
            if obstacles is not None:
                # Check if the obstacle label has already been added
                if not hasattr(self, "_obstacle_label_added"):
                    self._obstacle_label_added = False

                for obs in obstacles:
                    x, y, width, height = obs
                    # Add the label "Obstacle" only once
                    if not self._obstacle_label_added:
                        self._ax.add_patch(plt.Rectangle((x, y), width, height, color="gray", alpha=0.5, label="Obstacle"))
                        self._obstacle_label_added = True
                    else:
                        self._ax.add_patch(plt.Rectangle((x, y), width, height, color="gray", alpha=0.5))

            # Plot the initial trajectory
            initial_positions = np.array([state[0] for state in self.path])
            self._ax.plot(initial_positions[:, 0], initial_positions[:, 1], 'y--', label='Initial Trajectory')

            # Line for the current smoothed trajectory
            self._trajectory_line, = self._ax.plot([], [], '-o', markersize=2, label='Smoothed Trajectory')

            # Plot the milestones
            self._milestones, = self._ax.plot([], [], 'ro', markersize=8, label='Milestones')

            # Place legend in the upper-left corner
            self._ax.legend(loc="upper left")

            # Text for iteration number in the upper-right corner
            self._iteration_text = self._ax.text(0.95, 0.95, "", 
                                                transform=self._ax.transAxes, 
                                                fontsize=12, color="blue",
                                                ha="right", va="top")

        # Update the trajectory
        positions = []
        times = np.linspace(0, np.sum(self.segment_time), 500)
        for t in times:
            state = self._get_state_at_time(t)
            if state is not None:
                positions.append(state[0])

        positions = np.array(positions)
        self._trajectory_line.set_data(positions[:, 0], positions[:, 1])

        # Update milestones
        milestone_positions = np.array([state[0] for state in self.path])
        self._milestones.set_data(milestone_positions[:, 0], milestone_positions[:, 1])

        # Update iteration text
        if iteration is not None:
            self._iteration_text.set_text(f"Iteration: {iteration}")

        self._fig.canvas.draw()
        plt.pause(0.1)
