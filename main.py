import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from rrt import rrt

# Scene class with obstacles and plot method
class Scene:
    """
    A class representing a 10x10 scene with random elliptical obstacles.
    It provides start, goal, bounds, obstacles, and a collision_checker function.
    The start and goal points are generated to be collision-free and at least a specified minimum distance apart.
    """
    def __init__(self, bounds=(0, 10, 0, 10), num_obstacles=10, min_axis=0.5, max_axis=1.5, min_distance=5.0, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.bounds = bounds
        self.obstacles = self._generate_obstacles(num_obstacles, min_axis, max_axis)
        self.start = self._generate_collision_free_point()
        self.goal = self._generate_collision_free_point()
        # Ensure start and goal are at least min_distance apart
        while np.linalg.norm(self.goal - self.start) < min_distance:
            self.goal = self._generate_collision_free_point()
    
    def _generate_obstacles(self, num_obstacles, min_axis, max_axis):
        obstacles = []
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(num_obstacles):
            rx = random.uniform(min_axis, max_axis)
            ry = random.uniform(min_axis, max_axis)
            # Ensure the ellipse is fully within the bounds
            cx = random.uniform(x_min + rx, x_max - rx)
            cy = random.uniform(y_min + ry, y_max - ry)
            theta = random.uniform(0, np.pi)
            obstacles.append((np.array([cx, cy]), rx, ry, theta))
        return obstacles

    def _is_point_collision_free(self, pos):
        for (center, rx, ry, theta) in self.obstacles:
            dx = pos[0] - center[0]
            dy = pos[1] - center[1]
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            # Rotate the point into the ellipse coordinate frame
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t
            if (x_rot**2 / rx**2 + y_rot**2 / ry**2) <= 1:
                return False
        return True

    def _generate_collision_free_point(self, max_attempts=1000):
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(max_attempts):
            candidate = np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])
            if self._is_point_collision_free(candidate):
                return candidate
        raise RuntimeError("Failed to generate a collision-free point after many attempts.")

    def collision_checker(self, state):
        pos = state[0]
        return self._is_point_collision_free(pos)
    
    def plot_scene(self, tree=None, path=None, ax=None):
        """
        Plot the scene, including obstacles, start/goal points, the RRT search tree, and the found path.
        
        Parameters:
          - tree: (Optional) List of RRT nodes. If provided, plot tree edges with yellow lines (alpha=0.8).
          - path: (Optional) The found path as a sequence of states. If provided, plot the path with a bold red line.
          - ax: (Optional) Matplotlib axis to plot on.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        # Plot obstacles
        for (center, rx, ry, theta) in self.obstacles:
            ellipse = Ellipse(xy=center, width=2*rx, height=2*ry, angle=np.degrees(theta),
                              edgecolor='r', facecolor='gray', alpha=0.5)
            ax.add_patch(ellipse)
        # Plot start and goal points
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'bo', markersize=10, label='Goal')
        # Plot RRT tree edges if tree is provided
        if tree is not None:
            for node in tree:
                if node.parent is not None:
                    p1 = node.parent.position
                    p2 = node.position
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', alpha=0.8)
        # Plot found path if provided
        if path is not None:
            path_positions = np.array([state[0] for state in path])
            ax.plot(path_positions[:, 0], path_positions[:, 1], color='red', linewidth=3, label='Path')
        # Set plot limits and labels
        x_min, x_max, y_min, y_max = self.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title("Scene with Random Elliptical Obstacles and RRT")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        return ax

# Example usage
if __name__ == "__main__":
    # Create a scene with 10 random elliptical obstacles in a 10x10 area,
    # ensuring that the start and goal points are collision-free and at least 5 units apart.
    scene = Scene(num_obstacles=10, min_axis=0.5, max_axis=1.5, min_distance=5.0, seed=0)
    
    print("Scene bounds:", scene.bounds)
    print("Start point:", scene.start)
    print("Goal point:", scene.goal)
    
    # Run RRT on the scene using its collision_checker and bounds
    path, tree = rrt(scene.start, scene.goal, scene.bounds, collision_checker=scene.collision_checker, step_size=0.5, max_iterations=1000)
    
    if path is not None:
        print("RRT found a path.")
    else:
        print("No path was found.")
    
    # Plot the scene, the RRT tree, and the found path
    ax = scene.plot_scene(tree=tree, path=path)
    plt.show()