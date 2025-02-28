import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from rrt import rrt
from smoother import Smoother
from matplotlib.patches import Ellipse, Rectangle

class Scene:
    """
    A scene with both elliptical and rectangular obstacles. The scene is 10x10 by default.
    It generates collision-free start and goal points that are at least a specified minimum distance apart.
    """
    def __init__(self, bounds=(0, 10, 0, 10), 
                 num_ellipses=10, num_rectangles=5, 
                 min_axis=0.5, max_axis=1.5, 
                 min_side=0.5, max_side=1.5, 
                 min_distance=5.0):
        
        self.bounds = bounds
        self.obstacles = []
        # Generate elliptical obstacles; each obstacle is stored as ("ellipse", center, rx, ry)
        self.obstacles += self._generate_elliptical_obstacles(num_ellipses, min_axis, max_axis)
        # Generate rectangular obstacles; each obstacle is stored as ("rectangle", center, width, height)
        self.obstacles += self._generate_rectangular_obstacles(num_rectangles, min_side, max_side)
        
        self.start = self._generate_collision_free_point()
        self.goal = self._generate_collision_free_point()
        # Ensure that the start and goal are at least min_distance apart
        while np.linalg.norm(self.goal - self.start) < min_distance:
            self.goal = self._generate_collision_free_point()
    
    def _generate_elliptical_obstacles(self, num_obstacles, min_axis, max_axis):
        obstacles = []
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(num_obstacles):
            rx = random.uniform(min_axis/2, max_axis/2)
            ry = random.uniform(min_axis/2, max_axis/2)
            cx = random.uniform(x_min + rx, x_max - rx)
            cy = random.uniform(y_min + ry, y_max - ry)
            obstacles.append(("ellipse", np.array([cx, cy]), rx, ry))
        return obstacles
    
    def _generate_rectangular_obstacles(self, num_obstacles, min_side, max_side):
        """
        Generate rectangular obstacles:
          - Each obstacle is represented as ("rectangle", center, width, height)
          - To ensure that the rectangle remains within bounds even after rotation,
            we use half of its diagonal as a buffer from the scene boundaries.
        """
        obstacles = []
        x_min, x_max, y_min, y_max = self.bounds
        for _ in range(num_obstacles):
            width = random.uniform(min_side, max_side)
            height = random.uniform(min_side, max_side)
            half_diag = np.sqrt((width/2)**2 + (height/2)**2)
            cx = random.uniform(x_min + half_diag, x_max - half_diag)
            cy = random.uniform(y_min + half_diag, y_max - half_diag)
            obstacles.append(("rectangle", np.array([cx, cy]), width, height))
        return obstacles
    
    def _is_point_collision_free(self, pos):
        """
        Check if the given point pos collides with any obstacle.
        For ellipses: the point is rotated into the ellipse frame and checked against the ellipse equation.
        For rectangles: the point is rotated into the rectangle's coordinate frame and checked to see if it falls within.
        """
        for obs in self.obstacles:
            if obs[0] == "ellipse":
                _, center, rx, ry = obs
                dx = pos[0] - center[0]
                dy = pos[1] - center[1]
                if (dx**2 / rx**2 + dy**2 / ry**2) <= 1:
                    return False
            elif obs[0] == "rectangle":
                _, center, width, height = obs
                dx = pos[0] - center[0]
                dy = pos[1] - center[1]
                if abs(dx) <= width/2 and abs(dy) <= height/2:
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
    
    def plot_scene(self, tree=None, path=None, smoothed_path=None, ax=None, save_image=False, image_path="rrt.png"):
        """
        Plot the scene, including obstacles, start/goal points, the RRT tree, and the found path.
        Different obstacle types are shown with different colors.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        # Plot obstacles
        for obs in self.obstacles:
            if obs[0] == "ellipse":
                _, center, rx, ry = obs
                ellipse = Ellipse(xy=center, width=2*rx, height=2*ry,
                                edgecolor='r', facecolor='gray', alpha=0.5)
                ax.add_patch(ellipse)
            elif obs[0] == "rectangle":
                _, center, width, height = obs
                rect = Rectangle((center[0]-width/2, center[1]-height/2), width, height,
                                edgecolor='r', facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
        # Plot start and goal points
        ax.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'bo', markersize=10, label='Goal')
        # Plot RRT tree edges if provided
        if tree is not None:
            for node in tree:
                if node.parent is not None:
                    p1 = node.parent.position
                    p2 = node.position
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', alpha=0.8)
        # Plot path if provided
        if path is not None:
            path_positions = np.array([state[0] for state in path])
            ax.plot(path_positions[:, 0], path_positions[:, 1], color='red', linewidth=3, label='Path')
        if smoothed_path is not None:
            smoothed_path_positions = np.array([state[0] for state in smoothed_path])
            ax.plot(smoothed_path_positions[:, 0], smoothed_path_positions[:, 1], color='purple', linewidth=3, label='Smoothed Path')
        x_min, x_max, y_min, y_max = self.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title("Scene with Random Obstacles and RRT")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="lower left")
        
        # Save image if requested
        if save_image:
            fig.savefig(image_path)
            print(f"Scene image saved to {image_path}")
        
        return ax

# Example usage (assuming rrt and Smoother modules are available)
if __name__ == "__main__":
    seed = 42
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    scene = Scene(num_ellipses=10, num_rectangles=15, min_distance=5.0)
    
    print("Scene bounds:", scene.bounds)
    print("Start:", scene.start)
    print("Goal:", scene.goal)
    
    # Run RRT to search for a path (the rrt function must be implemented elsewhere)
    path, tree = rrt(scene.start, scene.goal, scene.bounds, collision_checker=scene.collision_checker, step_size=0.5, max_iterations=1000)
    
    if path is not None:
        print("RRT found a path.")
    else:
        print("No path was found.")
    
    ax = scene.plot_scene(tree=tree, path=path, save_image=False)
    plt.show()
    
    # Smooth the path
    vmax, amax = np.array([1.0, 1.0]), np.array([1.0, 1.0])
    smoother = Smoother(path=path, bounds=np.array(scene.bounds).reshape(-1, 2), vmax=vmax, amax=amax, collision_checker=scene.collision_checker, obstacles=scene.obstacles)
    smoother.smooth_path(plot_traj=True, save_gif=False)
    plt.close()

    smoothed_path = smoother.interpolate_control_trajectory()
    ax = scene.plot_scene(tree=tree, smoothed_path=smoothed_path, save_image=False, image_path='rrt_smoothed')
    plt.show()