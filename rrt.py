import random
import numpy as np

# Define a node for RRT
class Node:
    def __init__(self, position, parent=None):
        self.position = position  # 2D numpy array for position
        self.parent = parent      # Parent node to trace back the path

# Default collision checker: returns True for all states
def default_collision_checker(state):
    # state is in the form (position, velocity)
    return True

def rrt(start, goal, bounds, collision_checker=default_collision_checker, step_size=0.5, max_iterations=1000):
    """
    A simple RRT planner that also saves the search tree.
    
    Parameters:
      - start: Start position as a numpy array, e.g. np.array([x, y])
      - goal: Goal position as a numpy array, e.g. np.array([x, y])
      - bounds: Tuple defining planning bounds (x_min, x_max, y_min, y_max)
      - collision_checker: Function to check if a state (position, velocity) is collision-free
      - step_size: The step size to extend towards the random sample
      - max_iterations: Maximum number of iterations to try
      
    Returns:
      - A tuple (path_states, tree):
          path_states: If a path is found, a numpy array of states, each state is a tuple (position, velocity).
                       Otherwise, None.
          tree: List of all nodes explored (the search tree).
    """
    nodes = []
    start_node = Node(start)
    nodes.append(start_node)
    
    for i in range(max_iterations):
        # Randomly sample a position within the bounds
        rand_position = np.array([
            random.uniform(bounds[0], bounds[1]),
            random.uniform(bounds[2], bounds[3])
        ])
        
        # Find the nearest node in the tree to the sampled position
        nearest_node = min(nodes, key=lambda node: np.linalg.norm(rand_position - node.position))
        
        # Extend from the nearest node towards the sampled position
        direction = rand_position - nearest_node.position
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance
            new_position = nearest_node.position + direction * step_size
        else:
            new_position = rand_position
        
        # Check for collision along the line from the nearest node to the new position
        collision_free = True
        num_checks = 10
        for j in range(num_checks + 1):
            intermediate = nearest_node.position + (j / num_checks) * (new_position - nearest_node.position)
            # Here, velocity is set to zero
            if not collision_checker((intermediate, np.zeros_like(intermediate))):
                collision_free = False
                break
        if not collision_free:
            continue
        
        # Add the new node to the tree if collision-free
        new_node = Node(new_position, parent=nearest_node)
        nodes.append(new_node)
        
        # Check if the new node is close enough to the goal
        if np.linalg.norm(new_position - goal) < step_size:
            goal_node = Node(goal, parent=new_node)
            nodes.append(goal_node)
            # Trace back the path
            path = []
            current = goal_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path.reverse()  # From start to goal
            
            # Convert the path to the state format required by Smoother (velocity set to zero)
            path_states = [(pos, np.zeros_like(pos)) for pos in path]
            return np.array(path_states, dtype=object), nodes
    
    return None, nodes  # No path found within max_iterations