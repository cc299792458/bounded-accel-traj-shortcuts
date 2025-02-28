# Bounded-Accel-Traj-Shortcuts

This repository contains a Python implementation of a real-time trajectory smoothing algorithm based on the paper:  
**["Fast Smoothing of Manipulator Trajectories using Optimal Bounded-Acceleration Shortcuts"](https://ieeexplore.ieee.org/document/5509683), ICRA 2010**.  

It efficiently generates shortcut trajectories that satisfy **velocity and acceleration constraints**.

📌 **Note**: This implementation is a **reproduction** of the original paper. You can find the original implementation **[here](https://github.com/krishauser/KrisLibrary/blob/master/planning/ParabolicRamp.cpp)**.  

---

## ✨ Example: RRT Path vs. Smoothed Path  

The following example illustrates how the trajectory smoothing algorithm refines a raw RRT-generated path.  

| **Original RRT Path** | **Smoothed Path** |
|----------------|----------------|
| ![RRT Path](rrt.png) | ![Smoothed Path](rrt_smoothed.png) |

To better visualize the effect of smoothing, see the GIF below:  

<div align=center>
  <img src="trajectory_smoothing.gif" width="75%"/>
</div>

---

## 🛠 Usage
**[Smoother](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/smoother.py)** integrates the trajectory smoothing algorithm by applying optimal bounded-acceleration shortcuts to smooth a given path. Its main idea is to **continuously select two random points along the trajectory**, using their positions and velocities as boundary conditions. Under the given velocity and acceleration constraints, it **computes the optimal trajectory for that segment as a shortcut to replace the original path**. The optimal solution for this problem is the **trapezoidal velocity profile**. The method computes the optimal trajectory by employing four motion modes: "P+P-", "P-P+", "P+L+P-", and "P-L-P+", as detailed in the original paper.

### Some Core Components

- **[Univariate Time-Optimal Interpolants](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/univariate_time_optimal.py)**  
  This corresponds to Section IV.C of the paper. It calculates the shortest time required for a segment given the boundary conditions and velocity/acceleration constraints. **This problem always has a solution**. The file also provides several examples.

- **[Minimum-Acceleration Interpolants](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/minimum_acceleration.py)**  
  This corresponds to Section IV.D of the paper. It computes the minimum acceleration needed for a segment given the boundary conditions along with velocity and time constraints. **Note that this problem may not always have a solution**. The file similarly includes some examples.

- **[compute_trajectory_segment](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/compute_traj_segment.py)**
  This utilizes **[Univariate Time-Optimal Interpolants](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/univariate_time_optimal.py)** and **[Minimum-Acceleration Interpolants](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/minimum_acceleration.py)** to compute a time-optimal trajectory under given boundary conditions with velocity and acceleration constraints in the multi-dimensional case. It first calculates the shortest time for each dimension, taking the longest of these times as the trajectory time. Then, based on this time, it calculates the minimum acceleration required for each dimension. Finally, the function checks whether the computed trajectory is collision-free and within bounds.

- **[get motion state](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/get_motion_state.py)**
  This returns the motion state given a time t, whether it is within a specific trajectory segment (local) or across the entire trajectory (global).

- **[consistency_validatation](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/consistency_validate.py)**  
  This validates the consistency between **[Univariate Time-Optimal Interpolants](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/univariate_time_optimal.py)** and **[Minimum-Acceleration Interpolants](https://github.com/cc299792458/bounded-accel-traj-shortcuts/blob/main/minimum_acceleration.py)**.

---

This implementation carefully handles numerical precision issues to ensure robust performance.

🌟If you find this useful, please star it—thank you! If you have any questions, feel free to open an issue.
