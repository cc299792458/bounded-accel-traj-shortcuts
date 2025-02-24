import numpy as np

from tqdm import tqdm
from minimum_acceleration import minimum_acceleration_interpolants
from univariate_time_optimal import univariate_time_optimal_interpolants

# ------------------ Testing Code ------------------
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    # for i in tqdm(range(1_000_000)):
    #     # Sample random boundary conditions
    #     start_pos = np.random.uniform(-100, 100)
    #     end_pos = np.random.uniform(-100, 100)
    #     start_vel = np.random.uniform(-10, 10)
    #     end_vel = np.random.uniform(-10, 10)
    #     vmax = np.random.uniform(10, 20)
    #     amax = np.random.uniform(2, 10)

    #     trajectories, optimal_label = univariate_time_optimal_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, amax)
    #     T = trajectories[optimal_label][0]

    #     trajectories, optimal_label = minimum_acceleration_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, T, amax)
    #     amin = trajectories[optimal_label][0]

    #     assert np.isclose(amin, amax)
    #     assert amin <= amax

    for i in tqdm(range(1_000_000)):
        # Sample random boundary conditions
        start_pos = np.random.uniform(-100, 100)
        end_pos = np.random.uniform(-100, 100)
        start_vel = np.random.uniform(-10, 10)
        end_vel = np.random.uniform(-10, 10)
        vmax = np.random.uniform(10, 20)
        amax = np.random.uniform(2, 10)

        trajectories, optimal_label = univariate_time_optimal_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, amax)
        T = trajectories[optimal_label][0] * np.random.uniform(1.0, 10.0) 

        if i == 281:
            print("")

        trajectories, optimal_label = minimum_acceleration_interpolants(start_pos, end_pos, start_vel, end_vel, vmax, T, amax)
        amin = trajectories[optimal_label][0]

        assert amin <= amax