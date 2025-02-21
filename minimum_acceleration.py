import numpy as np

def minimum_acceleration_interpolants(self, start_pos, end_pos, start_vel, end_vel, vmax, T, dim, t_margin=1e-5, a_margin=1e-6):
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
            raise ValueError

        # Find the minimum acceleration and corresponding primitive
        a_min, selected_primitive = min(valid_results, key=lambda x: x[0])

        if a_min <= self.amax[dim] + a_margin:
            a_min = np.clip(a_min, 0, self.amax[dim])  
        else: 
            # Return None if the acceleration exceeds the limit
            raise ValueError

        return a_min, selected_primitive

if __name__ == '__main__':
    np.random.seed(42)

    start_pos = np.random.uniform(-10, 10)
    end_pos = np.random.uniform(-10, 10)
    start_vel = np.random.uniform(-2, 2)
    end_vel = np.random.uniform(-2, 2)
    vmax = np.random.uniform(1, 3)
    T = np.random.uniform(1, 10)