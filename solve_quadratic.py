import numpy as np

def solve_quadratic(a, b, c):
    """
    Solve a*x^2 + b*x + c = 0 using a numerically stable method.
    Returns a sorted list of real solutions.
    
    This method minimizes cancellation errors by choosing the branch
    based on the sign of b and falls back to the double-root formula when needed.
    """
    if abs(a) < 1e-15:  # Treat as linear if a is nearly zero.
        return [-c/b] if abs(b) >= 1e-15 else []
    
    D = b**2 - 4*a*c
    if D < 0:
        return []
    sqrtD = np.sqrt(D)
    
    # Choose branch to minimize cancellation errors.
    x1 = -0.5*(b + sqrtD)/a if b >= 0 else -0.5*(b - sqrtD)/a

    # If x1 is extremely small, assume near double-root.
    if abs(x1) < 1e-15:
        x1 = x2 = -b/(2*a)
    else:
        x2 = c/(a*x1)
    
    x1 = 0 if np.isclose(x1, 0, atol=1e-12) else x1
    x2 = 0 if np.isclose(x2, 0, atol=1e-12) else x2

    return sorted([x1, x2])
