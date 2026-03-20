import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxpy as cp


def optimal_staircase_cvxpy(x_data, y_data, n_steps, step_penalty=1.0):
    """
    Find optimal staircase approximation using convex optimization (CVXPY).
    
    This minimizes: ||y - staircase(x)||^2 + step_penalty * (total variation)
    
    The total variation term penalizes changes between consecutive steps,
    encouraging the staircase to stay flat where possible.
    
    Parameters:
    -----------
    x_data : ndarray
        Input x values (should be sorted)
    y_data : ndarray
        Target y values to approximate
    n_steps : int
        Number of steps in the staircase
    step_penalty : float
        Weight for the step change penalty (higher = fewer/smaller steps)
        
    Returns:
    --------
    step_edges : ndarray
        Optimized positions of step edges
    step_values : ndarray
        Optimized heights of each step
    """
    n_points = len(x_data)
    
    # Decision variables
    step_values = cp.Variable(n_steps)
    
    # We'll use a fixed equal spacing for step edges initially
    # (optimizing step locations is non-convex, so we do that separately)
    step_edges = np.linspace(x_data[0], x_data[-1], n_steps + 1)
    
    # Assign each data point to a step
    assignment = np.zeros(n_points, dtype=int)
    for i, x in enumerate(x_data):
        for j in range(n_steps):
            if step_edges[j] <= x < step_edges[j + 1]:
                assignment[i] = j
                break
        if x >= step_edges[-1]:
            assignment[i] = n_steps - 1
    
    # Build the design matrix (which step each point belongs to)
    A = np.zeros((n_points, n_steps))
    for i in range(n_points):
        A[i, assignment[i]] = 1
    
    # Objective: fit error + total variation penalty
    fit_error = cp.sum_squares(A @ step_values - y_data)
    
    # Total variation: sum of absolute differences between consecutive steps
    total_variation = cp.norm1(cp.diff(step_values))
    
    objective = cp.Minimize(fit_error + step_penalty * total_variation)
    
    # Solve
    problem = cp.Problem(objective)
    problem.solve()
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Warning: Optimization status: {problem.status}")
    
    return step_edges, step_values.value


def optimal_staircase_with_adaptive_edges(x_data, y_data, n_steps, step_penalty=1.0, 
                                          edge_optimization_iters=5):
    """
    Find optimal staircase with adaptive step edges.
    
    This uses an iterative approach:
    1. Fix edge locations, optimize step heights (convex)
    2. Fix step heights, optimize edge locations (using gradient descent)
    3. Repeat
    
    Parameters:
    -----------
    x_data : ndarray
        Input x values (should be sorted)
    y_data : ndarray
        Target y values to approximate
    n_steps : int
        Number of steps in the staircase
    step_penalty : float
        Weight for the step change penalty
    edge_optimization_iters : int
        Number of iterations for edge optimization
        
    Returns:
    --------
    step_edges : ndarray
        Optimized positions of step edges
    step_values : ndarray
        Optimized heights of each step
    """
    # Initialize with equal spacing
    step_edges = np.linspace(x_data[0], x_data[-1], n_steps + 1)
    
    for iteration in range(edge_optimization_iters):
        # Step 1: Optimize heights given fixed edges
        step_values = cp.Variable(n_steps)
        
        # Assign points to steps
        assignment = np.zeros(len(x_data), dtype=int)
        for i, x in enumerate(x_data):
            for j in range(n_steps):
                if step_edges[j] <= x < step_edges[j + 1]:
                    assignment[i] = j
                    break
            if x >= step_edges[-1]:
                assignment[i] = n_steps - 1
        
        # Build design matrix
        A = np.zeros((len(x_data), n_steps))
        for i in range(len(x_data)):
            A[i, assignment[i]] = 1
        
        # Optimize heights
        fit_error = cp.sum_squares(A @ step_values - y_data)
        total_variation = cp.norm1(cp.diff(step_values))
        objective = cp.Minimize(fit_error + step_penalty * total_variation)
        problem = cp.Problem(objective)
        problem.solve()
        
        heights = step_values.value
        
        # Step 2: Optimize edge locations given fixed heights
        # We'll use a simple heuristic: place edges where residuals change sign
        # or use k-means style updates
        for j in range(1, n_steps):
            # Find points near this edge
            nearby_left = (x_data >= step_edges[j-1]) & (x_data < step_edges[j])
            nearby_right = (x_data >= step_edges[j]) & (x_data < step_edges[j+1])
            
            if np.any(nearby_left) and np.any(nearby_right):
                # Move edge to minimize within-step variance
                x_left = x_data[nearby_left]
                x_right = x_data[nearby_right]
                
                # Use midpoint of the boundary region
                step_edges[j] = (x_left.max() + x_right.min()) / 2
    
    return step_edges, heights


def evaluate_staircase(x, step_edges, step_values):
    """Evaluate staircase function at given x values."""
    x = np.atleast_1d(x)
    result = np.zeros_like(x, dtype=float)
    n_steps = len(step_values)
    
    for i in range(n_steps):
        mask = (x >= step_edges[i]) & (x < step_edges[i + 1])
        result[mask] = step_values[i]
    
    result[x >= step_edges[-1]] = step_values[-1]
    
    return result


# Example usage
if __name__ == "__main__":
    # Generate sample data from a non-linear function
    np.random.seed(42)
    x_data = np.linspace(0, 10, 200)
    
    # Non-linear function (e.g., quadratic with some noise)
    y_true = 0.3 * x_data**2 - 2 * x_data + 5 + 0.5 * np.sin(2 * x_data)
    y_data = y_true + np.random.normal(0, 0.5, len(x_data))
    
    # Different step penalties to compare
    penalties = [0.1, 1.0, 5.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, penalty in enumerate(penalties):
        ax = axes[idx]
        
        # Optimize staircase
        n_steps = 15
        step_edges, step_values = optimal_staircase_cvxpy(
            x_data, y_data, n_steps, step_penalty=penalty
        )
        
        # Evaluate staircase
        y_staircase = evaluate_staircase(x_data, step_edges, step_values)
        
        # Plot
        ax.scatter(x_data, y_data, alpha=0.3, s=10, label='Data')
        ax.plot(x_data, y_true, 'g--', linewidth=2, label='True function', alpha=0.7)
        ax.plot(x_data, y_staircase, 'r-', linewidth=2, label='Staircase')
        
        # Mark step edges
        for edge in step_edges[1:-1]:
            ax.axvline(edge, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_title(f'Step penalty = {penalty}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate metrics
        mse = np.mean((y_data - y_staircase)**2)
        n_changes = np.sum(np.abs(np.diff(step_values)) > 0.1)
        ax.text(0.02, 0.98, f'MSE: {mse:.2f}\nActive steps: {n_changes}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/optimal_staircase.png', dpi=150)
    plt.show()
    
    print("\nOptimization complete!")
    print(f"As step_penalty increases, the staircase becomes 'flatter' with fewer changes.")