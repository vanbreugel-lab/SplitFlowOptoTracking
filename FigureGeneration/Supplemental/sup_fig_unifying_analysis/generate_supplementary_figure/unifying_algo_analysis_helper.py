import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import math
import scipy.stats

import pandas as pd

import copy

import cvxpy
cp = cvxpy

import figurefirst as fifi

from braid_analysis import braid_analysis_plots

def angle_distance(angle1, angle2):
    """
    Calculate the minimum distance between two angles.

    Parameters:
    -----------
    angle1, angle2 : float or array-like
        Angles in radians

    Returns:
    --------
    float or array
        Minimum distance between angles in radians (always positive)
        Range: [0, π]
    """
    diff = angle1 - angle2
    # Wrap to [-π, π]
    distance = np.arctan2(np.sin(diff), np.cos(diff))
    return distance

def mean_angle(angle):
    
    mean = np.arctan2( np.mean(np.sin(angle)), np.mean(np.cos(angle)) )
    return mean

def wrap_angle(angle):
    """Wrap angle to range [-π, π]"""
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def random_integers(min_val, max_val, count):
    """Generate a list of unique random integers between min_val and max_val."""
    if count > (max_val - min_val + 1):
        raise ValueError("Count exceeds available unique values")

    return random.sample(range(min_val, max_val + 1), count)

def miop_optimization(course, ix, plot=False, verbose=False, pis_gamma=0):

    slope = cvxpy.Variable(1)
    intercept = cvxpy.Variable(1)
    pis = cvxpy.Variable(len(course[ix]), integer=True)
    
    eq = (course[ix]) - (slope*ix + intercept + pis*2*np.pi)
    
    
    #objective = cvxpy.Minimize(cp.sum(cp.sqrt(cp.square(eq) + 1) - 1) + 0.01*cvxpy.sum_squares(pis))
    objective = cvxpy.Minimize(cvxpy.sum(cvxpy.huber(eq, M=1.0)) + pis_gamma*cvxpy.sum_squares(pis))
    #objective = cvxpy.Minimize(cvxpy.sum_squares(eq) + 1*cvxpy.norm1(eq) + 0.01*cvxpy.sum_squares(pis))
    
    constraints = [
      pis >= -10, pis <= 10,  # Adjust based on your data
      intercept >= course.min() - 10, intercept <= course.max() + 10
    ]
    
    problem = cvxpy.Problem(objective, constraints)
    
    problem.solve(solver=cvxpy.GUROBI,
                verbose=verbose,
                Threads=8,              # Use multiple cores
                Method=2,               # Try different algorithms: -1=auto, 0=primal, 1=dual, 2=barrier
                Presolve=2,             # Aggressive presolve
                FeasibilityTol=1e-5,    # Relax from 1e-6
                OptimalityTol=1e-5,     # Relax from 1e-6
                TimeLimit=20,          # Stop after 5 minutes
                MIPGap=0.01,            # For MIP: accept 1% gap (if applicable)
                )
    
    if plot:
      pred = (slope*ix + intercept + pis*2*np.pi).value
    
      pred_wrap = []
      for p in pred:
        if p < -np.pi:
          p += 2*np.pi
        if p > np.pi:
          p -= 2*np.pi
        pred_wrap.append(p)
    
      plt.plot(ix, pred_wrap, '.')
      plt.plot(course, '.')
    
    return slope.value, intercept.value

def find_affine_transform(course, slope, intercept, min_ix, max_ix, abs_min_ix, plot=False, use_ix_limits=True, include_translation=False):
    ix_all = np.arange(0,len(course))
    
    pred = (slope*ix_all + intercept)
    
    pred_wrap = wrap_angle(pred)
    
    abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
    abs_course_fit_x = np.cos(abs_course_fit)
    abs_course_fit_y = np.sin(abs_course_fit)
    abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
    abs_course_data = course[ix_all]
    abs_course_data_x = np.cos(abs_course_data)
    abs_course_data_y = np.sin(abs_course_data)
    abs_course_data_xy = np.vstack((abs_course_data_x, abs_course_data_y))

    if use_ix_limits:
        AM = abs_course_data_xy[:,min_ix:max_ix]@np.linalg.pinv(abs_course_fit_xy[:,min_ix:max_ix])
    else:
        AM = abs_course_data_xy[:,abs_min_ix:]@np.linalg.pinv(abs_course_fit_xy[:,abs_min_ix:])
    
    if plot:
      warped_fit = AM@abs_course_fit_xy
      course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])
    
      plt.plot(ix_all, pred_wrap, '.')
      plt.plot(ix_all, course, '.')
    
      plt.plot(ix_all,  course_warped_fit, '.')

    if include_translation:
        AMt = np.eye(3)
        AMt[0:2,0:2] = AM 
        AMt[0,2] = None
        AMt[1,2] = None
        return AMt
    else:
        return AM

def find_affine_transform_cvx(course, slope, intercept, min_ix, max_ix, abs_min_ix, 
                              plot=False, use_ix_limits=True,
                              include_translation=True,
                              ix_all=None):

    if 1:
        if ix_all is None:
            ix_all = np.arange(0,len(course))
            abs_course_data = course[ix_all]
        else:
            abs_course_data = course


        pred = (slope*ix_all + intercept)
        
        pred_wrap = wrap_angle(pred)
        
        abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
        abs_course_fit_x = np.cos(abs_course_fit)
        abs_course_fit_y = np.sin(abs_course_fit)
        abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
        
        
        abs_course_data_x = np.cos(abs_course_data)
        abs_course_data_y = np.sin(abs_course_data)
        abs_course_data_xy = np.vstack((abs_course_data_x, abs_course_data_y))
        
        if use_ix_limits:
            abs_course_data_xy = abs_course_data_xy[:,min_ix:max_ix]
            abs_course_fit_xy = abs_course_fit_xy[:,min_ix:max_ix]

        


        X = abs_course_data_xy
        Y = abs_course_fit_xy

        if include_translation:
            X = np.vstack((X, np.ones_like(X[0,:])))
            Y = np.vstack((Y, np.ones_like(Y[0,:])))

        #A = cp.Variable((2, 2))
        a = cp.Variable()
        b = cp.Variable()
        c = cp.Variable()
        A = cp.bmat([[a, b], [b, c]])

        if include_translation:
            tx = cp.Variable()
            ty = cp.Variable()
            A = cp.bmat([[a, b, tx], [b, c, ty], [0, 0, 1]])
        residual = X - A @ Y

        # Huber loss
        huber_loss = cp.sum(cp.huber(residual, 0.5))

        # A is positive semidefinite
        #constraints = [A >> 0, (A[0,1]-A[1,0])**2<=0.001]  


        problem = cp.Problem(
            cp.Minimize(huber_loss), # + 0.1*cp.norm(A, 'nuc')),
            constraints=None,
        )

        problem.solve()

        # Get the optimal A
        #A_optimal = A.value
        if not include_translation:
            A_optimal = np.array([[a.value, b.value], [b.value, c.value]])

        if include_translation:
            A_optimal = np.array([[a.value, b.value, tx.value], [b.value, c.value, ty.value], [0, 0, 1]])

    else:
        print('cvx failed, doing regular affine as back up...')
        A_optimal = find_affine_transform(course, slope, intercept, min_ix, max_ix, abs_min_ix, plot=plot, use_ix_limits=use_ix_limits)

    return A_optimal

def miop_and_affine_fit(course, ix, min_ix, max_ix, abs_min_ix, use_ix_limits=True, use_cvx_affine=False, include_translation=False):
    slope, intercept = miop_optimization(course, ix, plot=False)
    if use_cvx_affine:
        AM = find_affine_transform_cvx(course, slope, intercept, min_ix, max_ix, abs_min_ix, 
                              plot=False, use_ix_limits=use_ix_limits, include_translation=include_translation)
    else:
        AM = find_affine_transform(course, slope, intercept, min_ix, max_ix, abs_min_ix, plot=False, use_ix_limits=use_ix_limits, include_translation=include_translation)
    #rotation, eccentricity = extract_rotation_and_scales_svd(AM)

    return slope[0], intercept[0], AM #rotation, eccentricity

def extract_shear_from_vt(Vt):
    """
    Extract shear parameters from the Vt matrix.
    
    Parameters:
    -----------
    Vt : ndarray, shape (2, 2)
        The Vt matrix from SVD
    
    Returns:
    --------
    shear_angle : float
        The angle (in radians) that the Vt matrix rotates the principal axes
    shear_factor : float
        The shear magnitude (0 means no shear, represents how "tilted" the 
        coordinate system is)
    
    Notes:
    ------
    Vt can be decomposed as a rotation that aligns the principal axes of the
    transformation with the coordinate axes. The shear information comes from
    how much Vt deviates from being orthogonal.
    """
    # Method 1: Angle of rotation that Vt applies
    shear_angle = np.arctan2(Vt[1, 0], Vt[0, 0])
    
    # Method 2: Shear factor - measure of non-orthogonality
    # This is the tangent of the angle between the two column vectors of Vt
    # For an orthogonal matrix, this would be 0
    col1 = Vt[:, 0]
    col2 = Vt[:, 1]
    dot_product = np.dot(col1, col2)
    shear_factor = dot_product  # This will be 0 for orthogonal matrices
    
    return shear_angle, shear_factor

def get_total_rotation_from_model(unifying_algo_fit, bootstrap_n, save_total_rotation_to_rotation=False):

    AM_0_0 = unifying_algo_fit.iloc[bootstrap_n].AM_0_0
    AM_0_1 = unifying_algo_fit.iloc[bootstrap_n].AM_0_1
    AM_1_0 = unifying_algo_fit.iloc[bootstrap_n].AM_1_0
    AM_1_1 = unifying_algo_fit.iloc[bootstrap_n].AM_1_1
    AM = np.array([[AM_0_0, AM_0_1], [AM_1_0, AM_1_1]])

    total_rotation, major_axis, minor_axis, Vt, translation = decompose_affine(AM)

    if save_total_rotation_to_rotation:
        unifying_algo_fit.at[bootstrap_n, 'rotation'] = total_rotation
        return unifying_algo_fit


def decompose_affine(matrix):
    """
    Decompose a 2D affine transformation matrix into rotation, axes, and V matrix.
    
    Parameters:
    -----------
    matrix : array-like, shape (2, 2), (2, 3), or (3, 3)
        The affine transformation matrix
    
    Returns:
    --------
    rotation : float
        Rotation angle in radians (from U @ Vt)
    major_axis : float
        First singular value  
    minor_axis : float
        Second singular value
    v_matrix : ndarray, shape (2, 2)
        The Vt matrix from SVD (needed for exact reconstruction)
    translation : tuple or None
        (tx, ty) translation components, or None if input was 2x2
    
    Notes:
    ------
    Full SVD decomposition: M = U @ diag(S) @ Vt
    Where:
    - rotation comes from U @ Vt
    - major_axis, minor_axis are the singular values S
    - v_matrix is Vt (needed for reconstruction)
    """
    matrix = np.array(matrix)
    
    # Extract the 2x2 linear transformation part
    if matrix.shape == (2, 2):
        linear = matrix
        translation = None
    elif matrix.shape == (2, 3):
        linear = matrix[:2, :2]
        translation = tuple(matrix[:2, 2])
    elif matrix.shape == (3, 3):
        linear = matrix[:2, :2]
        translation = tuple(matrix[:2, 2])
    else:
        raise ValueError("Matrix must be shape (2, 2), (2, 3), or (3, 3)")
    
    # Perform SVD: M = U @ S @ Vt
    U, S, Vt = np.linalg.svd(linear)
    
    # Handle reflection to ensure proper rotation
    if np.linalg.det(U @ Vt) < 0:
        Vt[-1, :] *= -1
        S[-1] *= -1
    
    # The rotation matrix is U @ Vt
    rotation_matrix = U @ Vt
    rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Decompose shear
    shear_angle, shear_factor = extract_shear_from_vt(Vt)

    # Total rotation
    total_rotation = rotation + shear_angle
    
    # Singular values
    major_axis = S[0]
    minor_axis = S[1]
    
    return total_rotation, major_axis, minor_axis, Vt, translation


def reconstruct_affine(rotation, major_axis, minor_axis, v_matrix, translation=None):
    """
    Reconstruct a 2D affine transformation matrix from SVD components.
    
    Parameters:
    -----------
    rotation : float
        Rotation angle in radians
    major_axis : float
        First singular value
    minor_axis : float
        Second singular value
    v_matrix : ndarray, shape (2, 2)
        The Vt matrix from SVD decomposition
    translation : tuple or None, optional
        (tx, ty) translation components. If None, returns 2x2 matrix.
    
    Returns:
    --------
    matrix : ndarray, shape (2, 2) or (2, 3)
        The reconstructed affine transformation matrix
    
    Notes:
    ------
    Reconstructs as: M = (U @ Vt) @ diag(S) @ Vt^{-1} = U @ diag(S) @ Vt
    """
    # Create U @ Vt (rotation matrix)
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    U_Vt = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # We need to recover U from (U @ Vt) and Vt
    # U = (U @ Vt) @ Vt^T  (since Vt is orthogonal, Vt^T = inv(Vt))
    U = U_Vt @ v_matrix.T
    
    # Create scaling matrix
    S = np.diag([major_axis, minor_axis])
    
    # Reconstruct: M = U @ S @ Vt
    linear = U @ S @ v_matrix
    
    # Return 2x2 or 2x3
    if translation is None:
        return linear
    else:
        tx, ty = translation
        affine_matrix = np.zeros((2, 3))
        affine_matrix[:2, :2] = linear
        affine_matrix[:2, 2] = [tx, ty]
        return affine_matrix

def bootstrap_miop_affine_fit(course, abs_min_ix, abs_max_ix, min_ix_range = 150, max_ix_range = 200, npoints = 50, use_ix_limits=True, n_bootstraps=10, use_cvx_affine=False, include_translation=False):
    #abs_min_ix = 0
    #abs_max_ix = len(course)-1
    #min_ix_range = 150 # 1.5 seconds at 0.01 fps
    
    unifying_algo_fit = []
    for i in range(n_bootstraps):
        adj_abs_max_ix = np.min([abs_max_ix, len(course)])

        
        min_ix = np.random.randint(abs_min_ix, adj_abs_max_ix-min_ix_range-1)
        adj_abs_max_ix = np.min([adj_abs_max_ix, min_ix+max_ix_range])

        #print(min_ix, min_ix+min_ix_range, adj_abs_max_ix)
        
        max_ix = np.random.randint(min_ix+min_ix_range, adj_abs_max_ix )
        ix = np.sort(random_integers(min_ix, max_ix, np.min([npoints, len(course)])))
    
        slope, intercept, AM = miop_and_affine_fit(course, ix, min_ix, max_ix, abs_min_ix, use_ix_limits=use_ix_limits, use_cvx_affine=use_cvx_affine, include_translation=include_translation)
        rotation, major_axis, minor_axis, Vt, translation = decompose_affine(AM)

        eccentricity = np.sqrt( 1 - minor_axis**2 / major_axis**2 )
        axis_ratio = np.abs(minor_axis) / np.abs(major_axis)

        rmse_line, true_values, errors, predicted_values = evaluate_miop_affine_fit(course, slope, intercept, np.eye(2), min_ix, max_ix)
        
        rmse_affine, true_values, errors, predicted_values = evaluate_miop_affine_fit(course, slope, intercept, AM, min_ix, max_ix)
        #rmse, huber, r_value_pearson, p_value_pearson, p_value_spearman, r_value_linregress, p_value_linregress = result_affine

        mean_residuals = mean_angle(errors)


        rsq = circular_r_squared(np.ravel(true_values), np.ravel(predicted_values))

        if AM.shape[0] == 3:
            tx = AM[0,2]
            ty = AM[1,2]
        else:
            tx = None
            ty = None
        
        data = {'min_ix': min_ix,
                'max_ix': max_ix,
                'slope': slope,
                'intercept': intercept,
                'rotation': rotation,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'AM_0_0': AM[0,0],
                'AM_0_1': AM[0,1],
                'AM_1_0': AM[1,0],
                'AM_1_1': AM[1,1],
                'tx': tx,
                'ty': ty,
                'rsq': rsq, 
                'mean_residuals': mean_residuals,
                #'eccentricity': eccentricity,
                'axis_ratio': axis_ratio,
                'rmse_affine': rmse_affine,
                #'r_value_pearson_affine': result_affine[2],
                #'p_value_pearson_affine': result_affine[3],
                'rmse_linear': rmse_line,
                #'r_value_pearson_linear': result_linear[2],
                #'p_value_pearson_linear': result_linear[3],
               }
    
        unifying_algo_fit.append(data)

    return pd.DataFrame(unifying_algo_fit)

def get_true_vs_predicted(trajec, unifying_algo_fit, bootstrap_n, plot=True, ax=None, use_ix_limits=True):
    min_ix = int(unifying_algo_fit.iloc[bootstrap_n].min_ix)
    max_ix = int(unifying_algo_fit.iloc[bootstrap_n].max_ix)
    slope = unifying_algo_fit.iloc[bootstrap_n].slope
    intercept = unifying_algo_fit.iloc[bootstrap_n].intercept
    AM_0_0 = unifying_algo_fit.iloc[bootstrap_n].AM_0_0
    AM_0_1 = unifying_algo_fit.iloc[bootstrap_n].AM_0_1
    AM_1_0 = unifying_algo_fit.iloc[bootstrap_n].AM_1_0
    AM_1_1 = unifying_algo_fit.iloc[bootstrap_n].AM_1_1
    AM = np.array([[AM_0_0, AM_0_1], [AM_1_0, AM_1_1]])
    
    course = trajec.course_smoothish.values #[min_ix:]

    ix_all = np.arange(0,len(course))
    if max_ix is None:
        max_ix = ix_all[-1]

    pred = (slope*(ix_all) + intercept)
    pred_wrap = wrap_angle(pred)
    
    abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
    abs_course_fit_x = np.cos(abs_course_fit)
    abs_course_fit_y = np.sin(abs_course_fit)
    abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
    warped_fit = AM@abs_course_fit_xy
    course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])

    if use_ix_limits:
        rmse = np.sqrt(np.mean( angle_distance(course[min_ix:max_ix], course_warped_fit[min_ix:max_ix])**2 ) )
        huber = huber_loss(angle_distance(course[min_ix:max_ix], course_warped_fit[min_ix:max_ix]), np.zeros_like(course[min_ix:max_ix]), 0.5)
    else:
        rmse = np.sqrt(np.mean( angle_distance(course, course_warped_fit)**2 ) )
        huber = huber_loss(angle_distance(course, course_warped_fit), np.zeros_like(course), 0.5)
        
    ## do some stats
    if use_ix_limits:
        true_values = course[min_ix:max_ix]
        errors = angle_distance(course[min_ix:max_ix], course_warped_fit[min_ix:max_ix])
    else:
        true_values = course
        errors = angle_distance(course, course_warped_fit)
    predicted_values = true_values - errors

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(true_values, predicted_values)

    if plot:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')

        # Create fit line
        fit_line = slope * true_values + intercept
        
        # Plot
        ax.scatter(true_values, predicted_values, color='black', alpha=0.6)
        ax.plot(true_values, fit_line, 'r--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}\nR²={r_value**2:.3f}, p={p_value:.3e}')
        
    return true_values, predicted_values

def weighted_median(values, weights):
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cumsum = np.cumsum(sorted_weights)
    cutoff = 0.5 * cumsum[-1]
    
    return sorted_values[np.searchsorted(cumsum, cutoff)]

def get_weighted_value(unifying_algo_fit, col='axis_ratio', return_value='mean', use='rmse_affine'):
    one_minus_rvalues = np.max(unifying_algo_fit[use].values) - unifying_algo_fit[use].values
    #one_minus_rvalues[one_minus_rvalues<1e-3] = 1e-3
    #one_minus_rvalues[one_minus_rvalues>1] = 1
    values = unifying_algo_fit[col].abs().values
    
    weights = 1/ unifying_algo_fit[use].values**2 # 1 / one_minus_rvalues

    # mean
    weights = weights / weights.sum()  # Normalize to sum to 1
    weighted_mean = np.sum(weights * values)

    # median
    weighted_med = weighted_median(values, weights)

    if return_value == 'mean':
        return weighted_mean
    elif return_value == 'median':
        return weighted_med

def plot_fit_for_bootstrap_n(trajec, unifying_algo_fit, bootstrap_n):

    min_ix = int(unifying_algo_fit.iloc[bootstrap_n].min_ix)
    max_ix = int(unifying_algo_fit.iloc[bootstrap_n].max_ix)
    slope = unifying_algo_fit.iloc[bootstrap_n].slope
    intercept = unifying_algo_fit.iloc[bootstrap_n].intercept
    AM_0_0 = unifying_algo_fit.iloc[bootstrap_n].AM_0_0
    AM_0_1 = unifying_algo_fit.iloc[bootstrap_n].AM_0_1
    AM_1_0 = unifying_algo_fit.iloc[bootstrap_n].AM_1_0
    AM_1_1 = unifying_algo_fit.iloc[bootstrap_n].AM_1_1
    AM = np.array([[AM_0_0, AM_0_1], [AM_1_0, AM_1_1]])
    
    course = trajec.course_smoothish.values #[min_ix:]
    
    plot_miop_affine_fit(course, slope, intercept, AM, min_ix, max_ix)

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculates the Huber loss.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        delta (float): The threshold parameter that determines the
                       switch between L2 (squared) and L1 (absolute) loss regimes.

    Returns:
        float: The mean Huber loss over all samples.
    """
    error = y_true - y_pred
    abs_error = np.abs(error)
    # Quadratic part for small errors (|error| <= delta)
    quadratic_loss = 0.5 * error**2
    # Linear part for large errors (|error| > delta)
    linear_loss = delta * (abs_error - 0.5 * delta)
    
    # Combine the two parts based on the condition |error| <= delta
    # np.where selects elements from quadratic_loss if the condition is true,
    # otherwise selects from linear_loss
    loss = np.where(abs_error <= delta, quadratic_loss, linear_loss)
    
    return np.mean(loss)

def evaluate_miop_affine_fit(course, slope, intercept, AM, min_ix=0, max_ix=None):
    ix_all = np.arange(0,len(course))
    if max_ix is None:
        max_ix = ix_all[-1]

    pred = (slope*(ix_all) + intercept)
    pred_wrap = wrap_angle(pred)
    
    abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
    abs_course_fit_x = np.cos(abs_course_fit)
    abs_course_fit_y = np.sin(abs_course_fit)
    abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
    if AM.shape[0] == 2:
        warped_fit = AM@abs_course_fit_xy
    elif AM.shape[0] == 3:
        abs_course_fit_xy_aug = np.vstack((abs_course_fit_xy, np.ones_like(abs_course_fit_xy[0,:])))
        warped_fit = AM@abs_course_fit_xy_aug

    course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])

    rmse = np.sqrt(np.mean( angle_distance(course[min_ix:max_ix], course_warped_fit[min_ix:max_ix])**2 ) )
    #huber = huber_loss(angle_distance(course[min_ix:max_ix], course_warped_fit[min_ix:max_ix]), np.zeros_like(course[min_ix:max_ix]), 0.5)

    ## do some stats
    true_values = course[min_ix:max_ix]
    errors = angle_distance(course[min_ix:max_ix], course_warped_fit[min_ix:max_ix])
    predicted_values = true_values - errors
    
    # Calculate Pearson correlation
    #r_value_pearson, p_value_pearson = scipy.stats.pearsonr(true_values, predicted_values)
    
    # Spearman correlation (rank-based, more robust to outliers)
    #rho, p_value_spearman = scipy.stats.spearmanr(true_values, predicted_values)
    
    # Linear regression (gives slope, intercept, r-value, p-value, std_err)
    #slope, intercept, r_value_linregress, p_value_linregress, std_err = scipy.stats.linregress(true_values, predicted_values)

    
    #return rmse, huber, r_value_pearson, p_value_pearson, p_value_spearman, r_value_linregress, p_value_linregress
    return rmse, true_values, errors, predicted_values
    
# def plot_miop_affine_fit(course, slope, intercept, AM, min_ix=0, max_ix=None, abs_min_ix = 120, markersize=1):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
    
#     ## Plot from min_ix onwards
#     ix_all = np.arange(0,len(course))
#     if max_ix is None:
#         max_ix = ix_all[-1]

#     pred = (slope*(ix_all) + intercept)
    
#     pred_wrap = wrap_angle(pred)
    
#     abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
#     abs_course_fit_x = np.cos(abs_course_fit)
#     abs_course_fit_y = np.sin(abs_course_fit)
#     abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
#     abs_course_data = course #[ix_all]
#     abs_course_data_x = np.cos(abs_course_data)
#     abs_course_data_y = np.sin(abs_course_data)
#     abs_course_data_xy = np.vstack((abs_course_data_x, abs_course_data_y))
    
#     warped_fit = AM@abs_course_fit_xy
#     course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])

#     ax.plot(ix_all, course, '.', label='raw data', color='black', markersize=markersize)
#     ax.plot(ix_all[abs_min_ix:], pred_wrap[abs_min_ix:], '.', label='line fit', color='blue')
#     ax.plot(ix_all[abs_min_ix:],  course_warped_fit[abs_min_ix:], '.', label='after affine warp', color='magenta')
#     ax.legend()
 
#     ## provide some reference
#     ax.fill_betweenx([np.pi, -np.pi], 20, 20+68, color='red', alpha=0.3, edgecolor='none')
#     ax.fill_betweenx([np.pi, -np.pi], 20+68, abs_min_ix, color='gray', alpha=0.3, edgecolor='none')
#     ax.fill_betweenx([np.pi, -np.pi], min_ix, max_ix, color='yellow', alpha=0.3, edgecolor='none')

#     xticks = [0, 20, 20+68, 120, 220, 320, 420, 520]
#     xticklabels = np.array(xticks)-20
    
    
#     ax.set_xlim(0, len(course))
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels)

#     ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
#     ax.set_ylim(-np.pi, np.pi)
#     ax.set_yticklabels(['$-\pi$', '$-\pi/2$','$0$','$\pi/2$','$\pi$',])

#     fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'])

#     ax.set_xlabel('Time relative to flash, ms')
#     ax.set_ylabel('Course')

def run_miop_affine_fit_on_trajectories(braid_df, n_trajecs, obj_id_key='obj_id_unique_event',
                                        abs_min_ix=120, abs_max_ix=500, min_ix_range = 190, max_ix_range=210):
    obj_ids = braid_df[obj_id_key].unique()
    if n_trajecs > len(obj_ids):
        n_trajecs = len(obj_ids)

    data = []
    i = 0
    while len(data) < n_trajecs:
        # get a single trajectory
        obj_id = obj_ids[i]
        trajec = braid_df[braid_df[obj_id_key]==obj_id]
        trajec = trajec.dropna()

        unifying_algo_fit = bootstrap_miop_affine_fit(trajec.course_smoothish.values, 
                                                      abs_min_ix, abs_max_ix, min_ix_range, max_ix_range,
                                                      use_ix_limits=True,
                                                      use_cvx_affine=False,
                                                      include_translation=False)

        unifying_algo_fit['iteration'] = unifying_algo_fit.index
        unifying_algo_fit['obj_id_unique_event'] = trajec.obj_id_unique_event.unique()[0]

        data.append(unifying_algo_fit)
        i += 1

    all_unifying_algo_data = pd.concat(data)
    return all_unifying_algo_data


def run_model_alignment_analysis(modelfit_df_fname, braid_df_fname, ax_course=None, ax_rsq=None,
                                 ix_model_start=-100, ix_model_stop=100, show_red=False,
                                 unifying_slope=None, unifying_axis_ratio=None, show_linear_fit=True, show_affine_fit=True):
    #modelfit_df_fname = 'laminar_data_flash.parquet'
    #braid_df_fname =  '/media/caveman/Vorchard1/david_wind_gates/Orco_CsChrimson_Laminar/stuff/merged/opto_laminar_merged_preprocessed_optotrigger_trimmed.hdf'
    modelfit_df = pd.read_parquet(modelfit_df_fname)

    try:
        braid_df = pd.read_hdf(braid_df_fname)
    except:
        braid_df = pd.read_parquet(braid_df_fname)
        
    if ax_course is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = ax_course
    
    aligned_courses = []
    aligned_ixs = []
    slopes = []
    AMs = []
    indiv_rho_from_model = []
    for i in range(len(modelfit_df.obj_id_unique_event.unique())):
    
        obj_id = modelfit_df.obj_id_unique_event.unique()[i]
        modelfit_trajec = modelfit_df[modelfit_df.obj_id_unique_event==obj_id]
        ix = modelfit_trajec.rmse_affine.argmin()
        modelfit_trajec_best = modelfit_trajec.iloc[ix]
        trajec = braid_df[braid_df.obj_id_unique_event==obj_id]
        aligned_course, aligned_ix, rho = plot_miop_affine_fit_from_model(modelfit_trajec_best, 
                                                         trajec, ax=ax, N=len(modelfit_df.obj_id_unique_event.unique()),
                                                         ix_model_start=ix_model_start, 
                                                         ix_model_stop=ix_model_stop,
                                                         show_red=show_red,
                                                         markersize=0.2)
        
        if len(aligned_course) == 200:
            aligned_course = interpolate_nans(aligned_course)
            aligned_courses.append(aligned_course)
            slopes.append(np.abs(modelfit_trajec_best.slope))

            #AM_0_0 = modelfit_trajec_best.AM_0_0
            #AM_0_1 = modelfit_trajec_best.AM_0_1
            #AM_1_0 = modelfit_trajec_best.AM_1_0
            #AM_1_1 = modelfit_trajec_best.AM_1_1
            #AM = np.array([[AM_0_0, AM_0_1], [AM_1_0, AM_1_1]])

            ix_all = np.arange(ix_model_start, ix_model_stop)
            AM = find_affine_transform(aligned_course, ix_all, np.abs(modelfit_trajec_best.slope), 0)
            #ix_all = np.arange(modelfit_trajec_best.min_ix, modelfit_trajec_best.max_ix)
            #slope = np.abs(modelfit_trajec_best.slope)

            indiv_rho_from_model.append(rho)

            AMs.append(AM)

            aligned_ixs.append(aligned_ix)

    print('slopes: ', np.median(slopes))

    if unifying_slope is None:
        group_slope = np.median(slopes)
    else:
        group_slope = unifying_slope*0.01 # multiply by dt to put back in the right units
        
    ix_all = np.arange(ix_model_start, ix_model_stop)
    pred = (group_slope*ix_all + 0)
    pred_wrap = wrap_angle(pred)

    if show_linear_fit:
        ax.plot(ix_all, pred_wrap, '.', markerfacecolor='blue', markeredgecolor='none')

    ax.set_rasterization_zorder(1000)
    
    ## This part is only done on the block from -100 to +100, the best 2 seconds
    aligned_courses_arr = np.vstack(aligned_courses)
    ix_arr = np.vstack([np.arange(ix_model_start, ix_model_stop) for i in range(len(aligned_courses))])
    intercept = 0
    #slope = np.median(slopes)

    if unifying_axis_ratio is None:
        group_AM = find_affine_transform(np.ravel(aligned_courses_arr), np.ravel(ix_arr), 
                                   group_slope, intercept)
    else:
        group_AM = find_affine_transform(np.ravel(aligned_courses_arr), np.ravel(ix_arr), 
                                   group_slope, intercept)
        U, S, Vt = np.linalg.svd(group_AM)
        S[1] = S[0]*unifying_axis_ratio
        group_AM = U@np.diag(S)@Vt
    
    # Group prediction
    # uses a single slope and AM for ALL trajectories
    group_rho, group_ix_arr, group_true_values, group_predicted_values, group_rmse = get_stats(aligned_courses_arr, ix_arr, group_slope, group_AM, intercept=0, aligned_ixs=aligned_ixs)

    # Individual prediction
    # uses a single slope and AM for ALL trajectories
    indiv_rho, indiv_ix_arr, indiv_true_values, indiv_predicted_values, indiv_rmses = get_stats(aligned_courses_arr, ix_arr, [group_slope]*len(slopes), [group_AM]*len(AMs), intercept=0, aligned_ixs=aligned_ixs)

    # group_pred = (group_slope*np.ravel(ix_arr) + intercept)
    # group_pred_wrap = wrap_angle(group_pred)
    
    # abs_course_fit = group_pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
    # abs_course_fit_x = np.cos(abs_course_fit)
    # abs_course_fit_y = np.sin(abs_course_fit)
    # abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
    # warped_fit = group_AM@abs_course_fit_xy
    # course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])

    # if show_affine_fit:
    #     ax.plot(np.ravel(ix_arr), course_warped_fit, '.', markerfacecolor='magenta', markeredgecolor='none')
    #     ax.set_title(modelfit_df_fname.split('.')[0])

    # #########
    
    # rval, pval, true_values, predicted_values = get_stats(np.ravel(aligned_courses_arr), np.ravel(ix_arr), slope, intercept, AM)
    # print('For all data together: ')
    # print('rval: ', rval)
    # print('rsq: ', rval**2)
    # print(pval)
    
    # rvals = []
    # rsq_vals = []
    # pvals = []

    # true_values_by_trajec = []
    # predicted_values_by_trajec = []
    # for i in range(len(aligned_courses)):
    #     _, pval, true_values, predicted_values = get_stats(aligned_courses[i], ix_arr[i], slope, intercept, AM)
    #     rvals.append(_)
    #     pvals.append(pval)
    #     rsq_vals.append(rval**2)
    #     true_values_by_trajec.append(true_values)
    #     predicted_values_by_trajec.append(predicted_values)

    # # decompose the affine transform
    # rotation, major_axis, minor_axis, Vt, translation =  decompose_affine(AM)
    # axis_ratio = np.abs(minor_axis / major_axis)

    
    
    return group_slope, group_AM, np.abs(group_rho), group_ix_arr, group_true_values, group_predicted_values, slopes, AMs, indiv_rho, indiv_ix_arr, indiv_true_values, indiv_predicted_values, group_rmse, indiv_rmses

def find_first_crossing_after_min_ix(pred_wrap, min_ix, max_ix, threshold=np.pi):
    ix_crossing = None
    for ix in np.arange(min_ix+1, max_ix):
        diff = pred_wrap[ix] - pred_wrap[ix-1]
        diff_sign = np.sign(pred_wrap[ix]) - np.sign(pred_wrap[ix-1])
        if np.abs(diff) < threshold:
            if np.abs(diff_sign) > 1e-3:
                ix_crossing = ix
                return ix_crossing

    if ix_crossing is None:
        return int(np.mean([min_ix, max_ix]))

def plot_miop_affine_fit(course, slope, intercept, min_ix=0, max_ix=None, abs_min_ix = 120, 
                         shift_ix=0, ax=None, N=200, ix_model_start=-100, ix_model_stop=100, show_red=False, markersize=0.5):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    ## Plot from min_ix onwards
    ix_all = np.arange(0,len(course))
    if max_ix is None:
        max_ix = ix_all[-1]

    pred = (slope*(ix_all) + intercept)
    
    pred_wrap = wrap_angle(pred)
    
    abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
    abs_course_fit_x = np.cos(abs_course_fit)
    abs_course_fit_y = np.sin(abs_course_fit)
    abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
    abs_course_data = course #[ix_all]
    abs_course_data_x = np.cos(abs_course_data)
    abs_course_data_y = np.sin(abs_course_data)
    abs_course_data_xy = np.vstack((abs_course_data_x, abs_course_data_y))
    
    ax.scatter(ix_all - shift_ix, course,
               c='black',
               s = markersize,
               linewidths=0,
               edgecolors='none',
                label='raw data', alpha=20/N)
    #ax.plot(ix_all[abs_min_ix:] - shift_ix, pred_wrap[abs_min_ix:], '.', label='line fit', color='blue')
    #ax.legend()
 
    ## provide some reference
    if show_red:
        ax.fill_betweenx([np.pi, -np.pi], 20 - shift_ix, 20+68 - shift_ix, color='red', alpha=2/N, edgecolor='none')
    #ax.fill_betweenx([np.pi, -np.pi], 20+68 - shift_ix, abs_min_ix - shift_ix, color='gray', alpha=0.3, edgecolor='none')
    
    ax.fill_betweenx([np.pi, -np.pi], min_ix - shift_ix, max_ix - shift_ix, color='yellow', alpha=2/N, edgecolor='none', zorder=-1000)

    #xticks = [0, 20, 20+68, 120, 220, 320, 420, 520]
    #xticklabels = np.array(xticks)-20
    
    ax.set_xlim(0 - shift_ix, len(course) - shift_ix)
    #ax.set_xticks(xticks)
    #ax.set_xticklabels(xticklabels)

    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticklabels(['$-\pi$', '$-\pi/2$','$0$','$\pi/2$','$\pi$',])

    fifi.mpl_functions.adjust_spines(ax, ['left', 'bottom'])

    ax.set_xlabel('Time relative to flash, ms')
    ax.set_ylabel('Course')

    ix_middle = np.argmin( np.abs(ix_all - shift_ix) )
    return course[ix_middle+ix_model_start:ix_middle+ix_model_stop], ix_all[ix_middle+ix_model_start:ix_middle+ix_model_stop]

def plot_miop_affine_fit_from_model(modelfit_trajec_best, trajec, ax=None, N=200, ix_model_start=-100, ix_model_stop=100, show_red=False,
                                   markersize=0.5):
    dt = 0.01
    min_ix = int(modelfit_trajec_best.min_ix)
    max_ix = int(modelfit_trajec_best.max_ix)
    slope = modelfit_trajec_best.slope
    intercept = modelfit_trajec_best.intercept
    
    course = trajec.course_smoothish.values #[min_ix:]


    ix_all = np.arange(0,len(course))
    if max_ix is None:
        max_ix = ix_all[-1]
    pred = (slope*(ix_all) + intercept)
    pred_wrap = wrap_angle(pred)
    
    shift_ix = find_first_crossing_after_min_ix(pred_wrap, min_ix, max_ix)

    # if shift ix is in the second half of min_ix, max_ix, then shift in phase by 2pi
    if 0: # doesn't really help much
        if shift_ix > min_ix + 2/3*(max_ix-min_ix):
            phase_ix = (2*np.pi/slope)
            shift_ix -= int(phase_ix)
    
    if np.sign(slope) < 0:
        slope *= -1
        course *= -1
        intercept *= -1

    aligned_course, aligned_ix = plot_miop_affine_fit(course, slope, intercept, min_ix=min_ix, max_ix=max_ix, shift_ix=shift_ix, ax=ax, N=N,
                                          ix_model_start=ix_model_start, ix_model_stop=ix_model_stop, show_red=show_red, markersize=markersize)

    course_best_rmse = course[min_ix:max_ix]
    pred_wrap_best_rmse = pred_wrap[min_ix:max_ix]
    pred_wrap_best_rmse_xy = np.vstack((np.cos(pred_wrap_best_rmse), np.sin(pred_wrap_best_rmse)))
    AM_0_0 = modelfit_trajec_best.AM_0_0
    AM_0_1 = modelfit_trajec_best.AM_0_1
    AM_1_0 = modelfit_trajec_best.AM_1_0
    AM_1_1 = modelfit_trajec_best.AM_1_1
    AM = np.array([[AM_0_0, AM_0_1], [AM_1_0, AM_1_1]])
    warped_pred_wrap_best_rmse_xy = AM@pred_wrap_best_rmse_xy
    warped_pred_course = np.arctan2(warped_pred_wrap_best_rmse_xy[1,:], warped_pred_wrap_best_rmse_xy[0,:])
    rho = get_circular_rho(course_best_rmse, warped_pred_course)

    return aligned_course, aligned_ix, rho

def interpolate_nans(arr):
    """
    Interpolate NaN values in a 1D numpy array using linear interpolation.
    
    Parameters:
    arr : numpy array
        1D array with NaN values to interpolate
        
    Returns:
    numpy array with NaN values interpolated
    """
    arr = arr.copy()  # Don't modify the original array
    nans = np.isnan(arr)
    
    # Get indices of non-NaN and NaN values
    valid_indices = np.where(~nans)[0]
    nan_indices = np.where(nans)[0]
    
    # Interpolate NaN values
    if len(valid_indices) > 0 and len(nan_indices) > 0:
        arr[nan_indices] = np.interp(nan_indices, valid_indices, arr[valid_indices])
    
    return arr

def find_affine_transform(course, ix_all, slope, intercept):
    
    pred = (slope*ix_all + intercept)
    
    pred_wrap = wrap_angle(pred)
    
    abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
    abs_course_fit_x = np.cos(abs_course_fit)
    abs_course_fit_y = np.sin(abs_course_fit)
    abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
    
    abs_course_data = course
    abs_course_data_x = np.cos(abs_course_data)
    abs_course_data_y = np.sin(abs_course_data)
    abs_course_data_xy = np.vstack((abs_course_data_x, abs_course_data_y))

    AM = abs_course_data_xy@np.linalg.pinv(abs_course_fit_xy)

    
    return AM

def get_circular_rho(true, pred):
    mean_true = mean_angle(true)
    mean_pred = mean_angle(pred)

    num = np.sum(np.sin(true - mean_true)*np.sin(pred - mean_pred))
    den = np.sqrt(np.sum(np.sin(true - mean_true)**2))*np.sqrt(np.sum(np.sin(pred - mean_pred)**2))

    rho = num/den
    
    return rho

def get_stats(course, ix_arr, slope, AM, intercept=0, aligned_ixs=None):

    try:
        _ = slope[0]
        combined = False
    except:
        combined = True


    if combined:
        course = np.ravel(course)
        ix_arr = np.ravel(ix_arr)

        if aligned_ixs is not None:
            good_ix = np.where(np.ravel(aligned_ixs)>120)[0]
            course = course[good_ix]
            ix_arr = ix_arr[good_ix]

    
        aligned_courses_arr = course
         
        pred = (slope*np.ravel(ix_arr) + intercept)
        pred_wrap = wrap_angle(pred)
        
        abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
        abs_course_fit_x = np.cos(abs_course_fit)
        abs_course_fit_y = np.sin(abs_course_fit)
        abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
        
        warped_fit = AM@abs_course_fit_xy
        course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])

        # find rho in coordinates of the data
        true_values = aligned_courses_arr
        errors = angle_distance(np.ravel(true_values), np.ravel(course_warped_fit))
        predicted_values = true_values - errors
        rho = get_circular_rho(true_values, predicted_values)

        rmse = np.sqrt(np.mean(errors**2))

        # find rho in the coordiantes of the fit -- more uniform distribution
        # true_xy = np.vstack((np.cos(aligned_courses_arr), np.sin(aligned_courses_arr)))
        # true_xy_unwarp = np.linalg.inv(AM)@true_xy
        # true_unwarped = np.arctan2(true_xy_unwarp[1,:], true_xy_unwarp[0,:])
        # pred_values = slope*ix_arr
        # rho = get_circular_rho(true_unwarped, pred_values)

        return rho, ix_arr, true_values, predicted_values, rmse

    else:
        rhos = []
        true_values_by_trajec = []
        predicted_values_by_trajec = []
        rmses = []
        ix_arr_is = []

        for i in range(len(course)):    
            aligned_courses_arr_i = course[i]
            ix_arr_i = ix_arr[i]

            if aligned_ixs is not None:
                good_ix = np.where(aligned_ixs[i]>120)[0]
                aligned_courses_arr_i = aligned_courses_arr_i[good_ix]
                ix_arr_i = ix_arr_i[good_ix]
             
            pred = (slope[i]*ix_arr_i + intercept)
            pred_wrap = wrap_angle(pred)



            line = slope[i]*np.hstack(ix_arr_i)
            line_xy = np.vstack((np.cos(line), np.sin(line)))
            warped_line = AM[i]@line_xy
            predicted_values = np.arctan2(warped_line[1,:], warped_line[0,:])


            true_values = aligned_courses_arr_i
            errors = angle_distance(np.ravel(true_values), np.ravel(predicted_values))
            rmse = np.sqrt(np.mean(errors**2))
            
            # abs_course_fit = pred_wrap #pred = (slope*ii + intercept + pis*2*np.pi).value
            # abs_course_fit_x = np.cos(abs_course_fit)
            # abs_course_fit_y = np.sin(abs_course_fit)
            # abs_course_fit_xy = np.vstack((abs_course_fit_x, abs_course_fit_y))
            
            # warped_fit = AM[i]@abs_course_fit_xy
            # course_warped_fit = np.arctan2(warped_fit[1,:], warped_fit[0,:])

            # true_values = aligned_courses_arr_i
            # errors = angle_distance(np.ravel(true_values), np.ravel(course_warped_fit))
            # predicted_values = true_values - errors

            true_values_by_trajec.append(true_values)
            predicted_values_by_trajec.append(predicted_values)
            ix_arr_is.append(ix_arr_i)
            
            rho = get_circular_rho(true_values, predicted_values)


            
            # find rho in the coordiantes of the fit -- more uniform distribution
            # true_xy = np.vstack((np.cos(aligned_courses_arr_i), np.sin(aligned_courses_arr_i)))
            # true_xy_unwarp = np.linalg.pinv(AM[i])@true_xy
            # true_unwarped = np.arctan2(true_xy_unwarp[1,:], true_xy_unwarp[0,:])
            # pred_values = (slope[i]*ix_arr_i + intercept)
            # rho = get_circular_rho(true_unwarped, pred_values)

            
            rhos.append(rho)
            rmses.append(rmse)

        return rhos, ix_arr_is, true_values_by_trajec, predicted_values_by_trajec, rmses