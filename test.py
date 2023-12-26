import numpy as np
import math

# Assuming the existence of a function plane_fitting that returns [a, b, c] for the plane fitting
# and a function to get neighborhood events, get_neighborhood(x, y, t, size)

def compute_local_flow(events, neighborhood_size=5):
    local_flows = []
    for x, y, t in events:
        # 1. COMPUTE LOCAL FLOW (EDL):
        # Apply plane fitting to estimate the plane parameters [a, b, c]
        a, b, c = plane_fitting(x, y, t, neighborhood_size)

        U_hat = np.linalg.norm([a, b])
        z_hat = math.sqrt(a**2 + b**2)
        inliers_count = 0

        for xi, yi, ti in get_neighborhood(x, y, t, neighborhood_size):
            t_hat = (a * (xi - x)) + (b * (yi - y))
            if abs(ti - t_hat) < (z_hat / 2):
                inliers_count += 1

        if inliers_count >= 0.5 * neighborhood_size**2:
            theta = math.atan2(b, a)  # Arctan(y/x) for correct quadrant
            Un = (U_hat, theta)
        else:
            Un = (0, 0)

        # 2. MULTI-SPATIAL SCALE MAX-POOLING:
        # Define S = {σk}, the set of neighborhoods
        # ... Assuming a function to define S and get neighborhoods at different scales
        # ... Assuming a way to calculate time-past threshold 'tpast'

        if Un != (0, 0):
            U_sigma = []
            for sigma_k in S:  # S and tpast need to be defined based on your context
                # Calculate mean flow in neighborhood σk
                # ... Assuming a function to retrieve events in σk
                U_n_sigma_k = mean([plane_fitting(xi, yi, ti, neighborhood_size) for xi, yi, ti in neighborhood_sigma_k])
                U_sigma.append(U_n_sigma_k)

            # Identify σmax, the scale with maximum mean magnitude of flow
            sigma_max = max(U_sigma, key=lambda u: u[0])

        # 3. UPDATE FLOW:
        # ... This step requires accumulating flows from σmax and updating them
        # ... Assuming a function or method to update flow (x,y) based on calculated values

    return local_flows

import numpy as np

def plane_fitting(x: np.ndarray, y: np.ndarray, ts: np.ndarray):
    """
    Performs a least squares plane fit to the given points.
    
    @param x: np.ndarray of x coordinates.
    @param y: np.ndarray of y coordinates.
    @param ts: np.ndarray of time stamps or third dimension coordinates.
    
    @return: (a, b, c) coefficients of the fitted plane of the form z = ax + by + c.
             Returns None if fitting is not possible.
    """
    # Construct the design matrix with x, y, and a column of ones for the intercept term.
    A = np.c_[x, y, np.ones(x.shape[0])]
    
    # Use least squares to solve for the plane coefficients.
    # np.linalg.lstsq returns several values; we only want the coefficients here.
    coefficients, _, _, _ = np.linalg.lstsq(A, ts, rcond=None)
    
    # coefficients are [a, b, c] of the plane equation.
    return coefficients if coefficients.size else None

# Example usage
# Assuming x, y, ts are defined as arrays of your data points
# plane = plane_fitting(x, y, ts)
# print(plane)  # Outputs: array([a, b, c])


def get_neighborhood(x: np.ndarray, y: np.ndarray, ts: np.ndarray, event: int, N: int = 3, dt: int = 1000):
    """
    Determines the neighborhood of a given event within spatial and temporal constraints.
    
    @param x: np.ndarray of x coordinates.
    @param y: np.ndarray of y coordinates.
    @param ts: np.ndarray of timestamps.
    @param event: The index of the event around which to find the neighborhood.
    @param N: Spatial neighborhood size (distance in both x and y).
    @param dt: Temporal window size (time before and after the event timestamp).
    
    @return: Indices of the events in the neighborhood.
    """
    # Convert ts to microseconds for comparison and define time window
    ts_micro = ts * 10**6  # Assuming ts is in seconds and needs conversion to microseconds
    event_time = ts_micro[event]
    
    # Define temporal window around the event
    index_right = np.searchsorted(ts_micro, event_time + dt, side='right')
    index_left = np.searchsorted(ts_micro, event_time - dt, side='left')
    time_window = np.arange(index_left, index_right, 1)

    # Define spatial window around the event
    spatial_window = np.where(
        (x[event] - N <= x[time_window]) & (x[time_window] <= x[event] + N) &
        (y[event] - N <= y[time_window]) & (y[time_window] <= y[event] + N)
    )[0]

    # The neighborhood is the intersection of the spatial and temporal window
    neighborhood = time_window[spatial_window]

    return neighborhood

# Example usage
events = [(1, 2, 0.1), (2, 3, 0.2)]  # Define your events list
local_flows = compute_local_flow(events)
print(local_flows)