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

def plane_fitting(x, y, t, events, neighborhood_size):
    """
    Fits a plane to the neighborhood of a given event using least squares method.

    Parameters:
    x, y, t: coordinates of the event for which to calculate the flow.
    events: list of all events (x, y, t) to search for neighbors.
    neighborhood_size: the spatial extent of the neighborhood around the event.

    Returns:
    a, b, c: coefficients of the fitted plane, or None if fitting is not possible.
    """

    # Find the neighborhood indices for the event
    neighborhood_indices = get_neighborhood(x, y, t, events, neighborhood_size)

    # Extract the neighborhood points
    x_n = [events[i][0] for i in neighborhood_indices]
    y_n = [events[i][1] for i in neighborhood_indices]
    ts_n = [events[i][2] for i in neighborhood_indices]

    # Ensure there are enough points to fit a plane
    if len(x_n) < 3:
        return None, None, None  # Not enough points to define a plane

    # Setup the design matrix A for the least squares problem
    A = np.c_[x_n, y_n, np.ones(len(x_n))]
    B = np.array(ts_n)

    # Solve the least squares problem
    coefficients, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    # coefficients are [a, b, c] for the plane equation z = ax + by + c
    return coefficients if coefficients.size else (None, None, None)

# Note: This function expects a definition of get_neighborhood(x, y, t, events, neighborhood_size)
# and proper event structure. Adapt and test with your specific data and requirements.

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

#############################First methode to get neighbors
import bisect
def find_spatiotemporal_neighbors(x, y, ts, e, N, dt):
    """
    Find the spatiotemporal neighbors of an event within a 3D space.

    Parameters:
    x, y, ts (array-like): The x, y, and ts coordinates of the events.
    e (int): The index of the event of interest.
    N (float): The spatial extent of the neighborhood (in x and y).
    dt (float): The temporal extent of the neighborhood (in ts).

    Returns:
    Indices of the neighboring events within the specified window.
    """

    # Find the temporal window using binary search for efficiency
    up_indice = bisect.bisect_right(ts, ts[e] + dt)
    down_indice = bisect.bisect_left(ts, ts[e] - dt)
    time_window = np.arange(down_indice, up_indice, 1)

    # Find neighbors within the spatial window
    spatial_nei = time_window[np.where((x[e] - N <= x[time_window]) & (x[time_window] <= x[e] + N))[0]]
    spatial_nei = spatial_nei[np.where((y[e] - N <= y[spatial_nei]) & (y[spatial_nei] <= y[e] + N))[0]]

    return spatial_nei

####################### Seconde methode to get neighbors
def find_range(array, value, window):
    """
    A manual implementation to find the range around a value within an array.
    """
    low, high = 0, len(array)
    while low < high:
        mid = (low + high) // 2
        if array[mid] < value - window:
            low = mid + 1
        elif array[mid] > value + window:
            high = mid
        else:
            # Now iterate outwards from the mid point
            l, r = mid, mid
            while l > 0 and array[l] >= value - window:
                l -= 1
            while r < len(array) and array[r] <= value + window:
                r += 1
            return np.arange(l + 1, r)  # Return the range of indices
    return np.array([])  # Return an empty array if nothing is found

def find_3d_neighbors(coord_x, coord_y, time_stamps, target_idx, spatial_window, time_window):
    """
    Find the spatiotemporal neighbors of an event within a 3D space.

    Parameters:
    coord_x, coord_y, time_stamps (array-like): The x, y, and time_stamps coordinates of the events.
    target_idx (int): The index of the event of interest.
    spatial_window (float): The spatial extent of the neighborhood (in x and y).
    time_window (float): The temporal extent of the neighborhood (in time_stamps).

    Returns:
    Indices of the neighboring events within the specified window.
    """

    # Find the temporal window using a manual range finding method
    temporal_window = find_range(time_stamps, time_stamps[target_idx], time_window)

    # Find neighbors within the spatial window
    spatial_neighbors = temporal_window[np.where((coord_x[target_idx] - spatial_window <= coord_x[temporal_window]) & 
                                                 (coord_x[temporal_window] <= coord_x[target_idx] + spatial_window))[0]]
    spatial_neighbors = spatial_neighbors[np.where((coord_y[target_idx] - spatial_window <= coord_y[spatial_neighbors]) & 
                                                   (coord_y[spatial_neighbors] <= coord_y[target_idx] + spatial_window))[0]]

    return spatial_neighbors





# Example usage
events = [(1, 2, 0.1), (2, 3, 0.2)]  # Define your events list
local_flows = compute_local_flow(events)
print(local_flows)