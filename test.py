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

def get_neighborhood(x, y, t, events, neighborhood_size):
    """
    Finds the indices of events within a spatiotemporal neighborhood of a given event.

    Parameters:
    x, y, t: Coordinates of the event for which to find the neighborhood.
    events: List of all events (x, y, t).
    neighborhood_size: Spatial extent of the neighborhood around the event.

    Returns:
    Indices of events within the neighborhood.
    """

    # Convert the list of events into numpy arrays for efficient computation
    all_x = np.array([event[0] for event in events])
    all_y = np.array([event[1] for event in events])
    all_t = np.array([event[2] for event in events])

    # Calculate squared distances from the event to all other events
    distances = (all_x - x)**2 + (all_y - y)**2

    # Determine which events are within the spatial neighborhood_size
    spatial_neighbors = np.where(distances <= neighborhood_size**2)[0]

    # Optionally, if you want to include a temporal component, you can define a time window
    # For example, within ±delta_t of the event time:
    # delta_t = some_value  # Define a suitable time window
    # temporal_neighbors = np.where(abs(all_t - t) <= delta_t)[0]

    # Combine spatial and temporal criteria (if temporal is used, otherwise just use spatial)
    # neighborhood_indices = np.intersect1d(spatial_neighbors, temporal_neighbors)
    neighborhood_indices = spatial_neighbors  # Use this if only considering spatial criteria

    return neighborhood_indices

# Example usage
events = [(1, 2, 0.1), (2, 3, 0.2)]  # Define your events list
local_flows = compute_local_flow(events)
print(local_flows)