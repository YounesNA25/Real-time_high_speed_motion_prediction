import numpy as np
from tqdm import tqdm
import scipy.io as sio

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


def local_plane_fitting(x, y, ts, event_idx, neighborhood_size=5, time_threshold=1000):
    """
    Implementing local plane fitting with iterative refinement for event-based data, assuming a spatiotemporal neighborhood.
    This function estimates the plane parameters fitting locally around a specified event and refines the fit iteratively.

    Parameters:
        x, y: Arrays of x and y coordinates of the events.
        ts: Array of timestamps for each event.
        event_idx: Index of the event to consider for local plane fitting.
        neighborhood_size: Dimension of the square neighborhood (default is 5x5).
        time_threshold: Duration to consider for temporal window around the event (default 1000 microseconds).

    Returns:
        final_plane_params: Parameters (a, b, c) of the refined fitted plane.
        neighborhood_idxs: Indices of events in the final neighborhood after refinement.
    """
    # Define spatiotemporal neighborhood boundaries
    # print(event_idx)
    neighborhood_idxs = find_3d_neighbors(x, y, ts, event_idx, spatial_window=neighborhood_size, time_window= time_threshold)
    # print(neighborhood_idxs)
    # print(neighborhood_idxs)
    # Prepare matrices for plane fitting
    if len(neighborhood_idxs) >= 4:  # Need at least 4 points to fit a plane
        A = np.c_[x[neighborhood_idxs], y[neighborhood_idxs], np.ones(neighborhood_idxs.shape)]
        B = ts[neighborhood_idxs]

        # Fit the plane using least squares
        plane_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    else:
        return None, neighborhood_idxs  # Not enough points to fit a plane

    # Iterative refinement
    eps = 10e6  # Set to a large initial value
    th1, th2 = 1e-5, 0.05  # Thresholds for stopping criterion and outlier rejection

    while eps > th1:
        # Calculate distances from points to the plane
        distances = np.abs(np.dot(A, plane_params) - B)

        # Identify inliers as those within th2 distance from the plane
        inliers = distances <= th2

        # Check if there are enough inliers to continue
        if np.sum(inliers) < 4:
            break

        # Refit the plane using only inliers
        A = A[inliers]
        B = B[inliers]
        new_plane_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        # Calculate change in plane parameters
        eps = np.linalg.norm(new_plane_params - plane_params)
        plane_params = new_plane_params  # Update the plane parameters for the next iteration
        #inliers = np.where(inliers)
        # # Update neighborhood indices to only include inliers
        neighborhood_idxs = neighborhood_idxs[inliers]
    return plane_params, neighborhood_idxs




def robust_multi_scale_optical_flow(x_coords: np.ndarray, 
                                    y_coords: np.ndarray, 
                                    timestamps: np.ndarray, 
                                    neighborhood_size: int = 5, 
                                    time_threshold: int = 1000):
    """
    @ parameters:
    -------------
        x_coords, y_coords: np.arrays of x and y coordinates.
        timestamps: np.array of timestamps for each event.
        neighborhood_size: Size of the neighborhood for local flow calculation.
        time_threshold: Past time window to consider for multi-scale pooling.
    
    @ return:
    ---------
        local_flow: Calculated local flow for each point.
        corrected_flow: Flow after multi-scale pooling and correction.
    """
    
    assert x_coords.shape == y_coords.shape == timestamps.shape
    local_flow = np.zeros((len(x_coords), 2))
    corrected_flow = np.zeros((len(x_coords), 2))
    ts_seconds = timestamps * 1e-6  # convertto seconds
    
    for idx, (x, y, t) in tqdm(enumerate(zip(x_coords, y_coords, timestamps)), total=len(x_coords)):
        # Compute local flow using plane fitting method
        plane_params, neighbors = local_plane_fitting(x_coords, y_coords, ts_seconds*1e6 , idx)
        if plane_params is None:
            continue
        
        a, b = plane_params[:2]
        flow_magnitude = np.linalg.norm([a, b])
        z_hat = np.sqrt(a**2 + b**2)
        inlier_count = np.sum(np.abs((ts_seconds[neighbors] - t) - ((a * (x_coords[neighbors] - x)) + (b * (y_coords[neighbors] - y)))) < z_hat / 2)
        
        if inlier_count >= (0.5 * neighborhood_size**2):
            angle = np.arctan2(b, a)
            local_flow[idx] = np.array([flow_magnitude, angle])
        
        # Multi-spatial scale max-pooling
        relevant_events = (timestamps >= t - time_threshold) & (timestamps <= t + time_threshold)
        event_window = np.where(relevant_events)[0]
        U_mean_values = []
        angle_mean_values = []
        sigma_values = range(10, 100, 10)  # Define spatial scales
        
        for sigma in sigma_values:
            spatial_window = event_window[(np.abs(x_coords[event_window] - x) <= sigma) & (np.abs(y_coords[event_window] - y) <= sigma)]
            
            if spatial_window.size > 0:
                U_mean = np.mean(local_flow[spatial_window, 0])
                angle_mean = np.mean(local_flow[spatial_window, 1])
            else:
                U_mean = 0
                angle_mean = 0
            
            U_mean_values.append(U_mean)
            angle_mean_values.append(angle_mean)
        
        best_sigma_index = np.argmax(U_mean_values)
        
        # Update flow based on best spatial scale
        best_U = U_mean_values[best_sigma_index]
        best_angle = angle_mean_values[best_sigma_index]
        corrected_flow[idx] = np.array([best_U, best_angle])
    
    return local_flow, corrected_flow
compute = True
nb_images_to_show = 100
# Retrieving the current path
name_data_file = 'datamat.mat'
# Loading the .mat file
data = sio.loadmat(name_data_file)
N_event_to_load = 200000
# Access to the data in the .mat file
ts = data['ts'].reshape(-1)#[:N_event_to_load]
x  = data['x'] .reshape(-1)#[:N_event_to_load]
y  = data['y'] .reshape(-1)#[:N_event_to_load]

if compute:
    # estimate the local and the corrected flow
    flow_local, corrected_flow = robust_multi_scale_optical_flow(x, y, ts)
    # Save data into data folder
    np.save('flow_local_out.npy'  , flow_local   )
    np.save('corrected_flow_out.npy', corrected_flow)