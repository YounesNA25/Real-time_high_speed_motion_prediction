import numpy as np
from tqdm import tqdm
import scipy.io as sio
from utils import select_file
import bisect
import os



def find_range(array, value, window):
    """
    Finds the range of indices in 'array' such that the elements are within a specified 'window' of a given 'value'.
    
    Parameters:
    array (array-like): A sorted array of numerical values.
    value (numeric): The center value to find neighbors around.
    window (numeric): The window size around the 'value' to consider for finding neighbors.

    Returns:
    numpy.ndarray: An array of indices indicating the positions in 'array'.
    """
    left_idx = bisect.bisect_left(array, value - window)
    right_idx = bisect.bisect_right(array, value + window)
    return np.arange(left_idx, right_idx)

def find_3d_neighbors(coord_x, coord_y, time_stamps, target_idx, spatial_window, time_window):
    """
    Optimized version of find_3d_neighbors using efficient numpy operations
    and minimizing redundant calculations.
    """
    # Get the temporal neighbors first using the optimized find_range function
    temporal_window_indices = find_range(time_stamps, time_stamps[target_idx], time_window)

    # Calculate differences once and reuse
    dx = coord_x[temporal_window_indices] - coord_x[target_idx]
    dy = coord_y[temporal_window_indices] - coord_y[target_idx]

    # Use numpy's logical and to find indices that meet both spatial conditions
    spatial_condition = (np.abs(dx) <= spatial_window) & (np.abs(dy) <= spatial_window)

    # Filter the temporal window with the spatial condition
    spatial_neighbors = temporal_window_indices[spatial_condition]

    return spatial_neighbors


def local_plane_fitting(x, y, ts, event_idx, neighborhood_size=3, time_threshold=500):
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
    eps = 10e6  
    thresh1  = 1e-5 
    thresh2 = 0.05  
    neighborhood_idxs = find_3d_neighbors(x, y, ts, event_idx, spatial_window=neighborhood_size, time_window= time_threshold)
    if len(neighborhood_idxs) > 3:
        x_subset = x[neighborhood_idxs]
        y_subset = y[neighborhood_idxs]
        ones_array = np.ones(neighborhood_idxs.shape)
        A = np.column_stack((x_subset, y_subset, ones_array))
        B = ts[neighborhood_idxs]


        plane_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    else:
        return None, neighborhood_idxs  

    while eps > thresh1:
        dist = np.abs(np.dot(A, plane_params) - B)
        inl = dist <= thresh2
        if np.sum(inl) < 4:
            break

        A = A[inl]
        B = B[inl]

        new_plane_params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        eps = np.linalg.norm(new_plane_params - plane_params)
        plane_params = new_plane_params  
        neighborhood_idxs = neighborhood_idxs[inl]
    return plane_params, neighborhood_idxs




def calculate_local_flow(x: np.ndarray, y: np.ndarray, t: np.ndarray, neighborhood_size: int = 3):
    """
    Calculate local flow using local plane fitting for each event.

    Parameters:
    x, y, t : Arrays of coordinates and timestamps of events.
    neighborhood_size: Size of the spatial neighborhood.

    Returns:
    local_flow: Array of local flow vectors for each event.
    """
    local_flow = np.zeros((len(x), 2), dtype=np.float32)
    ts_seconds = t * 1e-6

    for idx, (xi, yi, ti) in tqdm(enumerate(zip(x, y, t)), total=len(x)):
        plane_params, neighbors = local_plane_fitting(x, y, ts_seconds, idx)
        if plane_params is None:
            continue

        a, b = plane_params[:2]
        u_chap = np.linalg.norm([a, b])
        z_chap = np.sqrt(a**2 + b**2)

        # Calculating t_chap, differences, inliers in a vectorized manner
        t_chap = a * (xi - x[neighbors]) + b * (yi - y[neighbors])
        differences = np.abs(ts_seconds[neighbors] - ts_seconds[idx] - t_chap)
        inl = differences < z_chap / 2

        inl_count = np.sum(inl)
        if inl_count >= (0.5 * neighborhood_size**2):
            angle = np.arctan2(a, b)
            local_flow[idx] = np.array([u_chap, angle])
    
    return local_flow


def multi_spatial_scale_maxpooling(x: np.ndarray, y: np.ndarray, t: np.ndarray, local_flow: np.ndarray, time_threshold: int = 500):
    """
    Perform multi-spatial scale max-pooling on the local flow vectors.

    Parameters:
    x, y, t : Arrays of coordinates and timestamps of events.
    local_flow: Array of local flow vectors for each event.
    time_threshold: Time window to consider for each event.

    Returns:
    corrected_flow: Array of flow vectors for each event after max-pooling correction.
    """
    flow = np.zeros_like(local_flow)

    for idx, (xi, yi, ti) in tqdm(enumerate(zip(x, y, t)), total=len(x)):
        temporal_mask = (t >= ti - time_threshold) & (t <= ti + time_threshold)
        spatial_indices = np.nonzero(temporal_mask)[0]

        Un = 0
        angle_n = 0
        max_mean_U = -np.inf 

        for sigma in range(10, 100, 10):
           
            spatial_mask = (np.abs(x[spatial_indices] - xi) <= sigma) & (np.abs(y[spatial_indices] - yi) <= sigma)
            indices_retained = spatial_indices[spatial_mask]
            if indices_retained.size > 0:
                U_mean = np.mean(local_flow[indices_retained, 0])
                mean_angle = np.mean(local_flow[indices_retained, 1])

                if U_mean > max_mean_U:
                    max_mean_U = U_mean
                    Un = U_mean
                    angle_n = mean_angle

        flow[idx] = np.array([Un, angle_n])

    return flow



if __name__ == "__main__":
    try:
        # Retrieving the current path
        # name_data_file = 'datda/datamat.mat'
        name_data_file = select_file()
        data = sio.loadmat(name_data_file)

        # Parameters for visualization
        min_range, max_range = (0, 100000)
        # Access to the data in the .mat file
        ts = data['ts'].reshape(-1)[min_range:max_range]
        x  = data['x'] .reshape(-1)[min_range:max_range]
        y  = data['y'] .reshape(-1)[min_range:max_range]
        flow_local = calculate_local_flow(x, y, ts)
        corrected_flow = multi_spatial_scale_maxpooling(x, y, ts,flow_local)

        # Define the folder name to save the result
        folder_name = "result"
        os.makedirs(folder_name, exist_ok=True)
        # Save data into data folder
        np.save('result/flow_local_out.npy'  , flow_local   )
        np.save('result/corrected_flow_out.npy', corrected_flow)
        print('done.')

    except Exception as e:
        print(f"An error occurred: {e}")