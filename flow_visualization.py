import numpy as np
import cv2 as cv
import scipy.io as sio
def calculate_image(x,y,flow_data, indices, dimensions):
    
    """
        Calculate the visualization image from the flow data.

        Parameters:
        - flow_data (np.ndarray): The flow data array where each row corresponds to a point and the second column contains flow values.
        - indices (np.ndarray): Array of indices to process in the current step.
        - dimensions (tuple): The dimensions (width, height) of the visualization image.

        Returns:
        - np.ndarray: The resulting image in BGR format ready for display.
    """
    image = np.zeros((dimensions[1], dimensions[0], 3), dtype=np.uint8)
    flow = flow_data[indices, 1]  # flow_data[:, 1] contient les valeurs de flux
    flow = np.uint8(flow * 255 / (2 * np.pi))

    x_coords = x[indices].astype(int)
    y_coords = y[indices].astype(int)

    for i, (x_coord, y_coord) in enumerate(zip(x_coords, y_coords)):
        if 0 <= x_coord < dimensions[0] and 0 <= y_coord < dimensions[1]:
            if flow[i] > 0:  # Si la valeur de flux est non nulle
                image[y_coord, x_coord] = [flow[i], 255, 255]  

    return cv.cvtColor(image, cv.COLOR_HSV2BGR)

def visualize_flow(x, y, ts, EDL, ARMS, time_delay, step_size):
    """
        Visualize the flow data using OpenCV.

        Parameters:
        - x (np.ndarray): The x coordinates of the flow data points.
        - y (np.ndarray): The y coordinates of the flow data points.
        - ts (np.ndarray): The timestamps for the flow data points.
        - EDL (np.ndarray): The EDL flow data.
        - ARMS (np.ndarray): The ARMS flow data.
        - time_delay (int): Delay in milliseconds between frames.
        - step_size (int): Number of data points to process in each step.

        The function displays two windows showing the flow visualization for EDL and ARMS. Press 'q' to quit the display loop.
    """
    x = x.astype(int)
    y = y.astype(int)

    height, width = max(y) + 1, max(x) + 1

    for ind_min in range(0, len(x), step_size):
        ind_max = min(ind_min + step_size, len(x))
        indices = np.arange(ind_min, ind_max)
        # Mise à jour des images EDL et ARMS
        EDL_image = calculate_image(x,y,EDL, indices, (width, height))
        ARMS_image = calculate_image(x,y,ARMS, indices, (width, height))
        cv.imshow('EDL Flow', EDL_image)
        cv.imshow('ARMS Flow', ARMS_image)

        if cv.waitKey(time_delay) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

if __name__ == "__main__":
    name_data_file = 'datamat.mat'
    data = sio.loadmat(name_data_file)
    # Access to the data in the .mat file
    ts = data['ts'].reshape(-1)
    x  = data['x'] .reshape(-1)
    y  = data['y'] .reshape(-1)
    flow_local = np.load('flow_local_out.npy')
    corrected_flow = np.load('corrected_flow_out.npy')
    visualize_flow(x, y, ts, flow_local, corrected_flow, time_delay=1000, step_size=1000)