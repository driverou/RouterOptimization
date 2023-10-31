import numpy as np
import math
import time
import cv2
import seaborn as sns
import matplotlib.pyplot as plt



def image_to_grayscale(image_path, blur_size=5, blur_sigma=1.5):
    """
    Load an image from the given path, convert it to grayscale, invert its colors, 
    and apply Gaussian blur.

    Args:
        image_path (str): Path to the image file.
        blur_size (int, optional): Size of the Gaussian blur kernel. Defaults to 5.
        blur_sigma (float, optional): Standard deviation of the Gaussian blur kernel. Defaults to 1.5.

    Returns:
        np.ndarray: Grayscale image pixel values after preprocessing.
    """
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image loaded successfully
    if img is None:
        raise ValueError(f'Failed to load image at path: {image_path}')
    
    # Invert the grayscale image colors
    img = 255 - img
    
    # Apply Gaussian blur to smooth the image
    img_smooth = cv2.GaussianBlur(img, (blur_size, blur_size), blur_sigma)
    
    return img_smooth

# def loss_function(matrix, placements):
#     """This function calculates the loss by multiplying the pixel value at (x,y) by the distance to each
#     (u,v) pair in placements and then summing it.

#     Args:
#         matrix (nparray): an m x n array with pixel values as its entries
#         placements (nparray): a k x 2 array with (u,v) pairs as its entries
#     """    
#     m, n = matrix.shape
#     k, _ = placements.shape

#     # Create coordinate grid for matrix
#     x, y = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
#     coords = np.stack((x, y), axis=-1)

#     # Compute squared distances between each point in the matrix and each placement
#     squared_distances = np.sum((coords[:, :, np.newaxis, :] - placements[np.newaxis, np.newaxis, :, :]) ** 2, axis=-1)

#     # Avoid division by zero by adding a small value
#     epsilon = 1e-10
#     distances = np.sqrt(1 / (squared_distances + epsilon))

#     # Handle the case where (u, v) == (r, c)
#     mask = squared_distances < epsilon
#     distances[mask] = 2

#     # Compute the loss
#     f = np.sum(matrix[:, :, np.newaxis] * distances)

#     return f
def loss_function(matrix, placements):
    """This function calculates the loss by multiplying the pixel value at (x,y) by the distance to the closest
    (u,v) pair in placements and then summing it.

    Args:
        matrix (nparray): an m x n array with pixel values as its entries
        placements (nparray): a k x 2 array with (u,v) pairs as its entries
    """    
    m, n = matrix.shape
    k, _ = placements.shape

    # Create coordinate grid for matrix
    x, y = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    coords = np.stack((x, y), axis=-1)

    # Compute squared distances between each point in the matrix and each placement
    squared_distances = np.sum((coords[:, :, np.newaxis, :] - placements[np.newaxis, np.newaxis, :, :]) ** 2, axis=-1)

    # Find the index of the closest placement for each pixel
    closest_placement_idx = np.argmin(squared_distances, axis=-1)

    # Compute the distance for each pixel to its closest placement
    distances_to_closest = np.sqrt(1 / (squared_distances[np.arange(m)[:, np.newaxis], np.arange(n), closest_placement_idx] + 1e-10))

    # Handle the case where a pixel's coordinates match a placement's coordinates
    mask = squared_distances[np.arange(m)[:, np.newaxis], np.arange(n), closest_placement_idx] < 1e-10
    distances_to_closest[mask] = 2

    # Compute the loss
    f = np.sum(matrix * distances_to_closest)

    return f

def new_placement(loss_function, matrix, placements):
    """This function takes in a matrix and placements, as well as a loss function, and return a set of new placements
    such that the loss function is maximized and each new placement is within 1 of the old placement. 

    Args:
        loss_function (function): computes the loss function given a matrix and placements
        matrix (nparray): matrix that we are optimizing over
        placements (nparray): (r, c) pairs that are our placements

    Returns:
        new_placements: new placements that maximize loss_function
    """    
    m, n = matrix.shape
    k, _ = placements.shape
    changes = np.zeros((k, 2))
    
    current_val = loss_function(matrix, placements)
    
    for i in range(k):
        for j in range(2):
            cur_max = current_val
            new_placement = placements.copy()
            new_placement[i, j] -= 1
            
            if j == 0 and 0 <= new_placement[i, j] < m:
                if loss_function(matrix, new_placement) > cur_max:
                    changes[i, j] = -1
                    cur_max = loss_function(matrix, new_placement)
            elif j == 1 and 0 <= new_placement[i, j] < n:
                if loss_function(matrix, new_placement) > cur_max:
                    changes[i, j] = -1
                    cur_max = loss_function(matrix, new_placement)
            
            new_placement[i, j] += 2
            if j == 0 and 0 <= new_placement[i, j] < m:
                if loss_function(matrix, new_placement) > cur_max:
                    changes[i, j] = 1
            elif j == 1 and 0 <= new_placement[i, j] < n:
                if loss_function(matrix, new_placement) > cur_max:
                    changes[i, j] = 1
    
    return placements + changes

def generate_random_array(k, m, n):
    first_column = np.random.randint(0, m, k)
    second_column = np.random.randint(0, n, k)
    return np.column_stack((first_column, second_column))

def gradient_decent(loss_function, gradient,matrix, num_routers, iters, attempts):
    m, n = matrix.shape
    max_val_achieved = 0
    best_placement = None
    history = []  # List to store the progression of placements

    for _ in range(attempts):
        placement = generate_random_array(num_routers, m, n)
        placement_history = [placement.copy()]  # Store the initial placement for this attempt

        for i in range(iters):
            placement = new_placement(loss_function, matrix, placement)
            placement_history.append(placement.copy())  # Append the new placement to the history
        if loss_function(matrix, placement) > max_val_achieved:
            max_val_achieved = loss_function(matrix, placement)
            best_placement = placement
            history = placement_history  # Update the main history with the current attempt's history

    return best_placement, max_val_achieved, history

def plot_values_and_placements(values, placements):
    """
    Plot the values of the grid as heights and overlay the placements (r, c) on the image.

    Args:
    - values (np.array): A 2D numpy array representing the values.
    - placements (np.array): A 2D numpy array of shape (k, 2) representing the placements (r, c).
    """
    
    # Plot the heatmap for values
    plt.figure(figsize=(10, 8))
    sns.heatmap(values, cmap='viridis', cbar=True)
    
    # Extract rows and columns from placements
    rows, cols = placements[:, 0], placements[:, 1]
    
    # Plot the placements on top of the heatmap
    plt.scatter(cols + 0.5, rows + 0.5, color='red', s=100, marker='X', label='Placement')
    
    # Invert the y-axis to match the array indexing
    plt.gca().invert_yaxis()
    
    # Display the legend
    plt.legend()
    
    # Show the plot
    plt.show()

def plot_values_and_progression(values, history):
    """
    Plot the values of the grid as heights and overlay the progression of placements (r, c) on the image.

    Args:
    - values (np.array): A 2D numpy array representing the values.
    - history (list): A list of 2D numpy arrays representing the progression of placements.
    """
    
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap for values
    sns.heatmap(values, cmap='viridis', cbar=True)
    
    # Use a colormap to get a range of colors based on the length of history
    colors = plt.cm.jet(np.linspace(0, 1, len(history)))
    
    for idx, placements in enumerate(history):
        # Extract rows and columns from placements
        rows, cols = placements[:, 0], placements[:, 1]
        
        # Plot the placements on top of the heatmap using different colors for progression
        plt.scatter(cols + 0.5, rows + 0.5, color=colors[idx], s=100, marker='X', label=f'Step {idx+1}')
    
    # Invert the y-axis to match the array indexing
    plt.gca().invert_yaxis()
    
    # Display the legend
    plt.legend()
    
    # Show the plot
    plt.show()


# matrix = np.array([[1,2], [3, 4]])
# placements = np.array([[0,0], [0, 1]])
# print(loss_function(matrix, placements))
# placements = new_placement(loss_function, matrix, placements)
# print(placements)
# placements = new_placement(loss_function, matrix, placements)
# print(placements)
# placements = new_placement(loss_function, matrix, placements)
# print(placements)
matrix = image_to_grayscale("/Users/diegorivero/Downloads/Untitled_Artwork 4.jpg")
start_time = time.perf_counter()
bp, mva, history = gradient_decent(loss_function, new_placement, matrix, 2, 5, 1)
print(bp, mva)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


plot_values_and_placements(matrix, bp)
# plot_values_and_progression(matrix, history)