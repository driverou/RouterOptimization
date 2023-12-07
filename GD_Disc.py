import numpy as np
import math
import time
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import numba
from numba.typed import List
from tqdm import trange
import torch

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

# def loss_function_squared_distance(matrix, placements):
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
#     squared_distances = np.sum((coords[:, :, None, :] - placements[None, None, :, :]) ** 2, axis=-1)

#     # Avoid division by zero by adding a small value
#     epsilon = 1e-10
#     distances = np.sqrt(1 / (squared_distances + epsilon))

#     # Handle the case where (u, v) == (r, c)
#     mask = squared_distances < epsilon
#     distances[mask] = 2

#     # Compute the loss
#     f = np.sum(matrix[:, :, None] * distances)

#     return f

# @torch.compile
def loss_function_squared_distance(matrix, placements):
    """This function calculates the loss by multiplying the pixel value at (x,y) by the distance to each
    (u,v) pair in placements and then summing it.

    Args:
        matrix (torch.Tensor): an m x n tensor with pixel values as its entries
        placements (torch.Tensor): a k x 2 tensor with (u,v) pairs as its entries
    """    
    m, n = matrix.shape
    k, _ = placements.shape

    # Create coordinate grid for matrix
    x, y = torch.meshgrid(torch.arange(m), torch.arange(n))
    coords = torch.stack((x, y), dim=-1)

    # Compute squared distances between each point in the matrix and each placement
    squared_distances = torch.sum((coords[:, :, None, :] - placements[None, None, :, :]) ** 2, dim=-1)

    # Avoid division by zero by adding a small value
    epsilon = 1e-10
    distances = 1 / torch.sqrt(squared_distances + epsilon)

    # Handle the case where (u, v) == (r, c)
    mask = squared_distances < epsilon
    distances[mask] = 2.0

    # Move tensors to GPU
    matrix = matrix.cuda()
    distances = distances.cuda()

    # Compute the loss
    f = torch.sum(matrix[:, :, None] * distances)

    return f

# @torch.compile
# def loss_function(matrix, placements):
#     """
#     Calculate the weighted sum of pixel values based on their distance to the nearest placement.

#     Parameters:
#     - matrix (np.array): m x n matrix of pixel values (0, 255).
#     - placements (np.array): k x 2 array where each row is a pair of (r, c) coordinates.

#     Returns:
#     - float: The weighted sum.
#     """
#     m, n = matrix.shape
#     rows, cols = np.indices((m, n))
    
#     # Calculate distances from each pixel to each placement
#     distances = np.sqrt((placements[:, 0][:, None, None] - rows)**2 + 
#                         (placements[:, 1][:, None, None] - cols)**2)
    
#     # Add epsilon to distance if pixel location = placement location
#     distances[distances == 0] += 0.1
    
#     # Find the minimum distance for each pixel
#     min_distances = np.min(distances, axis=0)
    
#     # Calculate the weight based on the nearest placement
#     weights = 1 / min_distances
    
#     return np.sum(matrix * weights)


def loss_function(matrix, placements):
    """
    Calculate the weighted sum of pixel values based on their distance to the nearest placement.

    Parameters:
    - matrix (torch.Tensor): m x n tensor of pixel values (0, 255).
    - placements (torch.Tensor): k x 2 tensor where each row is a pair of (r, c) coordinates.

    Returns:
    - float: The weighted sum.
    """
    m, n = matrix.shape
    rows, cols = torch.meshgrid(torch.arange(m, device='cuda:0'), torch.arange(n, device='cuda:0'))

    # Calculate distances from each pixel to each placement
    distances = torch.sqrt((placements[:, 0][:, None, None] - rows)**2 + 
                           (placements[:, 1][:, None, None] - cols)**2)

    # Add epsilon to distance if pixel location = placement location
    distances[distances == 0] += 0.1

    # Find the minimum distance for each pixel
    min_distances, _ = torch.min(distances, dim=0)

    # Calculate the weight based on the nearest placement
    weights = 1 / min_distances

    # Move tensors to GPU
    matrix = matrix.cuda()
    weights = weights.cuda()

    return torch.sum(matrix * weights).item()



# @torch.compile
# def new_placement(loss_function, matrix, placements, multiplier=1):
#     m, n = matrix.shape
#     k, _ = placements.shape
#     changes = np.zeros((k, 2))
    
#     current_val = loss_function(matrix, placements)
    
#     for i in range(k):
#         for j in range(2):
#             cur_max = current_val
#             new_placement = placements.copy()
#             new_placement[i, j] -= 1
            
#             if 0 <= new_placement[i, j] < (m if j == 0 else n):
#                 if loss_function(matrix, new_placement) > cur_max:
#                     changes[i, j] = -1
#                     cur_max = loss_function(matrix, new_placement)
            
#             new_placement[i, j] += 2
#             if 0 <= new_placement[i, j] < (m if j == 0 else n):
#                 if loss_function(matrix, new_placement) > cur_max:
#                     changes[i, j] = 1
    
#     return placements + changes * multiplier

def new_placement(loss_function, matrix, placements, multiplier=1):
    m, n = matrix.shape
    k, _ = placements.shape
    changes = torch.zeros((k, 2))

    current_val = loss_function(matrix, placements)

    for i in range(k):
        for j in range(2):
            cur_max = current_val
            new_placement = placements.clone()
            new_placement[i, j] -= 1

            if 0 <= new_placement[i, j] < (m if j == 0 else n):
                if loss_function(matrix, new_placement) > cur_max:
                    changes[i, j] = -1
                    cur_max = loss_function(matrix, new_placement)

            new_placement[i, j] += 2
            if 0 <= new_placement[i, j] < (m if j == 0 else n):
                if loss_function(matrix, new_placement) > cur_max:
                    changes[i, j] = 1

    # Move tensors to GPU
    placements = placements.cuda()
    changes = changes.cuda()

    return placements + changes * multiplier



# def generate_random_array(k, m, n):
#     first_column = np.random.randint(0, m, k)
#     second_column = np.random.randint(0, n, k)
#     return np.column_stack((first_column, second_column))


def generate_random_array(k, m, n):
    first_column = torch.randint(0, m, (k,), device='cuda:0')
    second_column = torch.randint(0, n, (k,), device='cuda:0')
    return torch.stack((first_column, second_column), dim=1)





# @torch.compile
# def gradient_decent(loss_function, gradient,matrix, num_routers = 2, iters = 500, attempts = 20, multiplier = 10):
#     m, n = matrix.shape
#     max_val_achieved = 0
#     best_placement = None
#     history = []  # List to store the progression of placements

#     for _ in range(attempts):
#         placement = generate_random_array(num_routers, m, n)
#         placement_history = [placement.copy()]  # Store the initial placement for this attempt

#         for i in trange(iters):
#             mult = math.floor(iters/(i+1))*multiplier
#             placement = new_placement(loss_function, matrix, placement, mult)
#             placement_history.append(placement.copy())  # Append the new placement to the history
#         if loss_function(matrix, placement) > max_val_achieved:
#             max_val_achieved = loss_function(matrix, placement)
#             best_placement = placement
#             history = placement_history  # Update the main history with the current attempt's history

#     return best_placement, max_val_achieved, history



def gradient_decent(loss_function, new_placement, matrix, num_routers=2, iters=500, attempts=20, multiplier=10):
    m, n = matrix.shape
    max_val_achieved = 0
    best_placement = None
    history = []  # List to store the progression of placements

    for _ in range(attempts):
        placement = generate_random_array(num_routers, m, n)
        placement_history = [placement.clone()]  # Store the initial placement for this attempt

        for i in trange(iters):
            mult = math.floor(iters / (i + 1)) * multiplier
            placement = new_placement(loss_function, matrix, placement, mult)
            placement_history.append(placement.clone())  # Append the new placement to the history
        if loss_function(matrix, placement) > max_val_achieved:
            max_val_achieved = loss_function(matrix, placement)
            best_placement = placement.clone()
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
        plt.scatter(cols + 0.5, rows + 0.5, color=colors[idx], s=100, marker='X')
        
        # If it's not the first placement, draw a line connecting the current placement to the previous one
        if idx > 0:
            prev_rows, prev_cols = history[idx-1][:, 0], history[idx-1][:, 1]
            for pr, pc, r, c in zip(prev_rows, prev_cols, rows, cols):
                plt.plot([pc + 0.5, c + 0.5], [pr + 0.5, r + 0.5], color=colors[idx], linestyle='-', linewidth=2)
    
    # Invert the y-axis to match the array indexing
    plt.gca().invert_yaxis()
    
    # Show the plot
    plt.show()

@numba.jit(nopython=True)
def compute_batch_errors(matrix, batch_rows, batch_cols, m, n):
    """
    Compute the errors for a batch of pixels.
    """
    batch_size = len(batch_rows)
    batch_errors = np.zeros(batch_size)
    
    for idx in range(batch_size):
        i, j = batch_rows[idx], batch_cols[idx]
        rows = np.arange(m)
        cols = np.arange(n)
        
        row_dists = np.abs(rows - i)
        col_dists = np.abs(cols - j)
        
        distances = np.sqrt(row_dists[:, None]**2 + col_dists**2)
        
        # Update distances using a loop
        for x in range(m):
            for y in range(n):
                if distances[x, y] == 0:
                    distances[x, y] += 0.1
        
        weights = 1 / distances
        
        batch_errors[idx] = np.sum(matrix * weights)
    
    return batch_errors


def compute_errors(matrix):
    """
    Compute the errors for all placements in the matrix using parallel processing.
    """
    m, n = matrix.shape
    errors = np.zeros((m, n))
    
    # Create batches of pixel placements
    batch_size = 100
    total_pixels = m * n
    num_batches = (total_pixels + batch_size - 1) // batch_size
    batches = [(list(range(i*batch_size, min((i+1)*batch_size, total_pixels))),) for i in range(num_batches)]
    
    # Parallel computation of errors for each batch of pixel placements
    results = Parallel(n_jobs=-1)(delayed(compute_batch_errors)(matrix, 
                                                                [idx // n for idx in batch[0]], 
                                                                [idx % n for idx in batch[0]], 
                                                                m, n) 
                                  for batch in batches)
    
    # Populate the errors matrix with the computed results
    for batch, batch_errors in zip(batches, results):
        for idx, error in zip(batch[0], batch_errors):
            errors[idx // n, idx % n] = error
    
    return errors



def plot_placement_error(matrix):
    """
    Plot the error for placing the placement at each pixel in a 3D plot.
    """
    errors = compute_errors(matrix)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    x, y = np.meshgrid(x, y)
    
    ax.plot_surface(x, y, errors, cmap='viridis')
    
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Column Index')
    ax.set_zlabel('Error')
    ax.set_title('Placement Error for Each Pixel')
    
    plt.show()

# matrix = image_to_grayscale("maincampusgray.jpg")
# matrix = torch.from_numpy(matrix).to('cuda:0')
# start_time = time.perf_counter()
# bp, mva, history = gradient_decent(loss_function,new_placement, matrix, num_routers=10, iters=30, attempts=5, multiplier=1)
# print(bp, mva)
# end_time = time.perf_counter()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time} seconds")

matrix = image_to_grayscale("maincampusgray.jpg")
matrix = 255-torch.from_numpy(matrix).to('cuda:0')
start_time = time.perf_counter()
bp, mva, history = gradient_decent(loss_function, new_placement, matrix, num_routers=50, iters=30, attempts=1, multiplier=1)
print(bp, mva)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


# plot_values_and_placements(matrix, bp)
plot_values_and_progression(matrix.cpu(), [x.cpu() for x in history])
