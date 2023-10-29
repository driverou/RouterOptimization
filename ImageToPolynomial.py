import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Set Seaborn's default aesthetics
sns.set()
sns.set_style("white")

def plot_3d_image(img_smooth):
    
    # Get the dimensions of the image
    h, w = img_smooth.shape

    # Create arrays of x, y coordinates
    y, x = np.mgrid[:h, :w]

    # Create a figure
    fig = plt.figure(figsize=(10, 8))

    # Create 3D axes
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface for smoothed image
    surf = ax.plot_surface(x, y, img_smooth, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.7)

    # Set labels and title
    ax.set_xlabel('X-axis', labelpad=10)
    ax.set_ylabel('Y-axis', labelpad=10)
    ax.set_zlabel('Pixel Value', labelpad=10)
    ax.set_title('3D Plot of Smoothed Image', pad=20)

    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Smoothed Image Value')
    ax.grid(False)  # Hide gridlines
    plt.show()
def plot_3d_polynomial(model, h, w):
    # Create arrays of x, y coordinates
    y, x = np.mgrid[:h, :w]

    # Create a grid of coordinates covering the image
    coordinates = np.hstack((x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)))

    # Evaluate the polynomial at each point in the grid
    z_fit = model.predict(coordinates).reshape(h, w)

    # Create a figure
    fig = plt.figure(figsize=(10, 8))

    # Create 3D axes
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface for fitted polynomial
    surf = ax.plot_surface(x, y, z_fit, cmap='viridis', linewidth=0, antialiased=True, alpha=0.7)

    # Set labels and title
    ax.set_xlabel('X-axis', labelpad=10)
    ax.set_ylabel('Y-axis', labelpad=10)
    ax.set_zlabel('Pixel Value', labelpad=10)
    ax.set_title('3D Plot of Fitted Polynomial', pad=20)

    # Add a colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label='Fitted Polynomial Value')
    ax.grid(False)  # Hide gridlines
    plt.show()
def image_to_polynomial(image_path, degree=2, blur_size=5, blur_sigma=1.5):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:  # Check if image loaded successfully
        raise ValueError(f'Failed to load image at path: {image_path}')
    img = 255 - img
    # Apply Gaussian blur to smooth the image
    img_smooth = cv2.GaussianBlur(img, (blur_size, blur_size), blur_sigma)

    # Get the dimensions of the image
    h, w = img_smooth.shape

    # Create arrays of x, y coordinates
    y, x = np.mgrid[:h, :w]

    # Flatten x, y coordinate arrays and grayscale image array for regression
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    z = img_smooth.flatten().reshape(-1, 1)  # Use the smoothed image here

    # Stack x and y coordinate arrays horizontally
    coordinates = np.hstack((x, y))

    # Create a pipeline of polynomial feature extraction and linear regression
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Fit the model to the data
    model.fit(coordinates, z)

    # Get the feature powers from PolynomialFeatures
    poly_features = model.named_steps['polynomialfeatures']
    powers = poly_features.powers_

    # Get the coefficients from LinearRegression
    linear_model = model.named_steps['linearregression']
    coef = linear_model.coef_.flatten()[1:]  # Exclude the bias term
    intercept = linear_model.intercept_[0]

    # Print the polynomial
    poly_str = f"Polynomial: f(x, y) = {intercept}"
    for i, power in enumerate(powers):
        if i < len(coef):  # Ensure that i is within bounds of the coef array
            poly_str += f" + {coef[i]}*x^{power[0]}*y^{power[1]}"
    print(poly_str)

    return model, img_smooth
def visualize_topology(image_path, model, img_smooth, degree, blur_size, blur_sigma):
    # Create a grid of x, y coordinates covering the image
    h, w = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).shape
    y, x = np.mgrid[:h, :w]
    coordinates = np.hstack((x.flatten().reshape(-1, 1), y.flatten().reshape(-1, 1)))

    # Evaluate the polynomial at each point in the grid
    z_fit = model.predict(coordinates).reshape(h, w)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cmap='gray', origin='upper')
    plt.title('Original Topology')
    plt.axis('off')  # Hide axes

    plt.subplot(1, 3, 2)
    plt.imshow(img_smooth, cmap='gray', origin='upper')
    plt.title(f'Smoothed Topology\nBlur Size: {blur_size}, Blur Sigma: {blur_sigma}')
    plt.axis('off')  # Hide axes

    plt.subplot(1, 3, 3)
    im = plt.imshow(z_fit, cmap='viridis', origin='upper')
    plt.title(f'Fitted Topology\nPolynomial Degree: {degree}')
    plt.axis('off')  # Hide axes
    cbar = plt.colorbar(im, label='Fitted Value')
    cbar.ax.yaxis.set_tick_params(pad=15)  # Adjust colorbar tick label padding

    plt.tight_layout()
    plt.show()



# Usage:
image_path = '/Users/diegorivero/Downloads/Untitled_Artwork 4.jpg'
degree = 7
blur_size = 11
blur_sigma = 5
model, img_smooth = image_to_polynomial(image_path, degree, blur_size, blur_sigma)
visualize_topology(image_path, model, img_smooth, degree, blur_size, blur_sigma)
h, w = img_smooth.shape
plot_3d_image(img_smooth)
plot_3d_polynomial(model, h, w)