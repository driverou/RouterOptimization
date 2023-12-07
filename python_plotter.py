import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
# Load matrix from matrix.csv
matrix = np.loadtxt("matrix.csv", delimiter=',')

# Load bp from bp.csv
bp = np.loadtxt("bp.csv", delimiter=',')


history_array = np.loadtxt("history.csv", delimiter=",")




def plot_values_and_progression(values, history):
    # Set a dark grid style for Seaborn
    sns.set_style("darkgrid")

    # Create a custom colormap for the heatmap
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    
    # Get the size of the values matrix
    m, n = values.shape
    
    # Define a gradient color range for the progression
    num_steps = len(history)
    colors = sns.color_palette("coolwarm", num_steps)
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))
    k = len(history_array[0]) // 2  # Assuming each history has 2k elements
  
    # Simplified heatmap with a subtle color palette
    sns.heatmap(values, cmap=cmap, ax=ax, cbar=True, cbar_kws={"shrink": 0.75})
    
    # Set title and possibly a subtitle
    ax.set_title("Placement Progression", fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.title.set_position([.5, 1.05])
    
    for idx, placements in enumerate(history_array):
        x_coords = placements[k:] + 0.5  # First half are x-coordinates
        y_coords = placements[:k] + 0.5  # Second half are y-coordinates

        # Determine marker style based on the index
        marker_style = "X" if idx == 0 else "P" if idx == num_steps - 1 else "o"
        label = "Start" if idx == 0 else "End" if idx == num_steps - 1 else None

        ax.scatter(x_coords, y_coords, color=colors[idx], s=100, label=label, marker=marker_style, edgecolors="black")

        # Connect the placements with smooth lines
        if idx > 0:
            prev_placements = history_array[idx-1]
            prev_x_coords = prev_placements[k:] + 0.5
            prev_y_coords = prev_placements[:k] + 0.5
            for px, py, x, y in zip(prev_x_coords, prev_y_coords, x_coords, y_coords):
                ax.plot([px, x], [py, y], color=colors[idx], linewidth=2.5, alpha=0.7)

    # Remove x and y ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a legend
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    
    # Save the plot to a file and show the plot
    plt.tight_layout()
    plt.savefig("placement_progression_plot_python.png", dpi=300)
    plt.show()



print("Started Plotting")
plot_values_and_progression(matrix, history_array)
# plot_values_and_progression_3d(matrix, history_array)
# animate_values_and_progression(matrix, history_array)
# animate_plot_values_and_progression_3d(matrix, history_array)