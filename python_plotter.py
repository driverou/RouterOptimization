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
    
    # Simplified heatmap with a subtle color palette
    sns.heatmap(values, cmap=cmap, ax=ax, cbar=True, cbar_kws={"shrink": 0.75})
    
    # Set title and possibly a subtitle
    ax.set_title("Placement Progression", fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.title.set_position([.5, 1.05])
    
    for idx, placements in enumerate(history):
        # No labels for a cleaner look
        if idx == 0:  # Starting placements
            marker_style = "X"
            label = "Start"
        elif idx == num_steps - 1:  # Ending placements
            marker_style = "P"
            label = "End"
        else:  # Intermediate placements
            marker_style = "o"
            label = None
        
        ax.scatter(placements[1::2] + 0.5, placements[::2] + 0.5, color=colors[idx], s=100, label=label, marker=marker_style, edgecolors="black")
        
        # Connect the placements with smooth lines
        if idx > 0:
            prev_placements = history[idx-1]
            for pr, pc, r, c in zip(prev_placements[::2], prev_placements[1::2], placements[::2], placements[1::2]):
                ax.plot([pc+0.5, c+0.5], [pr+0.5, r+0.5], color=colors[idx], linewidth=2.5, alpha=0.7)
    
    # Remove x and y ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a legend
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    
    # Save the plot to a file and show the plot
    plt.tight_layout()
    plt.savefig("placement_progression_plot_python.png", dpi=300)
    plt.show()

def plot_values_and_progression_3d(values, history):
    # Create a figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the size of the values matrix
    m, n = values.shape
    
    # Create a meshgrid for the values matrix
    x = np.arange(n)
    y = np.arange(m)
    X, Y = np.meshgrid(x, y)
    
    # Plot the surface with improved lighting and shading
    surf = ax.plot_surface(X, Y, values, cmap='viridis', edgecolor='none', alpha=0.7, shade=True, zorder=1)
    
    # Define a gradient color range for the progression
    num_steps = len(history)
    colors = sns.color_palette("coolwarm", num_steps)
    
    # Offset for the markers above the surface
    offset = 0.05 * np.max(values)
    
    for idx, placements in enumerate(history):
        # Extracting the coordinates and values for each placement
        x_coords = placements[1::2]
        y_coords = placements[::2]
        z_coords = [values[int(y), int(x)] + offset for x, y in zip(x_coords, y_coords)]
        
        # Plot the placements as pronounced spheres on the surface
        ax.scatter(x_coords, y_coords, z_coords, color=colors[idx], s=300, depthshade=True, edgecolors="w", linewidth=2, zorder=2)
        
        # Connect the placements with thicker lines in 3D space
        if idx > 0:
            prev_placements = history[idx-1]
            px_coords = prev_placements[1::2]
            py_coords = prev_placements[::2]
            pz_coords = [values[int(y), int(x)] + offset for x, y in zip(px_coords, py_coords)]
            for px, py, pz, x, y, z in zip(px_coords, py_coords, pz_coords, x_coords, y_coords, z_coords):
                ax.plot([px, x], [py, y], [pz, z], color=colors[idx], linewidth=3, alpha=0.8, zorder=3)
    
    # Set labels and title with improved font size and weight
    ax.set_xlabel('X', labelpad=20, fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', labelpad=20, fontsize=14, fontweight='bold')
    ax.set_zlabel('Value', labelpad=20, fontsize=14, fontweight='bold')
    ax.set_title("3D Placement Progression", fontdict={"fontsize": 24, "fontweight": "bold"})
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=30, azim=-45)
    
    # Add grid lines with improved visibility
    ax.zaxis._axinfo["grid"]['linestyle'] = "--"
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, which='both', ls="--", linewidth=0.5, color='gray', alpha=0.5)
    
    # Add a colorbar with improved appearance
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.set_title('Value', fontdict={"fontsize": 14, "fontweight": "bold"})
    cbar.ax.tick_params(labelsize=12)
    
    # Show the plot with a tight layout
    plt.tight_layout(pad=3.0)
    plt.savefig("3d_placement_progression_plot.png", dpi=300)
    plt.show()



def animate_values_and_progression(values, history):
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
    
    # Simplified heatmap with a subtle color palette
    sns.heatmap(values, cmap=cmap, ax=ax, cbar=True, cbar_kws={"shrink": 0.75})
    
    # Set title and possibly a subtitle
    ax.set_title("Placement Progression", fontdict={"fontsize": 20, "fontweight": "bold"})
    ax.title.set_position([.5, 1.05])
    
    # Remove x and y ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Initialize scatter and line objects
    scatters = [ax.scatter([], [], color=colors[i], s=100, edgecolors="black") for i in range(num_steps)]
    num_placements = len(history[0]) // 2
    lines = [ax.plot([], [], color=colors[i], linewidth=2.5, alpha=0.7)[0] for i in range(num_placements)]
    from scipy.interpolate import interp1d

    # Function to interpolate points between frames
    def interpolate_history(history, factor=5):
        interpolated_history = []
        for i in range(len(history) - 1):
            x0, y0 = history[i][::2], history[i][1::2]
            x1, y1 = history[i+1][::2], history[i+1][1::2]
            
            f_x = interp1d([0, 1], [x0, x1], axis=1)
            f_y = interp1d([0, 1], [y0, y1], axis=1)
            
            for t in np.linspace(0, 1, factor, endpoint=False):
                interpolated_history.append(np.column_stack((f_y(t), f_x(t))).flatten())
        
        # Add the last frame without interpolation
        interpolated_history.append(history[-1])
        return interpolated_history

    # Interpolate the history with a factor to determine smoothness
    interpolated_history = interpolate_history(history, factor=5)

    def update(frame):
        # Update scatter objects for the current frame
        scatters[frame].set_offsets(np.column_stack([history[frame][1::2] + 0.5, history[frame][::2] + 0.5]))
        
        # Update line objects to connect the progression of each placement through time
        for i in range(num_placements):
            # Check if the line already has data, if not, initialize with the first point
            if len(lines[i].get_xdata()) == 0:
                lines[i].set_data([history[0][i*2+1] + 0.5], [history[0][i*2] + 0.5])
            
            # Append new point data to the line
            x_data = np.append(lines[i].get_xdata(), history[frame][i*2+1] + 0.5)
            y_data = np.append(lines[i].get_ydata(), history[frame][i*2] + 0.5)
            lines[i].set_data(x_data, y_data)
        
        return scatters + lines

    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(interpolated_history), repeat=True, blit=True, interval=200)
    
    # Display the animation
    plt.tight_layout()
    plt.show()


def animate_plot_values_and_progression_3d(values, history):
    # Create a figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the size of the values matrix
    m, n = values.shape
    
    # Create a meshgrid for the values matrix
    x = np.arange(n)
    y = np.arange(m)
    X, Y = np.meshgrid(x, y)
    
    # Plot the surface with shading
    surf = ax.plot_surface(X, Y, values, cmap='viridis', edgecolor='none', alpha=0.8, shade=True)
    
    # Define a gradient color range for the progression
    num_steps = len(history)
    colors = sns.color_palette("coolwarm", num_steps)
    
    # Initialize scatter and line objects
    scatters = [ax.scatter([], [], [], color=colors[i], s=200, depthshade=True, edgecolors="black", linewidth=1.5) for i in range(num_steps)]
    num_placements = len(history[0]) // 2
    lines = [ax.plot([], [], [], color=colors[i], linewidth=2.5, alpha=0.7)[0] for i in range(num_placements)]
    
    def update(frame):
        # Extracting the coordinates and values for each placement
        x_coords = history[frame][1::2]
        y_coords = history[frame][::2]
        z_coords = [values[int(y), int(x)] + 0.05 * np.max(values) for x, y in zip(x_coords, y_coords)]  # Offset
        
        # Update scatter objects for the current frame
        scatters[frame]._offsets3d = (x_coords, y_coords, z_coords)
        
        # Update line objects to connect the progression of each placement through time
        if frame > 0:
            for i in range(num_placements):
                prev_x, prev_y = history[frame-1][i*2+1], history[frame-1][i*2]
                prev_z = values[int(prev_y), int(prev_x)] + 0.05 * np.max(values)
                x, y, z = x_coords[i], y_coords[i], z_coords[i]
                lines[i].set_data_3d([prev_x, x], [prev_y, y], [prev_z, z])
        
        return scatters + lines
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=num_steps, repeat=True, blit=False, interval=200)
    
    # Set labels and title
    ax.set_xlabel('X', labelpad=15)
    ax.set_ylabel('Y', labelpad=15)
    ax.set_zlabel('Value', labelpad=15)
    ax.set_title("3D Placement Progression", fontdict={"fontsize": 24, "fontweight": "bold"})
    
    # Adjust the view angle for better visualization
    ax.view_init(elev=25, azim=-60)
    
    # Add grid lines
    ax.zaxis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Add a colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=8)
    cbar.ax.set_title('Value', fontdict={"fontsize": 14, "fontweight": "bold"})
    
    # Show the plot
    plt.tight_layout()
    plt.show()


print("Started Plotting")
plot_values_and_progression(matrix, history_array)
# plot_values_and_progression_3d(matrix, history_array)
# animate_values_and_progression(matrix, history_array)
# animate_plot_values_and_progression_3d(matrix, history_array)