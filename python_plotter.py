import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Load matrix from matrix.csv
matrix = np.loadtxt("matrix.csv", delimiter=',')

# Load bp from bp.csv
bp = np.loadtxt("bp.csv", delimiter=',')


history_array = np.loadtxt("history.csv", delimiter=",")


def plot_values_and_progression(values, history):
    # Set a minimalist theme
    sns.set_theme(style="white")
    
    # Get the size of the values matrix
    m, n = values.shape
    
    # Define a gradient color range for the progression
    num_steps = len(history)
    colors = sns.color_palette("coolwarm", num_steps)
    
    # Simplified heatmap with a subtle color palette
    plt.imshow(values, cmap="viridis", aspect="auto")
    plt.title("Placement Progression", fontdict={"fontsize": 14, "fontweight": "bold"})
    plt.xlim(0.5, n - 0.5)
    plt.ylim(m - 0.5, 0.5)
    
    for idx, placements in enumerate(history):
        # No labels for a cleaner look
        plt.scatter(placements[1::2] + 0.5, placements[::2] + 0.5, color=colors[idx], s=50, label=None)
        
        # Connect the placements with smooth lines
        if idx > 0:
            prev_placements = history[idx-1]
            for pr, pc, r, c in zip(prev_placements[::2], prev_placements[1::2], placements[::2], placements[1::2]):
                plt.plot([pc+0.5, c+0.5], [pr+0.5, r+0.5], color=colors[idx], linewidth=1.5)
    
    # Remove x and y ticks for a cleaner look
    plt.xticks([])
    plt.yticks([])
    
    # Save the plot to a file
    plt.savefig("placement_progression_plot_python.png", dpi=300, bbox_inches="tight")
    plt.show()




plot_values_and_progression(matrix, history_array)