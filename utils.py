import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from typing import List, Tuple, Dict, Optional

def plot_value_function(ax, V: np.ndarray, title: Optional[str] = None):
    """
    plots the value function as a heatmap

    Args:
    ax: the axes to plot on
    V: the value function
    title: the title of the plot

    """
    im = ax.imshow(V, cmap='coolwarm', norm=Normalize(vmin=V.min(), vmax=V.max()), alpha=0.8)

    # Loop over data dimensions and create text annotations.
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            text = ax.text(j, i, f'{V[i, j]:.1f}',
                           ha="center", va="center", color="black")

    if title:
        ax.set_title(title)
    ax.axis('off')

def plot_policy(ax, policy: Dict[Tuple[int, int], List[int]], V_shape: Tuple[int,int], title: Optional[str] = None):
    """
    plots the policy as arrows on a heatmap of the value function

    Args:
    ax: the axes to plot on
    policy: the policy
    V: the value function
    title: the title of the plot

    """
    # Normalize the arrow length
    arrow_length = 0.3

    # Define a mapping from actions to vector components (dx, dy)
    action_vectors = {
        0: (-arrow_length, 0),  # left
        1: (0, -arrow_length),  # down (since y increases downwards in plot coordinates)
        2: (arrow_length, 0),   # right
        3: (0, arrow_length),   # up (since y increases downwards in plot coordinates)
    }
    
    # Plot the arrows for each action in the policy
    for state, action_probs in policy.items():
        x, y = state  # Swap x and y to match row and column ordering
        actions, _ = zip(*action_probs)
        for action in actions:  # Unpack the actions from the numpy array
            dx, dy = action_vectors[action]
            ax.arrow(y, x, dy, dx, head_width=0.1, head_length=0.1, fc='k', ec='k')  # Swap dx and dy to match the plot

    # Set aspect of the plot to be equal and scale the plot area to fit the axes
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, V_shape[1]-0.5)
    ax.set_ylim(-0.5, V_shape[0]-0.5)

    # Set grid
    ax.set_xticks(np.arange(-0.5, V_shape[1], 1))
    ax.set_yticks(np.arange(-0.5, V_shape[0], 1))
    ax.grid(True, linestyle='-', color='black', linewidth=2)

    # Remove labels and tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    
    if title:
        ax.set_title(title)
    ax.invert_yaxis()  # Invert y-axis to match the coordinate system

def plot_value_and_policy(V: np.ndarray, policy: Dict[Tuple[int, int], List[int]], title: str = 'Value Function and Policy'):
    """
    plots the value function as a heatmap and the policy as arrows on the heatmap

    Args:
    V: the value function
    policy: the policy
    title: the title of the plot

    """
    n, m = V.shape
    grid_aspect_ratio = m / n # Swap the ratio since x corresponds to rows and y to cols
    fig_height = 10  # You can adjust this height as necessary
    fig_width = fig_height * grid_aspect_ratio
    fig, ax = plt.subplots(figsize=(2*fig_width, fig_height), nrows=1, ncols=2)
    axes = ax.flatten()
    plot_value_function(axes[0], V)
    plot_policy(axes[1], policy, V.shape)

    plt.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])  # Adjust the bottom parameter as needed to accommodate the text
    
    axes[0].text(0.5, -0.025, '$v_*$', transform=axes[0].transAxes, ha='center', va='top', fontsize=25)
    axes[1].text(0.5, -0.025, '$\pi_*$', transform=axes[1].transAxes, ha='center', va='top', fontsize=25)

    plt.show()
    
def plot_policy_map(ax, title, policy):
    """
    Plots a policy map on the given Axes object.

    :param ax: Matplotlib Axes object to plot on.
    :param title: Title for the plot.
    :param policy: Dictionary mapping states (tuple of A_cars, B_cars) to a list of (action, probability) tuples.
    """
    # Assuming the maximum number of cars at each location is 20 for the state space.
    # Create a grid for the state space.
    X, Y = np.meshgrid(range(21), range(21))
    Z = np.zeros_like(X)

    # Evaluate the policy for each state to determine the Z-axis (actions).
    for i in range(21):
        for j in range(21):
            # Get the action with the highest probability for the current state.
            # If multiple actions have the same highest probability, choose the first one.
            actions, probabilities = zip(*policy[(j, i)])
            action = actions[np.argmax(probabilities)]
            Z[i, j] = action

    # Create a contour plot on the provided Axes object.
    contour = ax.contourf(X, Y, Z, levels=range(-5, 6), cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('#Cars at first location')
    ax.set_ylabel('#Cars at second location')

    # Add a colorbar to the contour plot.
    plt.colorbar(contour, ax=ax)