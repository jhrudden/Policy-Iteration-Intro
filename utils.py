import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List, Tuple, Dict, Optional

def plot_cardinal_value_function(ax, V: np.ndarray, title: Optional[str] = None, save_path: Optional[str] = None):
    """
    plots the value function as a heatmap for a cardinal gridworld

    Args:
    ax: the axes to plot on
    V: the value function
    title: the title of the plot
    save_path: the path to save the plot

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

    if save_path:
        plt.savefig(save_path, format='png')

def plot_cardinal_policy(ax, policy: Dict[Tuple[int, int], List[int]], V_shape: Tuple[int,int], title: Optional[str] = None):
    """
    plots the policy as arrows on a heatmap of the value function for a cardinal gridworld

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

def plot_cardinal_value_and_policy(V: np.ndarray, policy: Dict[Tuple[int, int], List[int]], title: str = 'Value Function and Policy', save_path: Optional[str] = None):
    """
    plots the value function as a heatmap and the policy as arrows on the heatmap for a cardinal gridworld

    Args:
    V: the value function
    policy: the policy
    title: the title of the plot
    save_path: the path to save the plot

    """
    n, m = V.shape
    grid_aspect_ratio = m / n # Swap the ratio since x corresponds to rows and y to cols
    fig_height = 10  # You can adjust this height as necessary
    fig_width = fig_height * grid_aspect_ratio
    fig, ax = plt.subplots(figsize=(2*fig_width, fig_height), nrows=1, ncols=2)
    axes = ax.flatten()
    plot_cardinal_value_function(axes[0], V)
    plot_cardinal_policy(axes[1], policy, V.shape)

    plt.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])  # Adjust the bottom parameter as needed to accommodate the text
    
    axes[0].text(0.5, -0.025, '$v_*$', transform=axes[0].transAxes, ha='center', va='top', fontsize=25)
    axes[1].text(0.5, -0.025, '$\pi_*$', transform=axes[1].transAxes, ha='center', va='top', fontsize=25)

    if save_path:
        plt.savefig(save_path, format='png')

    plt.show()
    
def plot_car_rental_policy_map(ax, policy, title=None):
    """
    Plots a policy map for the car rental problem on the given Axes object.

    :param ax: Matplotlib Axes object to plot on.
    :param policy: Dictionary mapping states (tuple of A_cars, B_cars) to a list of (action, probability) tuples.
    :param title: Title for the plot.
    """
    X, Y = np.meshgrid(range(21), range(21))
    Z = np.zeros_like(X, dtype=float)

    # Evaluate the policy for each state to determine the Z-axis (actions).
    for i in range(21):
        for j in range(21):
            # Get the action with the highest probability for the current state.
            # If multiple actions have the same highest probability, choose the first one.
            actions, probabilities = zip(*policy[(i, j)])
            action = actions[np.argmax(probabilities)]
            Z[i, j] = action

    # Create a contour plot on the provided Axes object.
    contour = ax.contourf(X, Y, Z, levels=np.arange(Z.min(), Z.max()+1), cmap='viridis')
    ax.set_title(title)
    ax.set_ylabel('Number of cars at first location')
    ax.set_xlabel('Number of cars at second location')

    # Add a colorbar to the contour plot.
    plt.colorbar(contour, ax=ax)

def plot_car_rental_value_function(ax, value_function, title):
    """
    Plots the value function on the given Axes object as a 3D surface plot for the car rental problem.

    :param ax: Matplotlib 3D Axes object to plot on.
    :param title: Title for the plot.
    :param value_function: 2D array representing the value function, where each element corresponds to the state value.
    """
    # Assuming the maximum number of cars at each location is 20 for the state space.
    # Create a grid for the state space.
    X, Y = np.meshgrid(range(21), range(21))

    # Plot the surface.
    surface = ax.plot_surface(X, Y, value_function, cmap='viridis')

    ax.set_title(title)
    ax.set_ylabel('#Cars at first location')
    ax.set_xlabel('#Cars at second location')
    ax.set_zlabel('Value')

    # Add a colorbar to the surface plot, adjusting its position.
    fig = plt.gcf()
    cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Value')

def plot_car_rental_value_and_policy(value_function: np.ndarray, policy: Dict[Tuple[int, int], List[int]], title: str = 'Car Rental Value Function and Policy', save_path: Optional[str] = None, policy_title: Optional[str] = None, value_function_title: Optional[str] = None):
    """
    Plots the value function as a 3D surface plot and the policy as a 2D contour plot for the car rental problem.

    Args:
    value_function: the value function
    policy: the policy
    title: the title of the plot
    save_path: the path to save the plot
    policy_title: the title of the policy plot
    value_function_title: the title of the value function plot
    """
    # Create a new figure with a 3D subplot for the value function and a 2D subplot for the policy map.
    fig = plt.figure(figsize=(24, 10))  # Adjust figure size as needed
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    value_function_title = value_function_title or 'Car Rental Value Function'
    # Plot the value function on the first subplot.
    plot_car_rental_value_function(ax1, value_function, title=value_function_title)

    policy_title = policy_title or 'Car Rental Policy'
    # Plot the policy map on the second subplot.
    plot_car_rental_policy_map(ax2, policy, title=policy_title)

    plt.suptitle(title, fontsize=25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout parameters as needed

    if save_path:
        plt.savefig(save_path, format='png')

    plt.show()