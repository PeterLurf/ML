import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
from matplotlib.animation import FuncAnimation

def generate_data(mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0, rho=0.0):
    """
    Compute the 2D Gaussian probability density on a grid.
    """
    mu = np.array([mu_x, mu_y])
    cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                    [rho * sigma_x * sigma_y, sigma_y**2]])
    
    # Create a grid of points
    x = np.linspace(mu_x - 4 * sigma_x, mu_x + 4 * sigma_x, 100)
    y = np.linspace(mu_y - 4 * sigma_y, mu_y + 4 * sigma_y, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Compute the Gaussian PDF
    inv_cov = np.linalg.inv(cov)
    diff = pos - mu
    Z = np.exp(-0.5 * np.einsum('...i,ij,...j', diff, inv_cov, diff))
    Z /= 2 * np.pi * np.sqrt(np.linalg.det(cov))
    return X, Y, Z

def animate(frame, ax):
    """
    Update the view angle for animation.
    """
    ax.view_init(elev=30, azim=frame)
    return ax,

def main():
    # Generate Gaussian data with default parameters.
    X, Y, Z = generate_data(mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0, rho=0.0)
    
    # Set up the figure and 3D axes.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Gaussian surface.
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('2D Gaussian Distribution (3D Animated)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    
    # Create an animation that rotates the view.
    anim = FuncAnimation(fig, animate, frames=np.arange(0, 360, 1), fargs=(ax,), interval=50, blit=False)
    
    # Display the plot.
    plt.show()

if __name__ == "__main__":
    main()
