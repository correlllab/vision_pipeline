import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create the data for the plot
prior_belief = np.linspace(0, 1, 50)
new_belief = np.linspace(0, 1, 50)
X, Y = np.meshgrid(prior_belief, new_belief)

# The updated belief is the product of the prior and the new belief
Z = X * Y

# Create the figure and a 3D subplot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with a color map
surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Set labels and title for clarity
ax.set_xlabel('Prior Belief p(prior)')
ax.set_ylabel('New Belief p(new)')
ax.set_zlabel('Updated Belief')
ax.set_title('Interactive Plot of Updated Belief')

# Add a color bar to map values to colors
fig.colorbar(surface, shrink=0.5, aspect=5)

# Show the plot. This will open a new interactive window.
plt.show()