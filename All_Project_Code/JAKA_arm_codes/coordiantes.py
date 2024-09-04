import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a new figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the coordinate frame
ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=1)
ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=1)
ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=1)

# Label the axes
ax.text(1, 0, 0, 'X', color='r')
ax.text(0, 1, 0, 'Y', color='g')
ax.text(0, 0, 1, 'Z', color='b')

# Set the limits of the plot
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Set the labels for each axis
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()
