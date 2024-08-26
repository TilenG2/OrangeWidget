from utils import *

import matplotlib.pyplot as plt
feature_bounds = [(0, 1), (0, 1), (0, 1)]
world_coords = generate_world_bounds(features = 3, feature_bounds = feature_bounds, endT = .2, endA = .1,  max_depth = 3)
bounds_with_class = {}
# seed(None)
for i, bounds in enumerate(world_coords):
    bounds_with_class[bounds] = random.randint(0, 1)

data = generate_true_values(bounds_with_class, feature_bounds)

#True Value 1, True Value 2, Class, Observed Value 1, Observed Value 2, Uncertainty 1, Uncertainty 2
world = data_to_world(data, errRange = 0.1, corr = 1)

# Example array of points [x, y, z, class]
data_3d = world

# Extracting x, y, z, and class
x_3d = data_3d[:, 0]
y_3d = data_3d[:, 1]
z_3d = data_3d[:, 2]
classes_3d = data_3d[:, 3]

# Plotting the points in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_3d, y_3d, z_3d, c=classes_3d, cmap='viridis', edgecolor='k')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()

# Extracting x, y, z, and class
x_3d = data_3d[:, 4]
y_3d = data_3d[:, 6]
z_3d = data_3d[:, 8]
classes_3d = data_3d[:, 3]

# Plotting the points in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_3d, y_3d, z_3d, c=classes_3d, cmap='viridis', edgecolor='k')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()