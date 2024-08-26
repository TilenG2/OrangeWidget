import random
import numpy as np
from copy import copy

def generate_world_bounds(features, feature_bounds = (0, 1), endT = .20, endA = .40, max_depth = 100, depth = 0, current_bounds = None):
    if type(features) is int:
        features = np.arange(features)
    if type(feature_bounds) is tuple and len(feature_bounds) == 2:
        feature_bounds = [feature_bounds for _ in range(len(features))]
    if len(features) != len(feature_bounds):
        print(f"features {features} and feature_bounds {feature_bounds} not the same size")
        return
    if current_bounds is None:
        current_bounds = copy(feature_bounds)
    
    args = {"features": features,
            "feature_bounds": feature_bounds,
            "endT": endT,
            "endA": endA,
            "max_depth": max_depth,
            }
    bounds = set()

    feature = random.randint(0, len(features) - 1)
    start, end = current_bounds[feature]
    f_start, f_end = feature_bounds[feature]
    random_offset = (f_end - f_start) * endT
    start_random, end_random = start + random_offset, end - random_offset
    if end_random <= start_random or \
            depth > max_depth or \
            np.multiply.reduce([b - a for a, b in feature_bounds]) * endA > np.multiply.reduce([b - a for a, b in current_bounds]):
        return {tuple(current_bounds)}
 
    cut = random.uniform(start_random, end_random)
    
    current_bounds[feature] = (start, cut)
    bounds |= generate_world_bounds(current_bounds = copy(current_bounds), depth = depth + 1, **args)
    
    current_bounds[feature] = (cut, end)
    bounds |= generate_world_bounds(current_bounds = copy(current_bounds), depth = depth + 1, **args)
    return bounds

def generate_true_values(bounds_with_class, feature_bounds):
    points = []
    for _ in range(10**4):
        val = [random.uniform(feature_bounds[i][0], feature_bounds[i][1]) for i in range(len(feature_bounds))]
        for bounds, cls in bounds_with_class.items():
            if all([start < val[i] < end for i, (start, end) in enumerate(bounds)]):
                points.append(np.array(val + [cls]))
    return np.array(points)

def draw_true_world(data, cmap='viridis'):
    import matplotlib.pyplot as plt
    
    x = data[:, 0]
    y = data[:, 1]
    classes = data[:, 2]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=classes, cmap=cmap, edgecolor='k')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Points by Class')
    plt.show()

def draw_observed_world(data, cmap='viridis'):
    import matplotlib.pyplot as plt
    
    x = data[:, 3]
    y = data[:, 5]
    classes = data[:, 2]

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=classes, cmap=cmap, edgecolor='k')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Points by Class')
    plt.show()

def draw_world_grid(world_coords, feature_bounds):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(111)

    plt.title("Rectangles")
    for (_x1, _x2), (_y1, _y2) in world_coords:
        rect1 = Rectangle((_x1, _y1), abs(_x2 - _x1), abs(_y2 - _y1), color = 'orange', fc = 'none', lw = 2)
        ax.add_patch(rect1)

    plt.xlim(feature_bounds[0])
    plt.ylim(feature_bounds[1])

    plt.show()
    
corr_dict = {
    0.57: -0.25,
    0.60: -0.1,
    0.65: 0, 
    0.70: 0.065,
    0.75: 0.135,
    0.80: 0.215,
    0.85: 0.305,
    0.90: 0.41,
    0.95: 0.56,
    1: 1, 
}
def add_unc(tv, errRange = 1, corr = 1):
    err = errRange * random.random() * [-1,+1][random.randint(0,1)]
    ov = tv + err * random.uniform(corr_dict[corr] , 1)
    unc = abs(err)
    return ov, unc

def data_to_world(data, errRange = 1, corr = 1):
    _, n_feat = data.shape
    n_feat -= 1
    args = {
        "errRange": errRange,
        "corr" : corr
    }
    arr = []
    i = 0
    for _ in range(n_feat):
        arr.append(data[:, i])
        i += 1
    arr.append(data[:, i])

    for i in range(n_feat):
        ovunc = np.array([add_unc(tv, **args) for tv in data[:, i]])
        arr.append(ovunc[:, 0]) 
        arr.append(ovunc[:, 1])

    #True Value 1, True Value 2, Class, Observed Value 1, Uncertainty 1, Observed Value 2, Uncertainty 2
    return np.stack(arr, axis=1)

def data_to_world(data, errRange = 1, corr = 1):
    _, n_feat = data.shape
    n_feat -= 1
    args = {
        "errRange": errRange,
        "corr" : corr
    }
    arr = []
    i = 0
    for _ in range(n_feat):
        arr.append(data[:, i])
        i += 1
    arr.append(data[:, i])

    for i in range(n_feat):
        ovunc = np.array([add_unc(tv, **args) for tv in data[:, i]])
        arr.append(ovunc[:, 0]) 
        arr.append(ovunc[:, 1])

    #True Value 1, True Value 2, Class, Observed Value 1, Uncertainty 1, Observed Value 2, Uncertainty 2
    return np.stack(arr, axis=1)