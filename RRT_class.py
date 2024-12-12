import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import time

rings = {
    'ring1': {'center': np.array([2.5, 1.4, 1.15]), 'radius': 0.4, 'normal': np.array([0, -1, 0])},
    'ring2': {'center': np.array([1.3, -0.5, 1.55]), 'radius': 0.4, 'normal': np.array([-1, 0, 0])}
}

# rings = {
#     'ring1': {'center': np.array([2.5, 1.4, 1.15]), 'radius': 0.4, 'normal': np.array([0, -1, 0])},
#     'ring2': {'center': np.array([0.5, -2.0, 1.55]), 'radius': 0.4, 'normal': np.array([-1, 0, 0])},
#     'ring3': {'center': np.array([1.0, 3.0, 0.8]), 'radius': 0.5, 'normal': np.array([1, 0, 1])},
#     'ring4': {'center': np.array([4.0, 1.5, 1.2]), 'radius': 0.35, 'normal': np.array([1, 1, 0])},
#     'ring5': {'center': np.array([3.0, -1.5, 0.5]), 'radius': 0.45, 'normal': np.array([0, 1, 1])},
#     'ring6': {'center': np.array([1.5, 0.5, 2.0]), 'radius': 0.5, 'normal': np.array([0, 1, 1])},
#     'ring7': {'center': np.array([4.5, -1.5, 1.0]), 'radius': 0.4, 'normal': np.array([1, -1, 1])},
#     'ring8': {'center': np.array([0.5, 4.0, 1.4]), 'radius': 0.3, 'normal': np.array([-1, 1, -0.5])},
#     'ring9': {'center': np.array([3.5, 2.0, 0.6]), 'radius': 0.4, 'normal': np.array([0.3, 1, 0.8])},
#     'ring10': {'center': np.array([2.0, -3.0, 1.8]), 'radius': 0.5, 'normal': np.array([1, -0.2, -0.3])}
# }

start_point = np.array([0.0, 0.0, 0.5])
end_point = np.array([0.0, 0.0, 0.5])
safety_margin = 0.2 
min_rand_area = [-1, -1, 0]
max_rand_area = [4, 4, 2]
step_size = 0.5 
max_iter = 1000 

def is_in_obstacle(point, rings, safety_margin):
    point = np.array(point)  
    for ring_name, ring in rings.items():
        center = ring['center']  
        radius = ring['radius']  
        normal = ring['normal']  
        normal = normal / np.linalg.norm(normal)  
        vector_to_point = point - center
        distance_to_plane = np.dot(vector_to_point, normal)
        if abs(distance_to_plane) > safety_margin:
            continue
        projected_point = point - distance_to_plane * normal
        distance_to_center = np.linalg.norm(projected_point - center)
        if distance_to_center <= radius + safety_margin:
            return True
    return False

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def can_connect_directly(point1, point2, rings, safety_margin, steps=10):
    for t in np.linspace(0, 1, steps):
        intermediate_point = np.array(point1) + t * (np.array(point2) - np.array(point1))
        if is_in_obstacle(intermediate_point, rings, safety_margin):
            return False
    return True

def rrt(start, goal, rings, safety_margin, step_size, max_iter=1000):

    class Node:
        def __init__(self, position):
            self.position = position
            self.parent = None  # 父节点

    def sample_random_point():
        return np.random.uniform(min_rand_area, max_rand_area) 

    def is_valid_point(point, rings, safety_margin):
        return not is_in_obstacle(point, rings, safety_margin)

    def closest_node(tree, point):
        return min(tree, key=lambda node: distance(node.position, point))

    def steer(from_node, to_point, step_size):
        direction = np.array(to_point) - np.array(from_node.position)
        length = np.linalg.norm(direction)
        direction = direction / length  
        new_position = np.array(from_node.position) + direction * min(step_size, length)
        new_node = Node(new_position)
        new_node.parent = from_node
        return new_node

    start_node = Node(start)
    goal_node = Node(goal)
    tree = [start_node]
    node_positions = [start]  

    for _ in range(max_iter):
        random_point = sample_random_point()
        nearest_node = closest_node(tree, random_point)
        new_node = steer(nearest_node, random_point, step_size)
        if is_valid_point(new_node.position, rings, safety_margin) and \
           can_connect_directly(nearest_node.position, new_node.position, rings, safety_margin):
            tree.append(new_node)
            node_positions.append(new_node.position)
            if distance(new_node.position, goal) <= step_size and \
               can_connect_directly(new_node.position, goal, rings, safety_margin):
                goal_node.parent = new_node
                break

    if goal_node.parent is None:
        print("Failed to find a path using RRT.")
        return [], node_positions

    path = []
    current = goal_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    path.reverse()
    return path, node_positions

def optimize_path(path, rings, safety_margin):
    if len(path) <= 2:
        return path
    optimized_path = [path[0]]
    current_point = path[0]
    for i in range(1, len(path)):
        if not can_connect_directly(current_point, path[i], rings, safety_margin):
            optimized_path.append(path[i - 1])
            current_point = path[i - 1]
    optimized_path.append(path[-1])
    return optimized_path

def save_positions_to_csv(node_positions, csv_filename):
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["X", "Y", "Z"])
        for position in node_positions:
            writer.writerow(position)
    print(f"Saved node positions to {csv_filename}")

def plot_rings(ax, rings):
    for ring_name, ring in rings.items():
        center = ring['center']
        radius = ring['radius']
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.array([np.cos(theta) * radius, np.sin(theta) * radius, np.zeros_like(theta)])
        z_axis = np.array([0, 0, 1])
        normal = ring['normal']
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(np.dot(z_axis, normal) / (np.linalg.norm(z_axis) * np.linalg.norm(normal)))
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
        else:
            rotation_matrix = np.eye(3)
        rotated_circle = rotation_matrix @ circle
        ax.plot(center[0] + rotated_circle[0, :], center[1] + rotated_circle[1, :], center[2] + rotated_circle[2, :], color='black', label=f'{ring_name}')

def set_axes_equal_and_uniform(ax, tick_interval):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    min_limit = min(x_limits[0], y_limits[0], z_limits[0])
    max_limit = max(x_limits[1], y_limits[1], z_limits[1])
    ax.set_xlim3d([min_limit, max_limit])
    ax.set_ylim3d([min_limit, max_limit])
    ax.set_zlim3d([min_limit, max_limit])
    ticks = np.arange(min_limit, max_limit + tick_interval, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_box_aspect([1, 1, 1])

def main():

    start_time = time.time()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_rings(ax, rings)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(0, 2)
    current_point = start_point
    full_path = []
    all_node_positions = []

    for i, ring_name in enumerate(rings):
        ring = rings[ring_name]
        entry_point = ring['center'] - ring['normal'] * (safety_margin + 0.05)
        exit_point = ring['center'] + ring['normal'] * (safety_margin + 0.05)
        path_to_entry, node_positions_entry = rrt(current_point, entry_point, rings, safety_margin, step_size)
        all_node_positions.extend(node_positions_entry)
        if len(path_to_entry) > 1:
            full_path.extend(path_to_entry)
            ax.plot(*np.array(path_to_entry).T, color='green', label=f'Path to {ring_name} entry')
        full_path.append(exit_point)
        ax.plot([entry_point[0], exit_point[0]], [entry_point[1], exit_point[1]], [entry_point[2], exit_point[2]], color='orange', label=f'{ring_name} entry to exit')
        current_point = exit_point
    path_to_goal, node_positions_goal = rrt(current_point, end_point, rings, safety_margin, step_size)
    all_node_positions.extend(node_positions_goal)

    if len(path_to_goal) > 1:
        full_path.extend(path_to_goal)
        ax.plot(*np.array(path_to_goal).T, color='green', label='Path to goal')

    optimized_path = optimize_path(full_path, rings, safety_margin)
    
    end_time = time.time()

    # 计算耗时
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    #----------------------------------------------------------


    save_positions_to_csv(optimized_path, "optimized_path.csv")
    ax.plot(*np.array(optimized_path).T, color='red', label='Optimized Path')
    ax.scatter(*start_point, color='blue', s=50, label='Start Point')
    ax.scatter(*end_point, color='red', s=50, label='End Point')
    set_axes_equal_and_uniform(ax, tick_interval=0.5)
    # plt.legend()
    plt.show()

main()
