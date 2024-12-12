import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import heapq
import time

rings = {
    'ring1': {'center': np.array([2.5, 1.4, 1.15]), 'radius': 0.4, 'normal': np.array([0, -1, 0])},
    'ring2': {'center': np.array([1.3, -0.5, 1.55]), 'radius': 0.4, 'normal': np.array([-1, 0, 0])},
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

step_size = 0.1  

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
    return np.linalg.norm(p1 - p2)

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None

import heapq

def a_star(start, goal, rings, safety_margin, step_size):


    def heuristic(point, goal):

        return np.linalg.norm(np.array(point) - np.array(goal))

    def is_valid_point(point, rings, safety_margin):

        return not is_in_obstacle(point, rings, safety_margin)

    def can_connect_directly(point1, point2, rings, safety_margin, steps=10):

        for t in np.linspace(0, 1, steps):
            intermediate_point = point1 + t * (np.array(point2) - np.array(point1))
            if is_in_obstacle(intermediate_point, rings, safety_margin):
                return False
        return True

    # 初始化开放列表和关闭列表
    open_set = []
    closed_set = set()
    came_from = {}  # 记录每个节点的父节点

    # 节点代价：g(实际代价)和f(总代价)
    g_score = {tuple(start): 0}  # 起点的实际代价为0
    f_score = {tuple(start): heuristic(start, goal)}  # 起点的总代价为启发值

    # 将起点加入开放列表
    heapq.heappush(open_set, (f_score[tuple(start)], tuple(start)))

    # 保存节点坐标
    node_positions = [start]

    while open_set:
        # 从开放列表中取出 f 值最低的节点
        _, current = heapq.heappop(open_set)

        # 如果可以直接连线到目标点，结束搜索
        if can_connect_directly(current, goal, rings, safety_margin):
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(start)
            path.reverse()
            path.append(goal)  # 直接连线到目标点
            node_positions.append(goal)
            print(f"Found a path to the goal with direct connection: {goal}")
            return path, node_positions

        # 将当前节点加入关闭列表
        closed_set.add(current)

        # 扩展当前节点的邻居
        for dx in np.linspace(-step_size, step_size, 3):
            for dy in np.linspace(-step_size, step_size, 3):
                for dz in np.linspace(-step_size, step_size, 3):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                    # 跳过无效点（障碍物或已访问）
                    if not is_valid_point(neighbor, rings, safety_margin) or tuple(neighbor) in closed_set:
                        continue

                    # 计算邻居的 g 值
                    tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor) - np.array(current))

                    # 如果发现更短路径，更新代价和路径
                    if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = current
                        g_score[tuple(neighbor)] = tentative_g_score
                        f_score[tuple(neighbor)] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[tuple(neighbor)], tuple(neighbor)))
                        node_positions.append(list(neighbor))

    # 如果未找到路径
    print(f"Failed to find a path to the goal: {goal}")
    return [], node_positions


def generate_ring_entry_exit(ring, margin):
    center = ring['center']
    normal = ring['normal']

    entry_point = center - normal * (margin + 0.05)

    exit_point = center + normal * (margin + 0.05)

    print(f"Generated entry point: {entry_point}, exit point: {exit_point}")
    return entry_point, exit_point


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


def can_connect_directly(point1, point2, rings, safety_margin, steps=10):

        for t in np.linspace(0, 1, steps):
            intermediate_point = point1 + t * (np.array(point2) - np.array(point1))
            if is_in_obstacle(intermediate_point, rings, safety_margin):
                return False
        return True

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

def set_axes_equal_and_uniform(ax, tick_interval):

    # 获取各个轴的范围
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # 找到所有轴的最小值和最大值
    min_limit = min(x_limits[0], y_limits[0], z_limits[0])
    max_limit = max(x_limits[1], y_limits[1], z_limits[1])

    # 设置相同的范围
    ax.set_xlim3d([min_limit, max_limit])
    ax.set_ylim3d([min_limit, max_limit])
    ax.set_zlim3d([min_limit, max_limit])

    # 设置刻度间隔一致
    ticks = np.arange(min_limit, max_limit + tick_interval, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # 设置为等比例显示
    ax.set_box_aspect([1, 1, 1])  # 等比例显示

def main():
    start_time = time.time()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制圆盘
    plot_rings(ax, rings)

    # ax.set_xlim(-1, 4)
    # ax.set_ylim(-1, 4)
    # ax.set_zlim(0, 2)

    current_point = start_point
    full_path = []
    all_node_positions = [] 
    entry_exit_points = []  

    for i, ring_name in enumerate(rings):
        ring = rings[ring_name]

        entry_point, exit_point = generate_ring_entry_exit(ring, safety_margin)

        if is_in_obstacle(entry_point, rings, safety_margin):
            print(f"Entry point {entry_point} is in obstacle!")
        if is_in_obstacle(exit_point, rings, safety_margin):
            print(f"Exit point {exit_point} is in obstacle!")

        entry_exit_points.append(entry_point)
        entry_exit_points.append(exit_point)

        path_to_entry, node_positions_entry = a_star(current_point, entry_point, rings, safety_margin, step_size)

        all_node_positions.extend(node_positions_entry)

        if len(path_to_entry) > 1:
            full_path.extend(path_to_entry)
            ax.plot(*np.array(path_to_entry).T, color='green', label=f'Path to {ring_name} entry')

        full_path.append(exit_point)
        ax.plot([entry_point[0], exit_point[0]], [entry_point[1], exit_point[1]], [entry_point[2], exit_point[2]], color='green', label=f'{ring_name} entry to exit')

        current_point = exit_point

    path_to_goal, node_positions_goal = a_star(current_point, end_point, rings, safety_margin, step_size)

    all_node_positions.extend(node_positions_goal)

    if len(path_to_goal) > 1:
        full_path.extend(path_to_goal)
        ax.plot(*np.array(path_to_goal).T, color='green', label='Path to goal')

    optimized_path = optimize_path(full_path, rings, safety_margin)

    end_time = time.time()

    # 计算并打印耗时
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")

    #----------------------------------------------------------


    # 保存关键节点到 CSV
    offset_and_save_positions_to_csv(optimized_path, -start_point, "key_nodes.csv")

    # 保存所有节点坐标到 CSV 文件
    save_positions_to_csv(full_path, "all_a_star_nodes.csv")

    ax.plot(*np.array(optimized_path).T, color='red', label='Optimized Path')

    # 标记起点和终点
    ax.scatter(*start_point, color='blue', s=50, label='Start Point')  # 起点
    ax.text(start_point[0], start_point[1], start_point[2], 'Start', color='blue')

    ax.scatter(*end_point, color='red', s=50, label='End Point')  # 终点
    ax.text(end_point[0], end_point[1], end_point[2], 'End', color='red')

    # 设置 x, y, z 轴等长，且刻度间隔一致为 0.1
    set_axes_equal_and_uniform(ax, tick_interval=0.5)

    # 显示绘制的图形
    # plt.legend()
    plt.show()

# 保存节点坐标到 CSV 文件
def save_positions_to_csv(node_positions, csv_filename):
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(["X", "Y", "Z"])
        # 写入节点坐标
        for position in node_positions:
            writer.writerow(position)
    print(f"Saved node positions to {csv_filename}")

def offset_and_save_positions_to_csv(node_positions, offset, csv_filename):

    # 偏移后的坐标
    offset_positions = [np.array(position) + np.array(offset) for position in node_positions]
    
    # 保存到 CSV 文件
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(["X", "Y", "Z"])
        # 写入偏移后的节点坐标
        for position in offset_positions:
            writer.writerow(position)
    
    print(f"Saved offset node positions to {csv_filename}")

main()


