import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import heapq

# Բ�̲���
rings = {
    'ring1': {'center': np.array([2.5, 1.4, 1.15]), 'radius': 0.4, 'normal': np.array([0, -1, 0])},
    'ring2': {'center': np.array([1.3, -0.5, 1.55]), 'radius': 0.4, 'normal': np.array([-1, 0, 0])}
}

# �����յ�
start_point = np.array([0.0, 0.0, 1.0])
end_point = np.array([0.0, 0.0, 1.0])
safety_margin = 0.2  # ������ȫ����

# RRT ����
max_iter = 3000  # ����������
step_size = 0.1  # ÿ������Ĳ���

def is_in_obstacle(point, rings, safety_margin):

    point = np.array(point)  # תΪ NumPy �����Է������

    for ring_name, ring in rings.items():
        # ��ȡԲ���Ĳ���
        center = ring['center']  # Բ������
        radius = ring['radius']  # Բ���뾶
        normal = ring['normal']  # Բ��������

        # ����㵽Բ��ƽ��ľ���
        normal = normal / np.linalg.norm(normal)  # �淶��������
        vector_to_point = point - center
        distance_to_plane = np.dot(vector_to_point, normal)

        # ����㵽ƽ��ľ��볬����չ��ȣ������ϰ�������
        if abs(distance_to_plane) > safety_margin:
            continue

        # ����ͶӰ��Բ��ƽ��
        projected_point = point - distance_to_plane * normal

        # ����ͶӰ�㵽Բ�ĵľ���
        distance_to_center = np.linalg.norm(projected_point - center)

        # ���ͶӰ����Բ����չ�뾶��Χ�ڣ�������ϰ�����
        if distance_to_center <= radius + safety_margin:
            return True

    # ����㲻���κ��ϰ�����
    return False

# ��������֮���ŷ����þ���
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# RRT �ڵ���
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

    # ��ʼ�������б�͹ر��б�
    open_set = []
    closed_set = set()
    came_from = {}  # ��¼ÿ���ڵ�ĸ��ڵ�

    # �ڵ���ۣ�g(ʵ�ʴ���)��f(�ܴ���)
    g_score = {tuple(start): 0}  # ����ʵ�ʴ���Ϊ0
    f_score = {tuple(start): heuristic(start, goal)}  # �����ܴ���Ϊ����ֵ

    # �������뿪���б�
    heapq.heappush(open_set, (f_score[tuple(start)], tuple(start)))

    # ����ڵ�����
    node_positions = [start]

    while open_set:
        # �ӿ����б���ȡ�� f ֵ��͵Ľڵ�
        _, current = heapq.heappop(open_set)

        # �������ֱ�����ߵ�Ŀ��㣬��������
        if can_connect_directly(current, goal, rings, safety_margin):
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(start)
            path.reverse()
            path.append(goal)  # ֱ�����ߵ�Ŀ���
            node_positions.append(goal)
            print(f"Found a path to the goal with direct connection: {goal}")
            return path, node_positions

        # ����ǰ�ڵ����ر��б�
        closed_set.add(current)

        # ��չ��ǰ�ڵ���ھ�
        for dx in np.linspace(-step_size, step_size, 3):
            for dy in np.linspace(-step_size, step_size, 3):
                for dz in np.linspace(-step_size, step_size, 3):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                    # ������Ч�㣨�ϰ�����ѷ��ʣ�
                    if not is_valid_point(neighbor, rings, safety_margin) or tuple(neighbor) in closed_set:
                        continue

                    # �����ھӵ� g ֵ
                    tentative_g_score = g_score[current] + np.linalg.norm(np.array(neighbor) - np.array(current))

                    # ������ָ���·�������´��ۺ�·��
                    if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                        came_from[tuple(neighbor)] = current
                        g_score[tuple(neighbor)] = tentative_g_score
                        f_score[tuple(neighbor)] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[tuple(neighbor)], tuple(neighbor)))
                        node_positions.append(list(neighbor))

    # ���δ�ҵ�·��
    print(f"Failed to find a path to the goal: {goal}")
    return [], node_positions

# ����Բ�̵���ںͳ��ڵ�
def generate_ring_entry_exit(ring, margin):
    center = ring['center']
    normal = ring['normal']

    # ��ڵ㣺�ط�����������ƽ�� margin
    entry_point = center - normal * (margin + 0.05)

    # ���ڵ㣺�ط�����������ƽ�� margin
    exit_point = center + normal * (margin + 0.05)

    print(f"Generated entry point: {entry_point}, exit point: {exit_point}")
    return entry_point, exit_point

# ����Բ��
def plot_rings(ax, rings):
    for ring_name, ring in rings.items():
        center = ring['center']
        radius = ring['radius']

        # ����Բ��
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.array([np.cos(theta) * radius, np.sin(theta) * radius, np.zeros_like(theta)])

        # ��ת��Բ�̵ķ���������
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
        ax.plot(center[0] + rotated_circle[0, :], center[1] + rotated_circle[1, :], center[2] + rotated_circle[2, :], label=f'{ring_name}')

def extract_key_nodes(full_path, entry_exit_points, angle_threshold=np.pi / 6):

    key_nodes = [full_path[0]]  # �����Ϊ��һ���ؼ��ڵ�

    # ����·������ⷽ��仯
    for i in range(1, len(full_path) - 1):
        # ��ǰ���ǰ��������
        v1 = np.array(full_path[i]) - np.array(full_path[i - 1])
        v2 = np.array(full_path[i + 1]) - np.array(full_path[i])
        
        # ����ǰ���������ļн�
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # ��ֹ�������
        
        # ����нǴ�����ֵ����ǰ���ǹؼ��ڵ�
        if theta > angle_threshold:
            key_nodes.append(full_path[i])

    # ���Բ������ںͳ��ڵ㣨������ڹؼ��ڵ��У�
    for point in entry_exit_points:
        if not any(np.allclose(point, key_node) for key_node in key_nodes):
            key_nodes.append(point)

    # ����յ�
    key_nodes.append(full_path[-1])

    return key_nodes

def set_axes_equal_and_uniform(ax, tick_interval):

    # ��ȡ������ķ�Χ
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # �ҵ����������Сֵ�����ֵ
    min_limit = min(x_limits[0], y_limits[0], z_limits[0])
    max_limit = max(x_limits[1], y_limits[1], z_limits[1])

    # ������ͬ�ķ�Χ
    ax.set_xlim3d([min_limit, max_limit])
    ax.set_ylim3d([min_limit, max_limit])
    ax.set_zlim3d([min_limit, max_limit])

    # ���ÿ̶ȼ��һ��
    ticks = np.arange(min_limit, max_limit + tick_interval, tick_interval)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # ����Ϊ�ȱ�����ʾ
    ax.set_box_aspect([1, 1, 1])  # �ȱ�����ʾ

def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ����Բ��
    plot_rings(ax, rings)

    # ���������᷶Χ
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(0, 2)

    # ����·��
    current_point = start_point
    full_path = []
    all_node_positions = []  # ���ڱ�������·���εĽڵ�����
    entry_exit_points = []  # ���ڱ���Բ������ںͳ��ڵ�

    # ��ÿ��Բ�̽��зֶ�·���滮
    for i, ring_name in enumerate(rings):
        ring = rings[ring_name]

        # ������ڵ�ͳ��ڵ�
        entry_point, exit_point = generate_ring_entry_exit(ring, safety_margin)

        # �����ڵ�ͳ��ڵ��Ƿ��ϰ��︲��
        if is_in_obstacle(entry_point, rings, safety_margin):
            print(f"Entry point {entry_point} is in obstacle!")
        if is_in_obstacle(exit_point, rings, safety_margin):
            print(f"Exit point {exit_point} is in obstacle!")

        # ������ںͳ��ڵ�
        entry_exit_points.append(entry_point)
        entry_exit_points.append(exit_point)

        # �ӵ�ǰ�㵽��ڵ��·����A* ������
        path_to_entry, node_positions_entry = a_star(current_point, entry_point, rings, safety_margin, step_size)

        # ����·���εĽڵ�����
        all_node_positions.extend(node_positions_entry)

        if len(path_to_entry) > 1:
            full_path.extend(path_to_entry)
            ax.plot(*np.array(path_to_entry).T, color='green', label=f'Path to {ring_name} entry')

        # ����ڵ�ֱ�����ߵ����ڵ�
        full_path.append(exit_point)
        ax.plot([entry_point[0], exit_point[0]], [entry_point[1], exit_point[1]], [entry_point[2], exit_point[2]], color='red', label=f'{ring_name} entry to exit')

        # ���µ�ǰ��Ϊ���ڵ�
        current_point = exit_point

    # �����һ�����ĳ��ڵ㵽�յ��·����A* ������
    path_to_goal, node_positions_goal = a_star(current_point, end_point, rings, safety_margin, step_size)

    # ��������·���εĽڵ�����
    all_node_positions.extend(node_positions_goal)

    if len(path_to_goal) > 1:
        full_path.extend(path_to_goal)
        ax.plot(*np.array(path_to_goal).T, color='blue', label='Path to goal')

    # ��ȡ�ؼ��ڵ�
    key_nodes = extract_key_nodes(full_path, entry_exit_points)
    print("Key nodes:", key_nodes)

    # ����ؼ��ڵ㵽 CSV
    save_positions_to_csv(key_nodes, "key_nodes.csv")

    # �������нڵ����굽 CSV �ļ�
    save_positions_to_csv(all_node_positions, "all_a_star_nodes.csv")

    # ��������յ�
    ax.scatter(*start_point, color='blue', s=50, label='Start Point')  # ���
    ax.text(start_point[0], start_point[1], start_point[2], 'Start', color='blue')

    ax.scatter(*end_point, color='red', s=50, label='End Point')  # �յ�
    ax.text(end_point[0], end_point[1], end_point[2], 'End', color='red')

    # ���� x, y, z ��ȳ����ҿ̶ȼ��һ��Ϊ 0.1
    set_axes_equal_and_uniform(ax, tick_interval=0.5)

    # ��ʾ���Ƶ�ͼ��
    plt.legend()
    plt.show()

# ����ڵ����굽 CSV �ļ�
def save_positions_to_csv(node_positions, csv_filename):
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # д���ͷ
        writer.writerow(["X", "Y", "Z"])
        # д��ڵ�����
        for position in node_positions:
            writer.writerow(position)
    print(f"Saved node positions to {csv_filename}")

main()


