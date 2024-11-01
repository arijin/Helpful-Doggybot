import numpy as np
import torch

class RobotTrainingManager:
    def __init__(self, field_size=18, grid_size=3, angle_resolution=10, min_distance=1, max_distance=2):
        self.field_size = field_size  # 场地大小 18x18 米
        self.grid_size = grid_size    # 网格大小 3x3 米
        self.angle_resolution = angle_resolution  # 朝向角度的划分精度 (10 度)
        self.min_distance = min_distance  # 新目标点的最小距离
        self.max_distance = max_distance  # 新目标点的最大距离
        
        # 初始化场地网格和方向计数
        self.grid_coverage = np.zeros((field_size // grid_size, field_size // grid_size), dtype=int)
        self.angle_coverage = np.zeros(360 // angle_resolution, dtype=int)

    def update_coverage(self, position, orientation):
        # 更新网格覆盖
        grid_x, grid_y = int(position[0] // self.grid_size), int(position[1] // self.grid_size)
        self.grid_coverage[grid_x, grid_y] += 1

        # 更新方向覆盖
        angle_index = int(orientation // self.angle_resolution) % (360 // self.angle_resolution)
        self.angle_coverage[angle_index] += 1

    def select_next_target(self, current_position):
        # Step 1: 根据覆盖度低的网格优先选择新网格
        grid_x, grid_y = self.select_least_covered_grid()
        base_x, base_y = grid_x * self.grid_size, grid_y * self.grid_size
        
        # Step 2: 从该网格中选定一个距离在1-2米范围内的新目标点
        distance = np.random.uniform(self.min_distance, self.max_distance)
        angle = np.random.uniform(0, 2 * np.pi)  # 随机角度选择
        offset_x, offset_y = distance * np.cos(angle), distance * np.sin(angle)
        target_position = np.clip([base_x + offset_x, base_y + offset_y], 0, self.field_size)

        return target_position

    def select_least_covered_grid(self):
        # 选择覆盖最少的网格
        min_coverage_value = np.min(self.grid_coverage)
        least_covered_grids = np.argwhere(self.grid_coverage == min_coverage_value)
        selected_grid = least_covered_grids[np.random.choice(len(least_covered_grids))]
        return selected_grid[0], selected_grid[1]

    def select_underrepresented_orientation(self):
        # 选择覆盖度较低的方向角度
        min_coverage_value = np.min(self.angle_coverage)
        least_covered_angles = np.argwhere(self.angle_coverage == min_coverage_value).flatten()
        selected_angle = least_covered_angles[np.random.choice(len(least_covered_angles))]
        return selected_angle * self.angle_resolution

    def training_step(self, robot_position, robot_orientation):
        # 更新当前点的覆盖情况
        self.update_coverage(robot_position, robot_orientation)

        # 选择新目标点并更新朝向
        target_position = self.select_next_target(robot_position)
        target_orientation = self.select_underrepresented_orientation()
        
        return target_position, target_orientation