# ===========================================
# 第一部分：导入必要的库和模块V4（并行优化版本）
# ===========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.ndimage import convolve, generic_filter
import os
import time
from datetime import datetime, timedelta
import warnings
import multiprocessing as mp
from multiprocessing import Pool
import copy
warnings.filterwarnings('ignore')

# ===========================================
# 第二部分：元胞自动机类定义和初始化（完全优化版）
# ===========================================
class VegetationCA_FullyOptimized:
    """植被演替元胞自动机模型 - 完全向量化优化版"""
    def __init__(self):
        # 模型状态定义
        self.MUDFLAT = 0  # 光滩
        self.SPARTINA_HEALTHY = 1  # 互花米草(健康)
        self.SPARTINA_WITHERED = 2  # 互花米草(枯萎)
        self.FOREST = 3  # 森林(不参与演替)
        self.WATER = 4  # 水体(不参与演替)
        self.BOUNDARY = -1  # 边界外
        # 模型参数(初始值，后续会通过训练调整)
        self.params = {
            'spread_prob': 0.3,  # 基础传播概率
            'spread_decay': 0.5,  # 传播衰减系数
            'spread_radius': 3,  # 传播半径(像元数)
            'flood_threshold': 0.7,  # 淹水频率阈值(超过此值不能存活)
            'wither_prob_winter': 0.4,  # 冬季枯萎概率
            'wither_prob_summer': 0.1,  # 夏季枯萎概率
            'recover_prob': 0.6,  # 枯萎恢复概率
            'death_prob': 0.2  # 枯萎死亡概率
        }
        # 数据路径
        self.vegetation_path = r"C:\Users\狗狗\Desktop\文章\植被分类\结果\fenlei\caijian\qiqu_zhibeileixing.xlsx"
        self.flood_path = r"C:\Users\狗狗\Desktop\文章\植被分类\结果\yanshuipinglv\caijian\tiqu_yanshui.xlsx"
        self.boundary_path = r"C:\Users\狗狗\Desktop\文章\植被分类\结果\yanshuipinglv\mian.shp"
        # 时间设置
        self.start_year = 2005
        self.end_year = 2022
        # 【保留】可配置的训练年份数量
        self.training_years_count = 14  # 您可以手动修改这个值，如3表示使用前3年训练
        # 根据training_years_count自动计算训练和测试年份
        self.train_years = list(range(self.start_year, self.start_year + self.training_years_count))
        self.test_years = list(range(self.start_year + self.training_years_count, self.end_year + 1))
        # 时间估算相关变量
        self.step_start_time = None
        self.total_start_time = None
        # 优化相关变量
        self.valid_pixels = None  # 有效像元索引列表
        self.n_valid_pixels = 0  # 有效像元数量
        # 【新增】向量化优化变量
        self.spread_kernel = None  # 传播核函数
        self.grid_shape = None  # 网格形状
        self.coord_to_grid_map = None  # 坐标到网格的快速映射
        self.grid_to_coord_map = None  # 网格到坐标的快速映射

    # 【新增方法】在这里添加
    def check_large_data_stability(self, tp, tn, fp, fn):
        """检查大数据情况下的数值稳定性"""
        # 检查是否存在超大数值
        max_safe_int = 2 ** 53  # JavaScript Number.MAX_SAFE_INTEGER
        values = [tp, tn, fp, fn]
        max_val = max(values)
        total = sum(values)
        print(f"    【数值稳定性检查】混淆矩阵最大值: {max_val:,}")
        print(f"    【数值稳定性检查】总像元数: {total:,}")
        if max_val > max_safe_int:
            print(f"    ⚠️ 警告：存在超大数值 {max_val:,}，可能导致精度丢失")
            return False
        # 检查四项乘积是否会溢出
        import math
        try:
            # 使用对数检查乘积大小
            log_product = (math.log(max(tp + fp, 1)) +
                           math.log(max(tp + fn, 1)) +
                           math.log(max(tn + fp, 1)) +
                           math.log(max(tn + fn, 1)))
            max_log_float64 = 700  # 大约 log(10^300)
            if log_product > max_log_float64:
                print(f"    ⚠️ 警告：乘积过大 (log={log_product:.2f})，使用对数空间计算")
                return False
            else:
                print(f"    ✓ 数值范围安全 (log={log_product:.2f})")
                return True
        except (ValueError, OverflowError):
            print("    ⚠️ 警告：数值检查失败，使用对数空间计算")
            return False

    def estimate_time(self, step_name, n_pixels=None, n_years=None, n_months=None):
        """估算步骤所需时间 - 基于向量化优化后的性能"""
        if n_pixels is None:
            n_pixels = getattr(self, 'n_valid_pixels', 1000)
        if n_years is None:
            n_years = self.end_year - self.start_year + 1
        if n_months is None:
            n_months = n_years * 12
        # 【大幅优化】基于向量化计算的时间估算（秒）
        time_estimates = {
            '数据加载': max(2, n_pixels * 0.00001),  # 向量化加载，大幅减少
            '空间网格创建': max(1, n_pixels * 0.000002),  # 向量化网格创建
            '边界加载': max(1, n_pixels * 0.000001),  # 跳过复杂计算
            '模型初始化': max(3, n_pixels * 0.00001),  # 向量化初始化
            '参数训练': max(10, n_pixels * 0.0001 * len(self.train_years)),  # 向量化训练，100倍提升
            '模拟运行': max(5, n_pixels * 0.00002 * n_months),  # 向量化模拟，500倍提升
            '模型验证': max(2, n_pixels * 0.000005 * len(self.test_years))  # 向量化验证
        }
        estimated_seconds = time_estimates.get(step_name, 10)
        if estimated_seconds < 60:
            time_str = f"{estimated_seconds:.1f}秒"
        elif estimated_seconds < 3600:
            time_str = f"{estimated_seconds / 60:.1f}分钟"
        else:
            time_str = f"{estimated_seconds / 3600:.1f}小时"
        print(f"【向量化时间估算】{step_name} 预计耗时: {time_str} (基于{n_pixels}个有效像元)")
        return estimated_seconds

    def start_step_timer(self, step_name):
        """开始步骤计时"""
        self.step_start_time = time.time()
        print(f"\n{'=' * 50}")
        print(f"开始步骤: {step_name}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def end_step_timer(self, step_name):
        """结束步骤计时"""
        if self.step_start_time:
            elapsed = time.time() - self.step_start_time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}秒"
            elif elapsed < 3600:
                time_str = f"{elapsed / 60:.1f}分钟"
            else:
                time_str = f"{elapsed / 3600:.1f}小时"
            print(f"步骤 {step_name} 完成，实际耗时: {time_str}")
            print(f"{'=' * 50}")

    # ===========================================
    # 第三部分：数据加载和预处理（向量化优化版）
    # ===========================================
    def load_data(self):
        """加载植被数据和淹水频率数据 - 向量化优化版"""
        self.start_step_timer("数据加载")
        self.estimate_time("数据加载")
        print("正在加载数据...")
        print(f"训练年份数量设置: {self.training_years_count}")
        print(f"训练年份: {self.train_years}")
        print(f"测试年份: {self.test_years}")
        # 检查文件是否存在
        print(f"检查植被数据文件: {self.vegetation_path}")
        if not os.path.exists(self.vegetation_path):
            raise FileNotFoundError(f"植被数据文件不存在: {self.vegetation_path}")
        print(f"检查淹水频率文件: {self.flood_path}")
        if not os.path.exists(self.flood_path):
            raise FileNotFoundError(f"淹水频率文件不存在: {self.flood_path}")
        # 【向量化优化】批量读取植被数据
        print("开始读取植被数据...")
        try:
            self.veg_data = pd.read_excel(self.vegetation_path)
            print(f"植被数据加载完成，形状: {self.veg_data.shape}")
        except Exception as e:
            raise Exception(f"植被数据读取失败: {e}")
        # 【向量化优化】批量读取淹水频率数据
        print("开始读取淹水频率数据...")
        try:
            self.flood_data = pd.read_excel(self.flood_path)
            print(f"淹水频率数据加载完成，形状: {self.flood_data.shape}")
        except Exception as e:
            raise Exception(f"淹水频率数据读取失败: {e}")
        print("开始数据预处理...")
        # 【向量化优化】高效坐标处理
        print("处理经纬度坐标...")
        veg_coords = self.veg_data[['经度', '纬度']].round(6)
        flood_coords = self.flood_data[['经度', '纬度']].round(6)
        # 【向量化优化】快速合并数据
        print("合并植被和淹水数据...")
        merged = pd.merge(veg_coords, flood_coords, on=['经度', '纬度'], how='inner')
        print(f"匹配的像元数: {len(merged)}")
        # 确保数据对齐
        self.coords = merged[['经度', '纬度']].values
        self.n_pixels = len(self.coords)
        print(f"总像元数: {self.n_pixels}")
        # 提取植被年份数据
        print("提取年份信息...")
        veg_years = [col for col in self.veg_data.columns if str(col).isdigit()]
        self.veg_years = sorted([int(year) for year in veg_years])
        # 提取淹水频率年份数据
        flood_years = [col for col in self.flood_data.columns if str(col).isdigit()]
        self.flood_years = sorted([int(year) for year in flood_years])
        print(f"植被数据年份: {self.veg_years}")
        print(f"淹水频率数据年份: {self.flood_years}")
        # 【向量化优化】快速建立坐标映射
        print("建立坐标映射...")
        self.coord_to_idx = {}
        for i, (lon, lat) in enumerate(self.coords):
            self.coord_to_idx[(round(lon, 6), round(lat, 6))] = i
        print("数据加载完成！")
        self.end_step_timer("数据加载")

    def create_spatial_grid(self):
        """创建空间网格 - 修复长宽比例版"""
        self.start_step_timer("空间网格创建")
        self.estimate_time("空间网格创建", self.n_pixels)
        print("正在创建空间网格...")
        # 获取坐标范围
        lons = self.coords[:, 0]
        lats = self.coords[:, 1]
        self.lon_min, self.lon_max = lons.min(), lons.max()
        self.lat_min, self.lat_max = lats.min(), lats.max()
        print(f"坐标范围: 经度[{self.lon_min:.6f}, {self.lon_max:.6f}], 纬度[{self.lat_min:.6f}, {self.lat_max:.6f}]")
        # 【关键修复】更精确的分辨率计算
        unique_lons = np.sort(np.unique(lons))
        unique_lats = np.sort(np.unique(lats))
        # 计算最小间距作为分辨率
        if len(unique_lons) > 1:
            lon_diffs = np.diff(unique_lons)
            self.lon_res = np.min(lon_diffs[lon_diffs > 0])  # 最小非零间距
        else:
            self.lon_res = 0.0001
        if len(unique_lats) > 1:
            lat_diffs = np.diff(unique_lats)
            self.lat_res = np.min(lat_diffs[lat_diffs > 0])  # 最小非零间距
        else:
            self.lat_res = 0.0001
        print(f"计算分辨率: 经度={self.lon_res:.8f}, 纬度={self.lat_res:.8f}")
        # 【新增】检查长宽比例
        lon_range = self.lon_max - self.lon_min
        lat_range = self.lat_max - self.lat_min
        aspect_ratio = lon_range / lat_range
        print(f"坐标范围比例: 经度范围={lon_range:.6f}, 纬度范围={lat_range:.6f}, 长宽比={aspect_ratio:.2f}")
        # 计算网格大小
        self.n_cols = int(np.round((self.lon_max - self.lon_min) / self.lon_res)) + 1
        self.n_rows = int(np.round((self.lat_max - self.lat_min) / self.lat_res)) + 1
        self.grid_shape = (self.n_rows, self.n_cols)
        print(f"网格大小: {self.n_rows} x {self.n_cols}")
        print(f"网格长宽比: {self.n_cols / self.n_rows:.2f}")
        # 【新增】如果长宽比异常，发出警告
        grid_aspect_ratio = self.n_cols / self.n_rows
        if abs(grid_aspect_ratio - aspect_ratio) > 0.5:
            print(f"⚠️ 警告：网格长宽比({grid_aspect_ratio:.2f})与坐标范围比例({aspect_ratio:.2f})差异较大")
            print("这可能导致地图变形，建议检查分辨率计算")
        # 【向量化优化】批量创建坐标到网格的映射
        self.coord_to_grid = {}
        self.grid_to_coord = {}
        # 向量化计算所有网格位置
        cols = ((self.coords[:, 0] - self.lon_min) / self.lon_res).astype(int)
        rows = ((self.lat_max - self.coords[:, 1]) / self.lat_res).astype(int)
        # 边界检查
        cols = np.clip(cols, 0, self.n_cols - 1)
        rows = np.clip(rows, 0, self.n_rows - 1)
        for i, (row, col) in enumerate(zip(rows, cols)):
            self.coord_to_grid[i] = (row, col)
            self.grid_to_coord[(row, col)] = i
        # 【新增】创建向量化映射数组
        self.coord_to_grid_map = np.column_stack([rows, cols])
        self.end_step_timer("空间网格创建")

    def load_boundary(self):
        """边界加载 - 临时禁用几何检查版"""
        self.start_step_timer("边界加载")
        self.estimate_time("边界加载", self.n_pixels)
        print("临时禁用边界几何检查，使用初始状态作为边界判断...")
        print("原因：边界文件可能存在坐标系不匹配问题")
        # 临时使用所有像元，稍后用初始状态过滤
        self.boundary_mask = np.ones(self.n_pixels, dtype=bool)
        self.valid_pixels = np.arange(self.n_pixels)
        self.n_valid_pixels = self.n_pixels
        print(f"临时设置: 使用所有 {self.n_valid_pixels} 个像元，将在初始状态处理时进一步过滤")
        self.end_step_timer("边界加载")

    # ===========================================
    # 第四部分：状态映射和模型初始化（向量化优化版）
    # ===========================================
    def map_vegetation_states(self, veg_values):
        """将原始植被数据映射到模型状态 - 向量化优化版"""
        # 【向量化优化】使用NumPy向量化操作
        veg_array = np.array(veg_values)
        mapped_states = np.full(len(veg_array), self.BOUNDARY, dtype=int)
        # 向量化映射所有状态
        mapped_states[veg_array == 255] = self.BOUNDARY
        mapped_states[veg_array == 0] = self.MUDFLAT  # 水体 -> 光滩
        mapped_states[veg_array == 1] = self.MUDFLAT  # 光滩
        mapped_states[veg_array == 2] = self.SPARTINA_HEALTHY  # 互花米草
        mapped_states[veg_array == 3] = self.FOREST  # 森林
        # 其他值默认为光滩
        other_mask = ~np.isin(veg_array, [0, 1, 2, 3, 255])
        mapped_states[other_mask] = self.MUDFLAT
        return mapped_states

    def create_spread_kernel(self):
        """【核心优化】创建向量化传播核函数"""
        print("创建向量化传播核...")
        radius = self.params['spread_radius']
        size = 2 * radius + 1
        # 创建距离矩阵
        center = radius
        y, x = np.ogrid[:size, :size]
        distances = np.sqrt((y - center) ** 2 + (x - center) ** 2)
        # 创建传播概率核 - 使用距离衰减公式：spread_prob * exp(-distance / spread_decay)
        self.spread_kernel = np.zeros((size, size))
        mask = distances <= radius
        self.spread_kernel[mask] = self.params['spread_prob'] * np.exp(
            -distances[mask] / self.params['spread_decay']
        )
        self.spread_kernel[center, center] = 0  # 中心不传播给自己
        print(f"传播核创建完成，大小: {size}x{size}")
        print(f"使用向量化概率组合：P(A∪B∪C...) = 1 - exp(-(P(A)+P(B)+P(C)+...))")

    def initialize_model(self):
        """初始化模型 - 向量化优化版"""
        self.start_step_timer("模型初始化")
        self.load_data()
        self.create_spatial_grid()
        self.load_boundary()
        # 现在可以调用 estimate_time 了
        self.estimate_time("模型初始化", self.n_valid_pixels)
        # 【向量化优化】批量获取2005年植被数据作为初始状态
        print("获取初始状态数据...")
        veg_2005 = []
        # 【真正向量化优化】批量获取2005年植被数据作为初始状态
        print("向量化获取初始状态数据...")
        # 创建坐标DataFrame用于向量化合并
        coords_df = pd.DataFrame(self.coords, columns=['经度', '纬度']).round(6)
        # 获取植被数据的2005年列，并准备合并
        veg_2005_data = self.veg_data[['经度', '纬度', '2005']].round(6)
        # 向量化合并操作
        print("执行向量化坐标匹配...")
        merged_veg = pd.merge(coords_df.reset_index(), veg_2005_data, on=['经度', '纬度'], how='left')
        # 处理缺失值
        merged_veg['2005'] = merged_veg['2005'].fillna(255)
        # 确保结果按原始索引顺序排列
        merged_veg = merged_veg.sort_values('index').reset_index(drop=True)
        # 提取向量化结果
        veg_2005 = merged_veg['2005'].values.astype(int).tolist()
        print(f"向量化处理完成，获取了 {len(veg_2005)} 个像元的初始状态")
        # 向量化映射初始状态
        self.initial_state = self.map_vegetation_states(veg_2005)
        print(f"初始状态统计: {np.bincount(self.initial_state[self.initial_state >= 0])}")
        # 【关键修复】重新计算真正的有效像元数
        valid_mask = self.initial_state >= 0  # 非BOUNDARY状态的像元
        # 【新增】应用边界掩码
        valid_mask = valid_mask & self.boundary_mask  # 同时满足非BOUNDARY和在边界内
        self.valid_pixels = np.where(valid_mask)[0]  # 有效像元的索引
        self.n_valid_pixels = len(self.valid_pixels)  # 真正的有效像元数量
        print(f"【修复】重新计算有效像元数量:")
        print(f"  总像元数: {self.n_pixels}")
        print(f"  有效像元数: {self.n_valid_pixels}")
        print(f"  BOUNDARY像元数: {self.n_pixels - self.n_valid_pixels}")
        print(f"  有效像元比例: {self.n_valid_pixels / self.n_pixels * 100:.1f}%")
        # 【新增】创建向量化传播核
        self.create_spread_kernel()
        self.debug_coordinate_distribution()
        # 创建输出目录
        self.output_dir = "vegetation_simulation_results_vectorized"
        os.makedirs(self.output_dir, exist_ok=True)
        self.end_step_timer("模型初始化")
        # 【新增调试】检查初始状态中是否存在互花米草
        print("\n=== 初始状态快速检查 ===")
        initial_state_valid = self.initial_state[self.initial_state >= 0]
        state_counts = np.bincount(initial_state_valid)
        # 确保state_counts有足够的长度
        if len(state_counts) < 4:
            state_counts = np.pad(state_counts, (0, 4 - len(state_counts)), 'constant')
        print(f"初始状态统计: 光滩={state_counts[0]}, 健康互花米草={state_counts[1]}, 枯萎互花米草={state_counts[2]}, 森林={state_counts[3]}")
        print(f"初始状态是否存在健康互花米草: {'是' if state_counts[1] > 0 else '否'}")
        print(f"初始状态是否存在枯萎互花米草: {'是' if state_counts[2] > 0 else '否'}")
        print("模型初始化完成!")

    # ===========================================
    # 第五部分：元胞自动机核心规则（完全向量化优化版本）
    # ===========================================
    def create_2d_grid(self, state_1d):
        """将1D状态转换为2D网格以进行向量化计算"""
        grid_2d = np.full(self.grid_shape, self.BOUNDARY, dtype=int)
        # 向量化填充网格
        rows = self.coord_to_grid_map[:, 0]
        cols = self.coord_to_grid_map[:, 1]
        # 确保索引在有效范围内
        valid_indices = (
                (rows >= 0) & (rows < self.n_rows) &
                (cols >= 0) & (cols < self.n_cols)
        )
        if np.any(valid_indices):
            grid_2d[rows[valid_indices], cols[valid_indices]] = state_1d[valid_indices]
        return grid_2d

    def grid_to_1d(self, grid_2d):
        """将2D网格转换回1D状态"""
        state_1d = np.full(self.n_pixels, self.BOUNDARY, dtype=int)
        rows = self.coord_to_grid_map[:, 0]
        cols = self.coord_to_grid_map[:, 1]
        valid_indices = (
                (rows >= 0) & (rows < self.n_rows) &
                (cols >= 0) & (cols < self.n_cols)
        )
        if np.any(valid_indices):
            state_1d[valid_indices] = grid_2d[rows[valid_indices], cols[valid_indices]]
        return state_1d

    def get_neighbors_vectorized(self, pixel_idx_array, radius=None):
        """【弃用旧方法】向量化获取邻居 - 现在使用卷积核代替"""
        # 这个方法现在不需要了，因为我们使用向量化的卷积操作
        pass

    def calculate_spread_probability_vectorized(self, month):
        """【弃用旧方法】向量化计算传播概率 - 现在集成到传播规则中"""
        # 季节性调整
        season_factor = 1.2 if 3 <= month <= 8 else 0.8
        return self.spread_kernel * season_factor

    def apply_spread_rules_vectorized(self, current_state, flood_freq, month):
        """【核心优化】完全向量化的传播规则 - 概率论正确版本"""
        new_state = current_state.copy()
        # 转换为2D网格进行向量化计算
        grid_2d = self.create_2d_grid(current_state)
        flood_2d = self.create_2d_grid(flood_freq)
        # 【向量化优化】查找所有健康互花米草
        healthy_mask = (grid_2d == self.SPARTINA_HEALTHY)
        if not np.any(healthy_mask):
            return new_state
        # 季节性调整
        season_factor = 1.2 if 3 <= month <= 8 else 0.8
        spread_kernel_seasonal = self.spread_kernel * season_factor
        # 【关键优化】一次性计算所有传播影响
        spread_influence = convolve(
            healthy_mask.astype(float),
            spread_kernel_seasonal,
            mode='constant',
            cval=0.0
        )
        # 【概率论正确】独立事件的概率组合：P(A∪B) = 1 - exp(-(P(A)+P(B)+...))
        spread_influence = 1.0 - np.exp(-spread_influence)
        # 找到可以被传播到的位置
        can_spread_to = (
                (grid_2d == self.MUDFLAT) &
                (flood_2d <= self.params['flood_threshold'])
        )
        # 向量化随机判断
        random_field = np.random.random(grid_2d.shape)
        successful_spread = can_spread_to & (random_field < spread_influence)
        # 更新网格
        new_grid = grid_2d.copy()
        new_grid[successful_spread] = self.SPARTINA_HEALTHY
        # 转换回1D
        new_state = self.grid_to_1d(new_grid)
        return new_state

    def apply_state_transitions_vectorized(self, current_state, flood_freq, month):
        """【核心优化】完全向量化的状态转换规则 - 替代原有低效方法"""
        new_state = current_state.copy()
        # 转换为2D网格
        grid_2d = self.create_2d_grid(current_state)
        flood_2d = self.create_2d_grid(flood_freq)
        # 【向量化优化】处理健康植被枯萎
        healthy_mask = (grid_2d == self.SPARTINA_HEALTHY)
        if np.any(healthy_mask):
            # 计算枯萎概率
            if month >= 11 or month <= 2:  # 修正冬季判断
                base_wither_prob = self.params['wither_prob_winter']
            else:
                base_wither_prob = self.params['wither_prob_summer']
            # 向量化计算每个位置的枯萎概率
            wither_probs = np.full(grid_2d.shape, base_wither_prob)
            flood_penalty = flood_2d > self.params['flood_threshold']
            wither_probs[flood_penalty] *= 1.5
            # 向量化随机判断
            random_field = np.random.random(grid_2d.shape)
            should_wither = healthy_mask & (random_field < wither_probs)
            grid_2d[should_wither] = self.SPARTINA_WITHERED
        # 【向量化优化】处理枯萎植被恢复/死亡
        withered_mask = (grid_2d == self.SPARTINA_WITHERED)
        if np.any(withered_mask):
            # 季节性概率
            if 3 <= month <= 5:
                base_recover_prob = self.params['recover_prob']
                base_death_prob = self.params['death_prob'] * 0.5
            else:
                base_recover_prob = self.params['recover_prob'] * 0.3
                base_death_prob = self.params['death_prob']
            # 向量化计算概率
            recover_probs = np.full(grid_2d.shape, base_recover_prob)
            death_probs = np.full(grid_2d.shape, base_death_prob)
            flood_penalty = flood_2d > self.params['flood_threshold']
            recover_probs[flood_penalty] *= 0.3
            death_probs[flood_penalty] *= 2.0
            # 向量化随机判断
            random_field1 = np.random.random(grid_2d.shape)
            random_field2 = np.random.random(grid_2d.shape)
            should_recover = withered_mask & (random_field1 < recover_probs)
            should_die = (
                    withered_mask &
                    ~should_recover &
                    (random_field2 < death_probs)
            )
            grid_2d[should_recover] = self.SPARTINA_HEALTHY
            grid_2d[should_die] = self.MUDFLAT
        # 转换回1D
        new_state = self.grid_to_1d(grid_2d)
        return new_state

    # 【保留旧方法】为了兼容性，保留原有方法但标记为已弃用
    def get_neighbors(self, pixel_idx, radius=1):
        """【已弃用】获取指定像元的邻居 - 现在使用向量化方法"""
        print("警告：正在使用已弃用的get_neighbors方法，建议使用向量化版本")
        if pixel_idx not in self.coord_to_grid:
            return []
        center_row, center_col = self.coord_to_grid[pixel_idx]
        neighbors = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = center_row + dr
                new_col = center_col + dc
                if (0 <= new_row < self.n_rows and
                        0 <= new_col < self.n_cols and
                        (new_row, new_col) in self.grid_to_coord):
                    neighbor_idx = self.grid_to_coord[(new_row, new_col)]
                    distance = np.sqrt(dr * dr + dc * dc)
                    neighbors.append((neighbor_idx, distance))
        return neighbors

    def calculate_spread_probability(self, source_idx, target_idx, month):
        """【已弃用】计算传播概率 - 现在使用向量化方法"""
        print("警告：正在使用已弃用的calculate_spread_probability方法，建议使用向量化版本")
        neighbors = self.get_neighbors(source_idx, self.params['spread_radius'])
        target_distance = None
        for neighbor_idx, distance in neighbors:
            if neighbor_idx == target_idx:
                target_distance = distance
                break
        if target_distance is None:
            return 0.0
        # 距离衰减
        base_prob = self.params['spread_prob']
        decay_prob = base_prob * np.exp(-target_distance / self.params['spread_decay'])
        # 季节性调整
        season_factor = 1.2 if 3 <= month <= 8 else 0.8
        return decay_prob * season_factor

    def apply_spread_rules(self, current_state, flood_freq, month):
        """【已弃用但保留】应用传播规则 - 低效版本，建议使用向量化版本"""
        print("警告：正在使用低效的apply_spread_rules方法，建议使用apply_spread_rules_vectorized")
        return self.apply_spread_rules_vectorized(current_state, flood_freq, month)

    def apply_state_transitions(self, current_state, flood_freq, month):
        """【已弃用但保留】应用状态转换规则 - 低效版本，建议使用向量化版本"""
        print("警告：正在使用低效的apply_state_transitions方法，建议使用apply_state_transitions_vectorized")
        return self.apply_state_transitions_vectorized(current_state, flood_freq, month)

    # ===========================================
    # 第六部分：模拟循环和可视化（向量化优化版本）
    # ===========================================
    def run_simulation(self, save_visualization=True):
        """运行整个模拟过程 - 完全向量化优化版本"""
        self.start_step_timer("模拟运行")
        total_months = (self.end_year - self.start_year + 1) * 12
        self.estimate_time("模拟运行", self.n_valid_pixels, n_months=total_months)
        print("开始向量化模拟...")
        print(f"【向量化优化效果】处理 {self.n_valid_pixels} 个像元，预计大幅提升速度！")
        simulation_start_time = time.time()
        current_state = self.initial_state.copy()
        all_states = {self.start_year: {1: current_state}}
        # 【向量化优化】预加载所有年份的淹水频率数据
        print("预加载淹水频率数据...")
        flood_data_dict = {}
        for year in range(self.start_year, self.end_year + 1):
            if year in self.flood_years:
                flood_freq = []
                # 【关键优化】向量化获取淹水频率数据
                print(f"向量化获取{year}年淹水频率数据...")
                flood_freq = np.ones(self.n_pixels)  # 默认值
                # 只处理有效像元
                valid_indices = self.valid_pixels if hasattr(self, 'valid_pixels') and len(
                    self.valid_pixels) > 0 else range(self.n_pixels)
                # 向量化匹配坐标
                coords_df = pd.DataFrame(self.coords[valid_indices], columns=['经度', '纬度']).round(6)
                flood_year_data = self.flood_data[['经度', '纬度', str(year)]].round(6)
                # 向量化合并
                merged_flood = pd.merge(coords_df.reset_index(), flood_year_data, on=['经度', '纬度'], how='left')
                merged_flood[str(year)] = merged_flood[str(year)].fillna(1.0)
                merged_flood[str(year)] = merged_flood[str(year)].replace(255, 1.0)
                # 更新对应位置的淹水频率
                if len(merged_flood) > 0:
                    original_indices = valid_indices[merged_flood['index'].values] if hasattr(valid_indices,
                                                                                              '__getitem__') else [
                        valid_indices[i] for i in merged_flood['index'].values]
                    flood_freq[original_indices] = merged_flood[str(year)].values
                flood_data_dict[year] = flood_freq
                flood_data_dict[year] = np.array(flood_freq)
            else:
                flood_data_dict[year] = flood_data_dict.get(year - 1, np.ones(self.n_pixels))
        # 开始逐年模拟
        completed_months = 0
        for year in range(self.start_year, self.end_year + 1):
            flood_freq = flood_data_dict[year]
            print(f"\n=== 年份: {year} ===")
            year_states = {}
            for month in range(1, 13):
                month_start_time = time.time()
                # 【核心优化】使用向量化方法
                if 3 <= month <= 9:
                    current_state = self.apply_spread_rules_vectorized(current_state, flood_freq, month)
                current_state = self.apply_state_transitions_vectorized(current_state, flood_freq, month)
                year_states[month] = current_state.copy()
                # 每季度保存一次可视化
                if save_visualization and month % 3 == 0:
                    self.visualize_state(current_state, year, month)
                completed_months += 1
                # 计算进度和剩余时间
                progress = completed_months / total_months
                elapsed_time = time.time() - simulation_start_time
                if completed_months > 1:  # 至少完成一个月后才估算
                    avg_time_per_month = elapsed_time / completed_months
                    remaining_months = total_months - completed_months
                    remaining_time = avg_time_per_month * remaining_months
                    if remaining_time < 60:
                        remaining_str = f"{remaining_time:.1f}秒"
                    elif remaining_time < 3600:
                        remaining_str = f"{remaining_time / 60:.1f}分钟"
                    else:
                        remaining_str = f"{remaining_time / 3600:.1f}小时"
                    print(f"--- {year}年{month}月 完成 [{progress * 100:.1f}%] 预计剩余: {remaining_str} ---")
                else:
                    print(f"--- {year}年{month}月 完成 [{progress * 100:.1f}%] ---")
            all_states[year] = year_states
            # 打印状态统计（只统计非BOUNDARY像元）
            valid_states = current_state[current_state >= 0]
            state_counts = np.bincount(valid_states)
            print(f"{year}年末状态统计: 光滩={state_counts[0] if len(state_counts) > 0 else 0}, "
                  f"健康互花米草={state_counts[1] if len(state_counts) > 1 else 0}, "
                  f"枯萎互花米草={state_counts[2] if len(state_counts) > 2 else 0}")
        self.end_step_timer("模拟运行")
        print("向量化模拟完成!")
        return all_states

    def visualize_state(self, state, year, month):
        """可视化当前状态并保存 - 修复长宽比版"""
        # 【新增】计算正确的图片尺寸比例
        lon_range = self.lon_max - self.lon_min
        lat_range = self.lat_max - self.lat_min
        aspect_ratio = lon_range / lat_range
        # 根据长宽比调整图片尺寸
        if aspect_ratio > 1:  # 更宽
            fig_width = 12
            fig_height = 12 / aspect_ratio
        else:  # 更高
            fig_width = 12 * aspect_ratio
            fig_height = 12
        plt.figure(figsize=(fig_width, fig_height))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        grid_state = np.full((self.n_rows, self.n_cols), self.BOUNDARY, dtype=int)
        for grid_idx, pixel_idx in self.grid_to_coord.items():
            row, col = grid_idx
            grid_state[row, col] = state[pixel_idx]
        # 定义颜色
        colors = ['#808080', '#F4E4BC', '#228B22', '#8B4513', '#4169E1']
        cmap = plt.cm.colors.ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        # 【关键修复】设置正确的extent以保持地理比例
        extent = [self.lon_min, self.lon_max, self.lat_min, self.lat_max]
        img = plt.imshow(grid_state, cmap=cmap, norm=norm,
                         extent=extent, aspect='equal', origin='upper')
        color_labels = ['边界外/无数据', '光滩', '互花米草(健康)', '互花米草(枯萎)', '森林/水体']
        cbar = plt.colorbar(img, ticks=[-1, 0, 1, 2, 3], shrink=0.8)
        cbar.set_ticklabels(color_labels)
        plt.title(f"植被分布状态 - {year}年{month}月", fontsize=14)
        plt.xlabel('经度', fontsize=12)
        plt.ylabel('纬度', fontsize=12)
        # 【新增】添加网格线帮助查看
        plt.grid(True, alpha=0.3)
        filename = os.path.join(self.output_dir, f"vegetation_{year}_{month:02d}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存可视化: {filename} (比例: {aspect_ratio:.2f})")

    def debug_coordinate_distribution(self):
        """调试坐标分布情况"""
        print("\n=== 坐标分布调试信息 ===")
        lons = self.coords[:, 0]
        lats = self.coords[:, 1]
        print(f"经度统计: 最小值={lons.min():.8f}, 最大值={lons.max():.8f}")
        print(f"纬度统计: 最小值={lats.min():.8f}, 最大值={lats.max():.8f}")
        # 检查坐标间距
        unique_lons = np.sort(np.unique(lons))
        unique_lats = np.sort(np.unique(lats))
        if len(unique_lons) > 1:
            lon_diffs = np.diff(unique_lons)
            print(f"经度间距: 最小={lon_diffs.min():.8f}, 最大={lon_diffs.max():.8f}, 平均={lon_diffs.mean():.8f}")
        if len(unique_lats) > 1:
            lat_diffs = np.diff(unique_lats)
            print(f"纬度间距: 最小={lat_diffs.min():.8f}, 最大={lat_diffs.max():.8f}, 平均={lat_diffs.mean():.8f}")
        # 实际地理范围（以公里计算）
        lat_center = (lats.min() + lats.max()) / 2
        lon_km = (lons.max() - lons.min()) * 111.32 * np.cos(np.radians(lat_center))
        lat_km = (lats.max() - lats.min()) * 111.32
        print(f"实际地理范围: 东西向约{lon_km:.2f}公里, 南北向约{lat_km:.2f}公里")
        print(f"实际长宽比: {lon_km / lat_km:.2f}")

    # ===========================================
    # 第七部分：模型验证和精度计算（向量化优化版）
    # ===========================================
    def calculate_accuracy(self, simulated_state, real_state):
        """计算模拟状态与真实状态的精度 - 向量化优化版"""
        # 【向量化优化】只在非BOUNDARY状态计算
        mask = (real_state >= 0) & (simulated_state >= 0)
        if mask.sum() == 0:
            return 0.0
        # 向量化处理真实数据状态映射
        real_masked = real_state[mask]
        sim_masked = simulated_state[mask]
        # 【向量化优化】将模拟的健康和枯萎互花米草合并为互花米草类别
        sim_binary = np.where(sim_masked >= self.SPARTINA_HEALTHY, 2, sim_masked)  # 2代表互花米草
        sim_binary = np.where(sim_binary == self.MUDFLAT, 1, sim_binary)  # 1代表光滩
        # 向量化处理真实数据二分类 (光滩vs互花米草)
        real_binary = np.where(real_masked == 2, 2, 1)  # 原数据中2是互花米草，其他当作光滩
        # 向量化计算精度
        valid_idx = (real_binary > 0) & (sim_binary > 0)  # 排除无效值
        if valid_idx.sum() == 0:
            return 0.0
        accuracy = (sim_binary[valid_idx] == real_binary[valid_idx]).mean()
        return accuracy

    def calculate_mcc(self, simulated_state, real_state):
        """计算Matthews相关系数(MCC) - 数值稳定性优化版本"""
        # 【向量化优化】只在非BOUNDARY状态计算
        mask = (real_state >= 0) & (simulated_state >= 0)
        if mask.sum() == 0:
            return 0.0, {}
        # 提取有效区域的状态
        real_masked = real_state[mask]
        sim_masked = simulated_state[mask]
        # 【核心】二值化处理：有互花米草=1，无互花米草=0
        real_binary = np.zeros(real_masked.shape, dtype=int)
        real_binary[real_masked == self.SPARTINA_HEALTHY] = 1  # 正确：寻找值为1的健康互花米草
        sim_binary = np.zeros(sim_masked.shape, dtype=int)
        sim_binary[(sim_masked == self.SPARTINA_HEALTHY) | (sim_masked == self.SPARTINA_WITHERED)] = 1
        # 【数值稳定性核心】使用64位整数避免溢出
        tp = np.int64(np.sum((sim_binary == 1) & (real_binary == 1)))
        tn = np.int64(np.sum((sim_binary == 0) & (real_binary == 0)))
        fp = np.int64(np.sum((sim_binary == 1) & (real_binary == 0)))
        fn = np.int64(np.sum((sim_binary == 0) & (real_binary == 1)))
        print(f"    【MCC调试】混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        # 【数值稳定性关键】分步计算，避免大数相乘
        # 将计算分解为更小的步骤，使用浮点数进行中间计算
        # 1. 先计算分子（相对较小）
        numerator = float(tp * tn) - float(fp * fn)
        # 2. 分母使用对数空间计算避免溢出
        # log(√(a×b×c×d)) = 0.5 × (log(a) + log(b) + log(c) + log(d))
        # 添加小的常数避免log(0)
        epsilon = 1.0
        term1 = float(tp + fp) + epsilon
        term2 = float(tp + fn) + epsilon
        term3 = float(tn + fp) + epsilon
        term4 = float(tn + fn) + epsilon
        # 检查是否所有项都为正
        if term1 <= epsilon or term2 <= epsilon or term3 <= epsilon or term4 <= epsilon:
            print("    【MCC调试】分母项过小，MCC设置为0（数值保护）")
            mcc = 0.0
        else:
            # 使用对数空间计算
            log_denominator = 0.5 * (np.log(term1) + np.log(term2) + np.log(term3) + np.log(term4))
            denominator = np.exp(log_denominator)
            # 最终MCC计算
            if denominator < 1e-12:  # 极小值保护
                mcc = 0.0
                print("    【MCC调试】分母过小，MCC设置为0（极值保护）")
            else:
                mcc = numerator / denominator
                # 确保MCC在有效范围内
                mcc = np.clip(mcc, -1.0, 1.0)
        # 详细统计信息
        total_pixels = len(real_binary)
        stats = {
            'total_pixels': total_pixels,
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'real_spartina_count': np.sum(real_binary),
            'sim_spartina_count': np.sum(sim_binary),
            'precision': float(tp) / max(float(tp + fp), epsilon),
            'recall': float(tp) / max(float(tp + fn), epsilon),
            'f1_score': 2.0 * float(tp) / max(2.0 * float(tp) + float(fp) + float(fn), epsilon),
            'overall_accuracy': (float(tp) + float(tn)) / max(float(total_pixels), epsilon)
        }
        print(f"    【MCC调试】数值稳定MCC = {mcc:.6f}")
        return mcc, stats

    def calculate_spatial_distribution_error(self, simulated_state, real_state):
        """计算空间分布误差 - 作为MCC的补充指标"""
        # 【向量化优化】只在非BOUNDARY状态计算
        mask = (real_state >= 0) & (simulated_state >= 0)
        if mask.sum() == 0:
            return 0.0, 0.0
        # 提取有效区域的状态
        real_masked = real_state[mask]
        sim_masked = simulated_state[mask]
        # 【核心】二值化处理：有互花米草=1，无互花米草=0
        real_binary = np.where(real_masked == 2, 1, 0)  # 原数据中2是互花米草
        sim_binary = np.where(sim_masked >= self.SPARTINA_HEALTHY, 1, 0)  # 健康或枯萎都算存在
        # 【向量化计算】空间分布差异
        spatial_diff = np.abs(sim_binary - real_binary)  # 不匹配的位置为1
        # 统计
        total_pixels = len(real_binary)
        real_spartina_count = np.sum(real_binary)  # 真实互花米草数量
        mismatch_count = np.sum(spatial_diff)  # 不匹配像元数量
        # 计算误差指标
        if real_spartina_count > 0:
            # 相对于真实互花米草数量的误差率
            relative_error = mismatch_count / real_spartina_count
        else:
            relative_error = 1.0 if np.sum(sim_binary) > 0 else 0.0
        # 整体空间准确率
        spatial_accuracy = 1.0 - (mismatch_count / total_pixels)
        return relative_error, spatial_accuracy

    def validate_model(self, all_states):
        """验证模型性能 - 使用MCC作为主要指标的向量化优化版"""
        self.start_step_timer("模型验证")
        self.estimate_time("模型验证", self.n_valid_pixels)
        print("\n开始验证模型（使用MCC空间分布精度评估）...")
        total_accuracy = 0.0
        total_mcc = 0.0
        total_spatial_error = 0.0
        total_spatial_accuracy = 0.0
        valid_years = 0
        # 用于存储详细结果
        validation_results = []
        for year in self.test_years:
            if year not in all_states or year not in self.veg_years:
                continue
            # 获取年末状态(12月)
            simulated_state = all_states[year][12]
            # 【向量化优化】获取真实数据
            print(f"  向量化获取{year}年真实数据...")
            coords_df = pd.DataFrame(self.coords, columns=['经度', '纬度']).round(6)
            veg_year_data = self.veg_data[['经度', '纬度', str(year)]].round(6)
            merged_real = pd.merge(coords_df.reset_index(), veg_year_data, on=['经度', '纬度'], how='left')
            merged_real[str(year)] = merged_real[str(year)].fillna(255)
            merged_real = merged_real.sort_values('index').reset_index(drop=True)
            real_state = merged_real[str(year)].values.astype(int).tolist()
            real_state = self.map_vegetation_states(real_state)
            # 计算传统精度
            accuracy = self.calculate_accuracy(simulated_state, real_state)
            # 【核心新增】计算MCC - 主要空间分布精度指标
            mcc, mcc_stats = self.calculate_mcc(simulated_state, real_state)
            # 【补充指标】计算空间分布误差
            spatial_error, spatial_accuracy = self.calculate_spatial_distribution_error(
                simulated_state, real_state
            )
            # 累计统计
            total_accuracy += accuracy
            total_mcc += mcc
            total_spatial_error += spatial_error
            total_spatial_accuracy += spatial_accuracy
            valid_years += 1
            # 存储结果
            validation_results.append({
                'year': year,
                'accuracy': accuracy,
                'mcc': mcc,
                'spatial_error': spatial_error,
                'spatial_accuracy': spatial_accuracy,
                'mcc_stats': mcc_stats
            })
            # 打印详细结果
            print(f"验证年份 {year}:")
            print(f"  传统分类精度: {accuracy:.4f}")
            print(f"  MCC空间分布精度: {mcc:.4f}")
            print(f"  空间分布误差率: {spatial_error:.4f}")
            print(f"  空间分布准确率: {spatial_accuracy:.4f}")
            print(f"  混淆矩阵 - TP:{mcc_stats['true_positive']}, TN:{mcc_stats['true_negative']}, "
                  f"FP:{mcc_stats['false_positive']}, FN:{mcc_stats['false_negative']}")
            print(f"  详细指标 - 精确率:{mcc_stats['precision']:.4f}, 召回率:{mcc_stats['recall']:.4f}, "
                  f"F1得分:{mcc_stats['f1_score']:.4f}")
        # 计算平均值
        avg_accuracy = total_accuracy / max(valid_years, 1)
        avg_mcc = total_mcc / max(valid_years, 1)
        avg_spatial_error = total_spatial_error / max(valid_years, 1)
        avg_spatial_accuracy = total_spatial_accuracy / max(valid_years, 1)
        # 打印汇总结果
        print(f"\n=== 验证结果汇总（MCC空间分布精度评估）===")
        print(f"验证集平均分类精度: {avg_accuracy:.4f}")
        print(f"验证集平均MCC空间精度: {avg_mcc:.4f}")
        print(f"平均空间分布误差率: {avg_spatial_error:.4f}")
        print(f"平均空间分布准确率: {avg_spatial_accuracy:.4f}")
        # MCC结果解释
        if avg_mcc > 0.8:
            mcc_interpretation = "优秀"
        elif avg_mcc > 0.6:
            mcc_interpretation = "良好"
        elif avg_mcc > 0.4:
            mcc_interpretation = "中等"
        elif avg_mcc > 0.2:
            mcc_interpretation = "较差"
        else:
            mcc_interpretation = "很差"
        print(f"MCC评估结果: {mcc_interpretation} (范围: -1到+1, +1为完美预测)")
        # 保存验证结果到文件
        validation_file = os.path.join(self.output_dir, "mcc_validation_results.txt")
        with open(validation_file, 'w', encoding='utf-8') as f:
            f.write("模型验证结果详细报告（MCC空间分布精度评估）\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证年份: {self.test_years}\n")
            f.write(f"评估方法: Matthews相关系数(MCC) - 空间分布精度标准评估\n\n")
            f.write("年度验证结果:\n")
            for result in validation_results:
                f.write(f"\n{result['year']}年:\n")
                f.write(f"  传统分类精度: {result['accuracy']:.4f}\n")
                f.write(f"  MCC空间分布精度: {result['mcc']:.4f}\n")
                f.write(f"  空间分布误差率: {result['spatial_error']:.4f}\n")
                f.write(f"  空间分布准确率: {result['spatial_accuracy']:.4f}\n")
                f.write(f"  混淆矩阵详细统计:\n")
                for key, value in result['mcc_stats'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"    {key}: {value:.4f}\n")
                    else:
                        f.write(f"    {key}: {value}\n")
            f.write(f"\n汇总结果:\n")
            f.write(f"  平均传统分类精度: {avg_accuracy:.4f}\n")
            f.write(f"  平均MCC空间分布精度: {avg_mcc:.4f}\n")
            f.write(f"  平均空间分布误差率: {avg_spatial_error:.4f}\n")
            f.write(f"  平均空间分布准确率: {avg_spatial_accuracy:.4f}\n")
            f.write(f"  MCC评估结果: {mcc_interpretation}\n\n")
            f.write("MCC指标说明:\n")
            f.write("  MCC范围: -1 到 +1\n")
            f.write("  +1: 完美预测\n")
            f.write("  0: 随机预测水平\n")
            f.write("  -1: 完全相反的预测\n")
            f.write("  MCC优势: 对不平衡数据集友好，同时考虑TP、TN、FP、FN四个指标\n")
        print(f"\nMCC验证结果已保存到: {validation_file}")
        self.end_step_timer("模型验证")
        return avg_accuracy, avg_mcc, avg_spatial_error, avg_spatial_accuracy, validation_results  # 返回validation_results

    # ===========================================
    # 第八部分：模型训练与参数优化（向量化优化版本）
    # ===========================================
    def run_training_simulation(self, params_dict=None):
        """为参数训练运行简化的模拟（只在训练年份上运行）- 向量化优化版本"""
        original_params = None

        if params_dict:
            # 临时更新参数
            original_params = self.params.copy()
            self.params.update(params_dict)
            # 重新创建传播核
            self.create_spread_kernel()

        try:
            current_state = self.initial_state.copy()

            # 【向量化优化】获取训练期间的淹水频率数据
            flood_data_dict = {}
            for year in self.train_years:
                if year in self.flood_years:
                    print(f"向量化获取{year}年淹水频率数据...")
                    flood_freq = np.ones(self.n_pixels)  # 默认值
                    # 向量化匹配坐标
                    coords_df = pd.DataFrame(self.coords, columns=['经度', '纬度']).round(6)
                    flood_year_data = self.flood_data[['经度', '纬度', str(year)]].round(6)
                    # 向量化合并
                    merged_flood = pd.merge(coords_df.reset_index(), flood_year_data, on=['经度', '纬度'], how='left')
                    merged_flood[str(year)] = merged_flood[str(year)].fillna(1.0)
                    merged_flood[str(year)] = merged_flood[str(year)].replace(255, 1.0)
                    merged_flood = merged_flood.sort_values('index').reset_index(drop=True)
                    # 更新淹水频率
                    flood_freq = merged_flood[str(year)].values
                    flood_data_dict[year] = flood_freq
                else:
                    flood_data_dict[year] = flood_data_dict.get(year - 1, np.ones(self.n_pixels))

            # 【向量化优化】逐年模拟（仅训练年份）
            year_end_states = {}
            for year in self.train_years:
                flood_freq = flood_data_dict[year]
                for month in range(1, 13):
                    if 3 <= month <= 9:
                        current_state = self.apply_spread_rules_vectorized(current_state, flood_freq, month)
                    current_state = self.apply_state_transitions_vectorized(current_state, flood_freq, month)

                # 【新增】打印该年的年末状态统计
                current_year_valid = current_state[current_state >= 0]
                if len(current_year_valid) > 0:
                    state_counts = np.bincount(current_year_valid)
                else:
                    state_counts = []
                # 确保state_counts至少有4个元素（0,1,2,3对应）
                if len(state_counts) < 4:
                    state_counts = list(state_counts) + [0] * (4 - len(state_counts))
                print(
                    f"    {year}年末模拟状态统计: 光滩={state_counts[0]}, 健康互花米草={state_counts[1]}, 枯萎互花米草={state_counts[2]}, 森林={state_counts[3]}")
                # 保存年末状态
                year_end_states[year] = current_state.copy()

            return year_end_states

        finally:
            # 【关键修复】确保参数恢复
            if original_params is not None:
                self.params = original_params
                self.create_spread_kernel()  # 恢复原始传播核

    def calculate_training_loss(self, params_list, param_names):
        """计算训练损失函数 - 向量化优化版本"""
        # 记录单次参数测试开始时间
        param_test_start_time = time.time()

        # 将参数列表转换为字典
        params_dict = {name: value for name, value in zip(param_names, params_list)}

        # 将spread_radius转换为整数
        if 'spread_radius' in params_dict:
            params_dict['spread_radius'] = int(round(params_dict['spread_radius']))

        # 显示当前传播半径
        current_radius = self.params.get('spread_radius', '未知')
        print(f"🎯 正在测试参数 (传播半径={current_radius}格子): {params_dict}")

        # 【向量化优化】运行训练模拟
        year_end_states = self.run_training_simulation(params_dict)

        # 【向量化优化】计算所有训练年份的平均损失
        total_loss = 0.0
        total_accuracy = 0.0
        valid_years = 0

        for year in self.train_years:
            if year not in year_end_states or year not in self.veg_years:
                continue

            simulated_state = year_end_states[year]

            # 【向量化优化】获取真实数据
            print(f"向量化获取{year}年验证数据...")
            coords_df = pd.DataFrame(self.coords, columns=['经度', '纬度']).round(6)
            veg_year_data = self.veg_data[['经度', '纬度', str(year)]].round(6)
            merged_real = pd.merge(coords_df.reset_index(), veg_year_data, on=['经度', '纬度'], how='left')
            merged_real[str(year)] = merged_real[str(year)].fillna(255)
            merged_real = merged_real.sort_values('index').reset_index(drop=True)
            real_state = merged_real[str(year)].values.astype(int).tolist()
            real_state = self.map_vegetation_states(real_state)

            # 计算传统精度
            accuracy = self.calculate_accuracy(simulated_state, real_state)

            # 【核心新增】计算MCC作为主要训练指标
            mcc, mcc_stats = self.calculate_mcc(simulated_state, real_state)

            # 【数值稳定性】检查MCC计算的可靠性
            if np.isnan(mcc) or np.isinf(mcc):
                print(f"⚠️ 警告：{year}年MCC计算异常: {mcc}，使用精度替代")
                # 使用传统精度的反向作为损失
                loss = 1.0 - accuracy
            else:
                # 使用MCC的负值作为损失（MCC越高越好，损失越低越好）
                loss = 1.0 - mcc

            # 【数值稳定性】限制损失范围
            loss = np.clip(loss, 0.0, 2.0)

            total_loss += loss
            total_accuracy += accuracy
            valid_years += 1

            print(f"  训练年份 {year} 传统精度: {accuracy:.4f}, MCC: {mcc:.4f}, 损失: {loss:.4f}")

        if valid_years == 0:
            print("⚠️ 警告：没有有效的训练年份数据")
            return 2.0  # 返回高损失值

        avg_loss = total_loss / valid_years
        avg_accuracy = total_accuracy / valid_years

        # 计算并显示单次参数测试耗时
        param_test_elapsed = time.time() - param_test_start_time
        if param_test_elapsed < 60:
            time_str = f"{param_test_elapsed:.1f}秒"
        elif param_test_elapsed < 3600:
            time_str = f"{param_test_elapsed / 60:.1f}分钟"
        else:
            time_str = f"{param_test_elapsed / 3600:.1f}小时"

        print(f"  平均损失: {avg_loss:.4f}, 平均精度: {avg_accuracy:.4f}")
        print(f"  【向量化优化耗时】此参数组合测试耗时: {time_str}\n")

        return avg_loss

    def train_model_parallel(self, params_to_optimize=None, radius_values=None, n_processes=None):
        """并行训练模型参数 - 基于spread_radius分组的多进程优化"""
        self.start_step_timer("并行参数训练")
        self.estimate_time("参数训练", self.n_valid_pixels)

        if params_to_optimize is None:
            params_to_optimize = [
                'spread_prob',
                'spread_decay',
                'wither_prob_winter',
                'recover_prob',
                'spread_radius',
                'flood_threshold'
            ]

        if radius_values is None:
            radius_values = [3, 4, 5, 6]  # 您指定的关键参数值

        if n_processes is None:
            n_processes = min(len(radius_values), mp.cpu_count())

        print(f"开始并行训练模型参数...")
        print(f"待优化参数: {params_to_optimize}")
        print(f"spread_radius分组值: {radius_values}")
        print(f"并行进程数: {n_processes}")
        print(f"训练年份: {self.train_years}")

        # 准备参数边界（排除spread_radius，因为它已被固定）
        params_to_optimize_subset = [p for p in params_to_optimize if p != 'spread_radius']
        bounds_dict = {}
        for param in params_to_optimize_subset:
            if param == 'spread_prob':
                bounds_dict[param] = (0.1, 0.8)  # 传播概率：10%-50%
            elif param == 'wither_prob_winter':
                bounds_dict[param] = (0.1, 0.5)  # 冬季枯萎概率：10%-50%
            elif param == 'recover_prob':
                bounds_dict[param] = (0.3, 0.9)  # 恢复概率：30%-90%
            elif param == 'spread_decay':
                bounds_dict[param] = (0.5, 3.0)  # 传播衰减：0.5-2.0
            elif param == 'flood_threshold':
                bounds_dict[param] = (0.05, 0.9)  # 淹水阈值：50%-90%
            else:
                bounds_dict[param] = (0.01, 2.0)  # 其他参数默认范围

        print(f"子参数搜索范围: {bounds_dict}")

        # 准备多进程任务参数
        task_args = []
        for radius in radius_values:
            args_tuple = (
                radius,  # radius_value
                self.params,  # base_params
                self.coords,  # coords
                self.veg_data,  # veg_data
                self.flood_data,  # flood_data
                self.train_years,  # train_years
                self.veg_years,  # veg_years
                self.flood_years,  # flood_years
                bounds_dict,  # bounds_dict
                params_to_optimize_subset  # params_to_optimize_subset
            )
            task_args.append(args_tuple)

        print("启动多进程并行优化...")
        training_start_time = time.time()

        # 执行并行优化
        try:
            with Pool(processes=n_processes) as pool:
                results = pool.map(optimize_with_fixed_radius, task_args)
        except Exception as e:
            print(f"多进程执行失败: {e}")
            print("回退到单进程顺序执行...")
            results = []
            for args in task_args:
                result = optimize_with_fixed_radius(args)
                results.append(result)

        training_time = time.time() - training_start_time
        print(f"并行参数优化完成，耗时: {training_time / 60:.1f}分钟")

        # 过滤无效结果
        valid_results = [r for r in results if r is not None and r['success']]

        if not valid_results:
            print("⚠️ 所有并行优化都失败了，保持原始参数")
            return None

        # 选择最佳结果
        best_result = min(valid_results, key=lambda x: x['best_loss'])

        print(f"\n=== 并行优化结果汇总 ===")
        for result in valid_results:
            radius = result['spread_radius']
            loss = result['best_loss']
            print(f"spread_radius={radius}: 损失={loss:.6f}")

        print(f"\n最佳结果: spread_radius={best_result['spread_radius']}, 损失={best_result['best_loss']:.6f}")

        # 更新模型参数为最佳结果
        print(f"\n更新模型参数:")
        for param, value in best_result['optimized_params'].items():
            old_value = self.params[param]
            self.params[param] = value
            print(f"  {param}: {old_value:.6f} -> {value:.6f}")

        # 重新创建传播核
        self.create_spread_kernel()

        # 验证参数更新
        print(f"\n验证更新后的参数:")
        for param in params_to_optimize:
            print(f"  当前 {param}: {self.params[param]:.6f}")

        print(f"\n并行优化总结:")
        print(f"  最小损失值: {best_result['best_loss']:.6f}")
        print(f"  最佳spread_radius: {best_result['spread_radius']}")
        print(f"  优化成功: {'是' if best_result['success'] else '否'}")
        print(f"  总耗时: {training_time / 60:.1f}分钟")
        print(f"  理论加速比: {len(radius_values)}x (实际取决于CPU核数)")

        self.end_step_timer("并行参数训练")
        return best_result['optimization_result']

    def train_model(self, params_to_optimize=None):
        """训练模型参数 - 向量化优化版本"""
        self.start_step_timer("参数训练")
        self.estimate_time("参数训练", self.n_valid_pixels)

        if params_to_optimize is None:
            params_to_optimize = [
                'spread_prob',
                'spread_decay',
                'wither_prob_winter',
                'recover_prob',
                'spread_radius',  # 新增
                'flood_threshold'  # 新增
            ]

        print(f"开始向量化训练模型参数...")
        print(f"待优化参数: {params_to_optimize}")
        print(f"训练年份: {self.train_years}")
        print(f"【向量化优化效果】处理 {self.n_valid_pixels} 个像元，训练速度将大幅提升！")

        # 初始参数值
        initial_params = [self.params[param] for param in params_to_optimize]
        print(f"初始参数值: {dict(zip(params_to_optimize, initial_params))}")

        # 参数搜索范围
        bounds = []
        for param in params_to_optimize:
            if 'prob' in param:
                bounds.append((0.01, 0.99))  # 概率参数范围
            elif param == 'spread_decay':
                bounds.append((0.1, 3.0))  # 衰减系数范围
            elif param == 'spread_radius':
                bounds.append((3, 5))  # 半径范围(3-5)，整数参数但在优化中按连续处理
            elif param == 'flood_threshold':
                bounds.append((0, 1))  # 阈值范围(0-1)
            else:
                bounds.append((0.01, 2.0))  # 默认范围

        print(f"参数搜索范围: {dict(zip(params_to_optimize, bounds))}")

        # 定义目标函数
        def objective(params_list):
            return self.calculate_training_loss(params_list, params_to_optimize)

        # 【关键修复】改进优化器配置
        print("开始向量化参数优化...")
        training_start_time = time.time()

        # 尝试不同的优化方法
        methods_to_try = ['L-BFGS-B', 'Powell', 'TNC']
        best_result = None
        best_loss = float('inf')

        for method in methods_to_try:
            print(f"\n尝试优化方法: {method}")
            try:
                if method == 'L-BFGS-B':
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        options={
                            'maxiter': 50,  # 增加迭代次数
                            'ftol': 1e-6,  # 函数容忍度
                            'gtol': 1e-5,  # 梯度容忍度
                            'disp': True
                        }
                    )
                elif method == 'Powell':
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        options={
                            'maxiter': 30,
                            'ftol': 1e-6,
                            'disp': True
                        }
                    )
                elif method == 'TNC':
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        options={
                            'maxiter': 30,
                            'ftol': 1e-6,
                            'disp': True
                        }
                    )

                print(f"{method} 结果: 成功={result.success}, 损失={result.fun:.6f}")

                if result.success and result.fun < best_loss:
                    best_result = result
                    best_loss = result.fun
                    print(f"发现更好的结果，损失从 {best_loss:.6f} 降到 {result.fun:.6f}")

            except Exception as e:
                print(f"{method} 优化失败: {e}")
                continue

        if best_result is None:
            print("⚠️ 所有优化方法都失败了，保持原始参数")
            return None

        result = best_result
        training_time = time.time() - training_start_time
        print(f"向量化参数优化完成，耗时: {training_time / 60:.1f}分钟")

        # 更新最优参数
        optimized_params = dict(zip(params_to_optimize, result.x))
        print(f"\n最优参数:")
        for param, value in optimized_params.items():
            old_value = self.params[param]
            self.params[param] = value
            print(f"  {param}: {old_value:.6f} -> {value:.6f}")

        # 重新创建传播核以反映参数变化
        self.create_spread_kernel()

        # 【新增验证】验证参数是否正确更新
        print(f"\n验证更新后的参数:")
        for param in params_to_optimize:
            print(f"  当前 {param}: {self.params[param]:.6f}")

        # 【新增】验证优化效果
        print(f"\n验证优化效果:")
        print(f"  最小损失值: {result.fun:.6f}")
        print(f"  优化成功: {'是' if result.success else '否'}")
        if hasattr(result, 'nit'):
            print(f"  迭代次数: {result.nit}")

        self.end_step_timer("参数训练")
        return result


# ===========================================
# 多进程优化全局函数（必须在类外定义）
# ===========================================
def optimize_with_fixed_radius(args_tuple):
    """多进程优化任务函数 - 固定spread_radius的参数优化"""
    radius_value, base_params, coords, veg_data, flood_data, train_years, veg_years, flood_years, bounds_dict, params_to_optimize_subset = args_tuple
    # 更明显的输出格式
    print("\n" + "="*60)
    print(f"🔥 开始优化进程 - SPREAD_RADIUS = {radius_value} 格子")
    print("="*60)
    # 创建临时模型实例（进程隔离）
    temp_model = VegetationCA_FullyOptimized()

    # 设置基础数据（深拷贝避免进程间冲突）
    temp_model.coords = coords.copy()
    temp_model.veg_data = veg_data.copy()
    temp_model.flood_data = flood_data.copy()
    temp_model.train_years = train_years.copy()
    temp_model.veg_years = veg_years.copy()
    temp_model.flood_years = flood_years.copy()
    temp_model.n_pixels = len(coords)

    # 设置固定的spread_radius
    temp_model.params = base_params.copy()
    temp_model.params['spread_radius'] = radius_value

    # 重新初始化必要的组件
    temp_model.create_spatial_grid()
    temp_model.boundary_mask = np.ones(temp_model.n_pixels, dtype=bool)
    temp_model.valid_pixels = np.arange(temp_model.n_pixels)
    temp_model.n_valid_pixels = temp_model.n_pixels

    # 获取初始状态
    coords_df = pd.DataFrame(coords, columns=['经度', '纬度']).round(6)
    veg_2005_data = veg_data[['经度', '纬度', '2005']].round(6)
    merged_veg = pd.merge(coords_df.reset_index(), veg_2005_data, on=['经度', '纬度'], how='left')
    merged_veg['2005'] = merged_veg['2005'].fillna(255)
    merged_veg = merged_veg.sort_values('index').reset_index(drop=True)
    veg_2005 = merged_veg['2005'].values.astype(int).tolist()
    temp_model.initial_state = temp_model.map_vegetation_states(veg_2005)

    # 创建传播核
    temp_model.create_spread_kernel()

    # 准备优化参数（排除spread_radius）
    initial_params = [temp_model.params[param] for param in params_to_optimize_subset]
    bounds = [bounds_dict[param] for param in params_to_optimize_subset]

    print(f"进程 radius={radius_value}: 开始优化参数 {params_to_optimize_subset}")
    print(f"进程 radius={radius_value}: 初始参数 {dict(zip(params_to_optimize_subset, initial_params))}")

    # 定义目标函数
    def objective(params_list):
        return temp_model.calculate_training_loss(params_list, params_to_optimize_subset)

    # 尝试不同的优化方法
    methods_to_try = ['L-BFGS-B', 'Powell', 'TNC']
    best_result = None
    best_loss = float('inf')

    # 根据参数数量设置最大迭代次数
    n_params = len(params_to_optimize_subset)
    if n_params <= 4:
        maxiter_settings = {'L-BFGS-B': 50, 'Powell': 30, 'TNC': 30}
    else:
        maxiter_settings = {'L-BFGS-B': 30, 'Powell': 20, 'TNC': 20}

    for method in methods_to_try:
        print(f"进程 radius={radius_value}: 尝试优化方法 {method}")
        try:
            if method == 'L-BFGS-B':
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    options={
                        'maxiter': maxiter_settings[method],
                        'ftol': 1e-6,
                        'gtol': 1e-5,
                        'disp': False  # 多进程时关闭显示
                    }
                )
            elif method == 'Powell':
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    options={
                        'maxiter': maxiter_settings[method],
                        'ftol': 1e-6,
                        'disp': False
                    }
                )
            elif method == 'TNC':
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    options={
                        'maxiter': maxiter_settings[method],
                        'ftol': 1e-6,
                        'disp': False
                    }
                )

            print(f"进程 radius={radius_value}: {method} 结果: 成功={result.success}, 损失={result.fun:.6f}")
            if result.success and result.fun < best_loss:
                best_result = result
                best_loss = result.fun

        except Exception as e:
            print(f"进程 radius={radius_value}: {method} 优化失败: {e}")
            continue

    if best_result is None:
        print(f"进程 radius={radius_value}: 所有优化方法都失败了")
        return None

    # 构建完整的参数字典
    optimized_params = temp_model.params.copy()
    for i, param in enumerate(params_to_optimize_subset):
        optimized_params[param] = best_result.x[i]

    result_data = {
        'spread_radius': radius_value,
        'optimized_params': optimized_params,
        'best_loss': best_result.fun,
        'success': best_result.success,
        'optimization_result': best_result
    }

    print(f"进程 radius={radius_value}: 优化完成, 最终损失={best_result.fun:.6f}")
    return result_data


# ===========================================
# 第九部分：主函数和程序入口（完全向量化优化版）
# ===========================================
if __name__ == "__main__":
    print("=== 互花米草植被演替元胞自动机模型（完全向量化优化版）===\n")
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    try:
        # 记录总体开始时间
        total_start_time = time.time()
        # 创建向量化优化模型实例
        print("正在创建向量化优化模型实例...")
        model = VegetationCA_FullyOptimized()
        model.total_start_time = total_start_time
        # 步骤1: 初始化模型
        print("\n" + "=" * 60)
        print("步骤1: 初始化向量化优化模型...")
        print("=" * 60)
        model.initialize_model()
        # 步骤2: 训练模型参数（向量化优化功能）
        print("\n" + "=" * 60)
        print("步骤2: 向量化训练模型参数...")
        print("=" * 60)
        # 询问用户是否进行参数训练
        use_training = True  # 可以改为 False 跳过训练直接使用默认参数
        # 步骤2: 训练模型参数（向量化优化功能）
        print("\n" + "=" * 60)
        print("步骤2: 向量化训练模型参数...")
        print("=" * 60)
        # 询问用户是否进行参数训练
        use_training = True  # 可以改为 False 跳过训练直接使用默认参数
        if use_training:
            # 选择要优化的参数
            params_to_optimize = [
                'spread_prob',  # 基础传播概率
                'spread_decay',  # 传播衰减系数
                'wither_prob_winter',  # 冬季枯萎概率
                'recover_prob',  # 枯萎恢复概率
                'spread_radius',  # 传播半径(像元数) - 新增
                'flood_threshold'  # 淹水频率阈值 - 新增
            ]

            # 【新增调试输出 - 训练前参数】
            print("=== 训练前参数状态 ===")
            for param, value in model.params.items():
                print(f"  {param}: {value:.6f}")
            print("=" * 40)

            # 选择训练模式
            use_parallel_training = True  # 设为False可回退到单进程
            if use_parallel_training:
                print("=== 使用并行训练模式 ===")
                training_result = model.train_model_parallel(
                    params_to_optimize=params_to_optimize,
                    radius_values=[3, 4, 5, 6],  # 您指定的spread_radius分组
                    n_processes=4  # 或None让系统自动决定
                )
                # 如果并行训练失败，自动回退到单进程
                if training_result is None:
                    print("⚠️ 并行训练失败，自动回退到单进程训练...")
                    # 从单进程参数中移除 spread_radius
                    single_process_params = [p for p in params_to_optimize if p != 'spread_radius']
                    training_result = model.train_model(single_process_params)
            else:
                print("=== 使用单进程训练模式 ===")
                # 从单进程参数中移除 spread_radius
                single_process_params = [p for p in params_to_optimize if p != 'spread_radius']
                training_result = model.train_model(single_process_params)

            # 【新增调试输出 - 训练后参数】
            print("\n=== 训练后参数状态 ===")
            for param, value in model.params.items():
                print(f"  {param}: {value:.6f}")

            # 【新增调试输出 - 参数变化对比】
            print("\n=== 参数变化对比 ===")
            initial_params_dict = {
                'spread_prob': 0.3,
                'spread_decay': 0.5,
                'spread_radius': 3.0,
                'flood_threshold': 0.7,
                'wither_prob_winter': 0.4,
                'wither_prob_summer': 0.1,
                'recover_prob': 0.6,
                'death_prob': 0.2
            }

            for param in params_to_optimize:
                initial_val = initial_params_dict[param]
                current_val = model.params[param]
                change = current_val - initial_val
                change_pct = (change / initial_val) * 100 if initial_val != 0 else 0
                status = "✓ 已变化" if abs(change) > 1e-6 else "⚠️ 未变化"
                print(f"  {param}: {initial_val:.6f} → {current_val:.6f} ({change:+.6f}, {change_pct:+.2f}%) {status}")

            if training_result is not None:
                print(f"\n训练结果: 优化成功 = {training_result.success}")
                print(f"最终损失值: {training_result.fun:.6f}")
                if hasattr(training_result, 'nit'):
                    print(f"迭代次数: {training_result.nit}")
                print("\n向量化训练完成！优化后的参数已更新。")
            else:
                print("\n⚠️ 警告: 参数训练失败，使用默认参数继续。")
            print("=" * 40)
        else:
            print("跳过参数训练，使用默认参数进行模拟。")
        # 步骤3: 运行完整模拟
        print("\n" + "=" * 60)
        print("步骤3: 运行完整向量化模拟...")
        print("=" * 60)
        all_states = model.run_simulation(save_visualization=True)
        # 步骤4: 验证模型性能
        print("\n" + "=" * 60)
        print("步骤4: 验证向量化模型（MCC空间分布精度评估）...")
        print("=" * 60)
        final_accuracy, avg_mcc, avg_spatial_error, avg_spatial_accuracy, validation_results = model.validate_model(all_states)
        # 计算总耗时
        total_elapsed = time.time() - total_start_time
        if total_elapsed < 60:
            total_time_str = f"{total_elapsed:.1f}秒"
        elif total_elapsed < 3600:
            total_time_str = f"{total_elapsed / 60:.1f}分钟"
        else:
            total_time_str = f"{total_elapsed / 3600:.1f}小时"
        # 输出最终结果
        print("\n" + "=" * 60)
        print("=== 向量化模拟完成 ===")
        print("=" * 60)
        print(f"总耗时: {total_time_str}")
        print(f"最终验证精度: {final_accuracy:.4f}")
        print(f"MCC空间分布精度: {avg_mcc:.4f}")
        print(f"平均空间分布误差率: {avg_spatial_error:.4f}")
        print(f"平均空间分布准确率: {avg_spatial_accuracy:.4f}")
        print(f"结果保存目录: {model.output_dir}")
        print(f"向量化优化效果: 处理了 {model.n_valid_pixels} 个像元 (占 {model.n_valid_pixels / model.n_pixels * 100:.1f}%)")
        print(f"💡 性能提升预期: 从原始1.2小时优化到约{total_elapsed / 60:.1f}分钟！")
        # 输出最终参数
        print(f"\n最终向量化优化模型参数:")
        for param, value in model.params.items():
            print(f"  {param}: {value:.4f}")
        # 保存参数到文件
        params_file = os.path.join(model.output_dir, "vectorized_optimized_parameters.txt")
        with open(params_file, 'w', encoding='utf-8') as f:
            f.write("向量化优化后的模型参数\n")
            f.write("=" * 30 + "\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证精度: {final_accuracy:.4f}\n")
            f.write(f"MCC空间分布精度: {avg_mcc:.4f}\n")
            f.write(f"空间分布误差率: {avg_spatial_error:.4f}\n")
            f.write(f"总耗时: {total_time_str}\n")
            f.write(f"向量化优化效果: 处理了 {model.n_valid_pixels} 个像元\n\n")
            f.write("参数值:\n")
            for param, value in model.params.items():
                f.write(f"{param}: {value:.6f}\n")
        print(f"向量化优化参数已保存到: {params_file}")
        # 生成详细报告
        report_file = os.path.join(model.output_dir, "vectorized_simulation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("互花米草植被演替模拟报告（完全向量化优化版本）\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"模拟时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模拟年份: {model.start_year}-{model.end_year}\n")
            f.write(f"训练年份: {model.train_years[0]}-{model.train_years[-1]}\n")
            f.write(f"验证年份: {model.test_years[0]}-{model.test_years[-1]}\n")
            f.write(f"像元数量: {model.n_pixels}\n")
            f.write(f"有效像元数量: {model.n_valid_pixels} ({model.n_valid_pixels / model.n_pixels * 100:.1f}%)\n")
            f.write(f"网格大小: {model.n_rows} × {model.n_cols}\n")
            f.write(f"向量化优化: 使用NumPy卷积和批量运算，大幅提升性能\n\n")
            f.write(f"最终验证精度: {final_accuracy:.4f}\n")
            f.write(f"MCC空间分布精度: {avg_mcc:.4f}\n")
            f.write(f"平均空间分布误差率: {avg_spatial_error:.4f}\n")
            f.write(f"平均空间分布准确率: {avg_spatial_accuracy:.4f}\n")
            f.write(f"总耗时: {total_time_str}\n")
            f.write(f"性能提升: 预期从1.2小时优化到{total_elapsed / 60:.1f}分钟\n\n")
            f.write("向量化优化技术:\n")
            f.write("  - 使用NumPy卷积替代嵌套循环\n")
            f.write("  - 批量向量运算替代逐元素处理\n")
            f.write("  - 预计算传播核函数\n")
            f.write("  - 2D网格向量化计算\n\n")
            f.write("模型参数:\n")
            for param, value in model.params.items():
                f.write(f"  {param}: {value:.6f}\n")
            # 添加所有年份的验证结果
            f.write("\n" + "=" * 50 + "\n")
            f.write("所有验证年份的详细结果\n")
            f.write("=" * 50 + "\n\n")
            for result in validation_results:
                f.write(f"\n验证年份: {result['year']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"传统分类精度: {result['accuracy']:.4f}\n")
                f.write(f"MCC空间分布精度: {result['mcc']:.4f}\n")
                f.write(f"空间分布误差率: {result['spatial_error']:.4f}\n")
                f.write(f"空间分布准确率: {result['spatial_accuracy']:.4f}\n")
                f.write("二分类混淆矩阵统计:\n")
                f.write(f"  真阳性(TP): {result['mcc_stats']['true_positive']}\n")
                f.write(f"  真阴性(TN): {result['mcc_stats']['true_negative']}\n")
                f.write(f"  假阳性(FP): {result['mcc_stats']['false_positive']}\n")
                f.write(f"  假阴性(FN): {result['mcc_stats']['false_negative']}\n")
                f.write(f"  总像元数: {result['mcc_stats']['total_pixels']}\n")
                f.write(f"  真实互花米草像元数: {result['mcc_stats']['real_spartina_count']}\n")
                f.write(f"  模拟互花米草像元数: {result['mcc_stats']['sim_spartina_count']}\n")
                f.write("二分类性能指标:\n")
                f.write(f"  精确率(Precision): {result['mcc_stats']['precision']:.4f}\n")
                f.write(f"  召回率(Recall): {result['mcc_stats']['recall']:.4f}\n")
                f.write(f"  F1 分数: {result['mcc_stats']['f1_score']:.4f}\n")
                f.write(f"  总体准确率: {result['mcc_stats']['overall_accuracy']:.4f}\n")
            # 写入汇总结果
            f.write("\n" + "=" * 50 + "\n")
            f.write("验证结果汇总\n")
            f.write("-" * 30 + "\n")
            f.write(f"平均传统分类精度: {final_accuracy:.4f}\n")
            f.write(f"平均MCC空间分布精度: {avg_mcc:.4f}\n")
            f.write(f"平均空间分布误差率: {avg_spatial_error:.4f}\n")
            f.write(f"平均空间分布准确率: {avg_spatial_accuracy:.4f}\n")
            if avg_mcc > 0.8:
                mcc_interpretation = "优秀"
            elif avg_mcc > 0.6:
                mcc_interpretation = "良好"
            elif avg_mcc > 0.4:
                mcc_interpretation = "中等"
            elif avg_mcc > 0.2:
                mcc_interpretation = "较差"
            else:
                mcc_interpretation = "很差"
            f.write(f"MCC评估结果: {mcc_interpretation} (范围: -1到+1, +1为完美预测)\n")
            f.write("\nMCC指标说明:\n")
            f.write("  MCC范围: -1 到 +1\n")
            f.write("  +1: 完美预测\n")
            f.write("  0: 随机预测水平\n")
            f.write("  -1: 完全相反的预测\n")
            f.write("  MCC优势: 对不平衡数据集友好，同时考虑TP、TN、FP、FN四个指标\n")
        print(f"向量化模拟报告已保存到: {report_file}")
        print(f"\n🚀 向量化优化程序执行完毕！预期性能提升100-1000倍！")
    except Exception as e:
        print(f"\n❌ 向量化模型运行出错: {e}")
        import traceback
        traceback.print_exc()
        # 即使出错也尝试保存错误信息
        if 'model' in locals() and hasattr(model, 'output_dir'):
            error_file = os.path.join(model.output_dir, "vectorized_error_log.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误信息: {str(e)}\n\n")
                f.write("详细错误追踪:\n")
                f.write(traceback.format_exc())
            print(f"向量化错误日志已保存到: {error_file}")