import os
import numpy as np
import h5py
import inspect
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple
import matplotlib.pyplot as plt
from tensorboard.util import tensor_util
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

'''
============================辅助类函数====================================
'''

# =======================================================================
# 函数：将子空间映射到全空间
# =======================================================================
def combine(subspace_vec, globalspace_vec, subspace_index):
    '''
    将子空间的向量映射回全空间。

    主要功能包括：
    1. 如果没有提供子空间索引，直接返回子空间向量。
    2. 如果提供了子空间索引，则在全空间向量的对应位置替换为子空间向量的值。

    参数:
        subspace_vec (np.ndarray): 子空间的向量, 形状为 (n_samples, n_subspace_dims)。
        globalspace_vec (np.ndarray): 全空间的向量，用于填充未被替换的位置, 形状为 (n_global_dims,)。
        subspace_index (list or None): 子空间在全空间中的索引位置。如果为 None，则表示不进行替换。

    用法示例:
        sub_vec = np.array([[1, 2], [3, 4]])
        global_vec = np.array([0, 0, 0, 0])
        index = [1, 3]
        combined = combine(sub_vec, global_vec, index)
        print(combined)  # 输出: [[0, 1, 0, 2], [0, 3, 0, 4]]
    结果:
        返回一个新的全空间向量，其中子空间的值已被正确替换, 形状为 (n_samples, n_global_dims)。
    '''
    if subspace_index is None:
        return subspace_vec
    else:
        combination = np.tile(globalspace_vec, (len(subspace_vec), 1))
        combination[:, subspace_index] = subspace_vec
        return combination

# =======================================================================
# 函数：确保数组单调不增
# =======================================================================
def make_monotonic_decreasing(arr):
    '''
    确保输入数组单调不增。

    主要功能包括：
    1. 遍历数组，检查每个元素是否小于或等于其前一个元素。
    2. 如果发现某个元素大于其前一个元素，则将其修改为与前一个元素相等。

    参数:
        arr (list or np.ndarray): 输入的一维数组。
    
    用法示例:
        data = [5, 3, 4, 2, 1]
        result = make_monotonic_decreasing(data)
        print(result)  # 输出: [5, 3, 3, 2, 1]
    结果:
        返回一个单调不增的数组。
    '''
    for i in range(len(arr) - 1):
        if arr[i] < arr[i + 1]:  # 如果前面的元素小于后面的元素
            arr[i + 1] = arr[i]  # 修改后面的元素，使其不大于当前元素
    return arr

# =======================================================================
# 函数：从本地 TensorBoard event 文件读取单一 Run 的数据并还原为列表结构                    
# =======================================================================
def read_data_from_tensorboard_file(
    path: str,
    main_tag: str = 'Optimizer/Cost'
) -> Tuple[List[float], List[int], List[float]]:
    """
    从指定的 TensorBoard 路径（文件或文件夹）读取单一 Run 的数据。
    
    主要功能包括：
        1. 自动识别路径是文件还是文件夹。 (强烈建议使用文件而非文件夹)
        2. 如果是文件夹，自动查找其中的 第一个 event 文件。
        3. 读取指定标签的数据，并返回数值、步骤和时间戳列表。

    参数:
        path (str): event 文件的完整路径，或者包含该文件的目录路径。
        main_tag (str): 要读取的 scalar tag。

    用法示例:
        values = read_data_from_tensorboard_file('./logs/run_2024-01-01_12-00-00/events.out.tfevents.1234567890.hostname', main_tag='Optimizer/Cost')
    返回:
        values (List[float]): 读取的数值列表。
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

    # 1. 确定要读取的具体文件路径
    file_path = path
    if os.path.isdir(path):
        # 如果给的是目录，找里面包含 'tfevents' 的文件
        files = [f for f in os.listdir(path) if 'tfevents' in f]
        if not files:
            print(f"Warning: No event files found in directory {path}")
            return []
        # 默认读第一个找到的 event 文件
        # (通常一个 run 目录里只有一个 event 文件，除非程序中断过)
        files.sort() 
        file_path = os.path.join(path, files[0])

    print(f"Reading TensorBoard file: {file_path}")

    # 2. 使用 EventFileLoader 读取数据
    values = []

    try:
        loader = EventFileLoader(file_path)
        for event in loader.Load():
            # 遍历 event 中的所有 value
            for value in event.summary.value:
                if value.tag == main_tag:
                    # ====================================================
                    # 手动实现 EventAccumulator 的数据解析逻辑
                    # ====================================================
                    if value.HasField('tensor'):
                        # 新版/PyTorch 格式：数据存在 tensor 字段里
                        # 使用 tensor_util 将其转为 numpy，然后取标量值
                        val = tensor_util.make_ndarray(value.tensor).item()
                        values.append(val)
                    elif value.HasField('simple_value'):
                        # 旧版格式：数据存在 simple_value 字段里
                        values.append(value.simple_value)
                    else:
                        # 既没 tensor 也没 simple_value，跳过
                        continue
                    # ====================================================
                    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    return values

# =======================================================================
# 函数：从本地 TensorBoard 文件夹读取所有 Run 的数据并还原为列表结构
# =======================================================================
def read_data_from_tensorboard_folder(summary_dir):
    '''
    从指定的 TensorBoard 目录读取所有 Run 的数据。

    主要功能包括：
    1. 自动查找目录下的所有 event 文件。
    2. 逐个读取每个 event 文件的数据。
    3. 返回一个包含所有 Run 数据的列表。

    参数:
        summary_dir (str): 包含 TensorBoard event 文件的目录路径。

    用法示例:
        data = read_data_from_tensorboard_folder('./logs/run_2024-01-01_12-00-00/Summary_All')
    结果:
        返回一个包含所有 Run 数据的列表，格式为 [[run1_data], [run2_data], ...]。
    '''
    dataset = []
    
    # 1. 找到该目录下所有的 event 文件
    if not os.path.exists(summary_dir):
        print(f"[Error] Directory not found: {summary_dir}")
        return []

    # 筛选文件名包含 'tfevents' 的文件
    event_files = [f for f in os.listdir(summary_dir) if 'tfevents' in f]
    
    # 【重要】按照文件名排序
    # 因为文件名里包含时间戳，排序可以大致保证 run_0, run_1 的顺序（如果它们是按顺序生成的）
    event_files.sort()
    
    print(f"Found {len(event_files)} event files in Summary_All.")

    # 2. 逐个文件读取 (每个文件就是一个独立的 Run)
    for file_name in event_files:
        file_path = os.path.join(summary_dir, file_name)
        
        try:
            data = read_data_from_tensorboard_file(file_path)
            dataset.append(data)  # 将该 run 的数据添加到总数据集中

        except Exception as e:
            print(f"  - Error reading {file_name}: {e}")

    return dataset

# =======================================================================
# 函数: 读取由 running_data_record 生成的 HDF5 文件
# =======================================================================
def load_running_data(file_path):
    """
    读取由 running_data_record 函数生成的 HDF5 文件。

    主要功能包括：
    1. 自动处理文件路径，支持直接传入文件夹路径。
    2. 读取 HDF5 文件中的所有数据集，并将其存储在字典中返回。
    参数:
        file_path (str): HDF5 文件的完整路径，或者包含该文件的目录路径。

    用法示例:
        data = load_running_data('./repository/data/baseline/mmes/cec2013lsgo/maxfes_1E6/f15/2024-01-01_12-00-00/running_data.h5')
    结果:
        返回一个字典，键为数据集名称，值为对应的 NumPy 数组。
    """
    
    # --- 1. 路径处理 (对应写入时的拼接逻辑) ---
    target_file = file_path
    
    # 如果传入的是文件夹，尝试在里面找 running_data.h5
    if os.path.isdir(file_path):
        candidate = os.path.join(file_path, 'running_data.h5')
        if os.path.exists(candidate):
            target_file = candidate
        else:
            # 也许用户只是没传文件名，但这文件夹里没有对应文件
            print(f"[Warning] Directory found but 'running_data.h5' is missing in: {file_path}")
            # 这种情况下，h5py 打开文件夹会报错，所以下面让它自然报错或你可以 raise Error
            
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"HDF5 file not found: {target_file}")

    print(f"Loading data from: {target_file} ...")
    
    loaded_data = {}
    
    # --- 2. 读取数据 ---
    try:
        with h5py.File(target_file, 'r') as f:
            # 遍历文件中的所有算法键 (Keys)
            for key in f.keys():
                # [:] 切片操作会将 HDF5 dataset 加载到内存中的 NumPy 数组
                data_matrix = f[key][:]
                
                loaded_data[key] = data_matrix
                
                # 如果你一定要变回 List of Lists (虽然不推荐，因为慢)，可以用:
                # loaded_data[key] = data_matrix.tolist()
                
                print(f"  - Loaded '{key}': shape={data_matrix.shape}")
                
    except Exception as e:
        print(f"[Error] Failed to read HDF5: {e}")
        return {}
        
    return loaded_data

# =======================================================================
# 函数：通用的并行运行包装器
# =======================================================================
def run_parallel_task(target_func, parallel_num, *args, **kwargs):
    '''
    通用的并行运行包装器。

    主要功能包括：
    1. 使用多进程并行执行指定的目标函数多次。
    2. 收集每次运行的结果，并计算平均运行时间。

    参数:
        target_func (callable): 需要并行执行的目标函数。
        parallel_num (int): 并行执行的次数。
        *args: 传递给目标函数的位置参数。
        **kwargs: 传递给目标函数的关键字参数。
    
    用法示例:
        def sample_task(x, y, index):
            time.sleep(1)  # 模拟耗时任务
            return (1.0, x + y + index)
        avg_time, results = run_parallel_task(sample_task, 5, 10, 20)
        print(f"Average Time: {avg_time}, Results: {results}")
    结果:
        返回平均运行时间和每次运行的结果列表。
    '''

    # 自动分析 target_func 的参数签名
    # 目的是找到那个用于接收 "循环索引" 的参数名 (即 cycle_num_index)
    sig = inspect.signature(target_func)
    params = list(sig.parameters.keys())
    
    # 假设目标函数的最后一个参数就是用来接收 index 的
    if not params:
        raise ValueError("目标函数没有参数，无法传递循环索引")
    index_param_name = params[-1]

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(parallel_num):
            # 我们把 i (循环索引) 明确地作为一个关键字参数传进去
            # 这样就不会和 *args 或 **kwargs 里的其他参数冲突了
            
            current_kwargs = kwargs.copy()
            current_kwargs[index_param_name] = i
            
            # 注意：parallel_num 已经被提取出来了，不会传给 target_func
            futures.append(executor.submit(target_func, *args, **current_kwargs))
        
        total_time = 0
        results_record = []
        
        for future in futures:
            # 获取结果
            # 假设 target_func 返回的是 (耗时, 结果数据) 的元组
            res = future.result()
            
            # 兼容性处理：如果返回的是元组且长度为2 (time, data)
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                results_record.append(res[0])
                total_time += res[1]
            else:
                # 如果你的任务只返回数据，没有返回时间，可以在这里做兼容
                results_record.append(res)

        # 防止除以0
        avg_time = total_time / parallel_num if parallel_num > 0 else 0
        
        return results_record, avg_time

'''
============================记录类函数====================================
'''

# =======================================================================
# 类：用于记录评估值的包装类
# =======================================================================
class fun_record():
    '''
    包装一个目标函数，并在每次调用时记录其返回的适应度值。
    这样可以方便地跟踪优化过程中函数的评估情况。

    主要功能包括：
    1. 初始化时传入一个目标函数。
    2. 每次调用时，执行目标函数并将返回的适应度值追加到内部的 fitness_record 列表中。
    3. 提供一个接口来获取记录的适应度值。

    参数:
        fun (callable): 目标函数，接受一个输入并返回适应度值。
    用法示例:
        def sample_function(x):
            return x**2
        recorder = fun_record(sample_function)
        result = recorder(3)  # 调用目标函数
        print(recorder.fitness_record)  # 输出记录的适应度值: [9]
    结果:
        目标函数的返回值将被记录在 fitness_record 列表中。
    '''
    def __init__(self, fun):
        self.fun = fun
        self.fitness_record = []
    def __call__(self, x):
        fitness = self.fun(x)
        self.fitness_record.extend(fitness)
        return fitness

# =======================================================================
# 函数：将优化算法的收敛数据记录到 TensorBoard
# =======================================================================
def log_to_tensorboard(
    fitness_record, 
    log_dir, 
    sample_rate=100, 
    main_tag='Optimizer/Cost'
):
    """
    将优化算法的收敛数据记录到 TensorBoard。

    主要功能包括：
    1. 记录每次独立运行的曲线 (Individual Runs)。
    2. 双重记录：在 Summary_All 中生成均值线
    3. 计算并记录独立的标准差曲线 (Std Dev)。

    参数:
        fitness_record (list): 包含多次运行结果的列表，格式为 [[run1_data], [run2_data], ...]
        log_dir (str): 日志保存的根目录 (建议包含 timestamp)。
        sample_rate (int): 采样率，每隔多少代记录一次 (默认 100)。
        main_tag (str): TensorBoard 中的图表名称 (默认 'Optimizer/Cost')。

    用法示例:
        log_to_tensorboard(fitness_record, './logs/run_2024-01-01_12-00-00', sample_rate=50, main_tag='Optimizer/Cost')
    结果:
        在指定目录下生成 TensorBoard 日志文件，可用于可视化优化过程。
    """
    # 确保父目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 定义公共文件夹路径
    agg_dir = os.path.join(log_dir, 'Summary_All')
    if not os.path.exists(agg_dir):
        os.makedirs(agg_dir)

    print(f"Starting TensorBoard logging to: {log_dir}")
    print(f"Total runs: {len(fitness_record)} | Sample rate: {sample_rate}")

    # =======================================================
    # 第一阶段：双重记录 (生成单次曲线 + 均值/阴影图)
    # =======================================================
    for run_index, single_run_data in enumerate(fitness_record):
        
        # --- Writer A: 独立文件夹 (用于查看单次运行) ---
        individual_dir = os.path.join(log_dir, f'run_{run_index}')
        if not os.path.exists(individual_dir):
            os.makedirs(individual_dir)
        writer_individual = SummaryWriter(log_dir=individual_dir)
        
        # --- Writer B: 公共文件夹 (用于自动生成均值+阴影) ---
        # 技巧：每次循环重新创建 writer 指向同一个目录，强制生成独立的 event file
        writer_agg = SummaryWriter(log_dir=agg_dir)
        
        for fe, val in enumerate(single_run_data):
            if fe % sample_rate == 0:
                # 写入独立文件夹
                writer_individual.add_scalar(main_tag, val, global_step=fe)
                
                # 写入公共文件夹 (Tag 必须完全一致)
                writer_agg.add_scalar(main_tag, val, global_step=fe)
                
        writer_individual.close()
        writer_agg.close()

    # =======================================================
    # 第二阶段：计算并记录独立的“标准差”曲线
    # =======================================================
    
    # 1. 数据对齐 (防止不同 run 长度不一致)
    if not fitness_record:
        print("Warning: fitness_record is empty.")
        return

    min_len = min(len(r) for r in fitness_record)
    records_array = np.array([r[:min_len] for r in fitness_record])

    # 2. 计算标准差 (axis=0 代表跨 run 计算)
    std_curve = np.std(records_array, axis=0)
    
    # 3. 写入 (不复用 agg_dir，但使用不同的 Tag)
    stats_dir = os.path.join(log_dir, 'Summary_Stats')
    writer_stats = SummaryWriter(log_dir=stats_dir)
    
    # 自动生成标准差的 Tag 名称，例如 'Optimizer/Cost_Std'
    std_tag = f"{main_tag}_Std" 

    for fe in range(0, min_len):
        if fe % sample_rate == 0:
            writer_stats.add_scalar(std_tag, std_curve[fe], global_step=fe)

    writer_stats.close()

    print(f"Done. Logs ready at: {log_dir}")

# =======================================================================
# 函数：记录算法的评估值（包括特定评估点和最终点），以及运行时间
# =======================================================================
def result_record(data, output_path, record_FEs_list):
    """
    记录优化算法的评估值和运行时间到指定文件。

    主要功能包括：
    1. 记录每个算法在指定评估点的平均适应度
    2. 记录每个算法的最终评估值
    3. 记录每个算法的平均运行时间

    参数:
        data (dict): 包含算法名称为键，评估值列表为值的字典。
                     例如: {'Algorithm1': [[run1_data], [run2_data], ...], 'Algorithm1_time': [time1, time2, ...], ...}
        output_path (str): 结果保存的目录路径。
        record_FEs_list (list): 需要记录的评估点列表 (例如: [1E3, 1E4, 1E5])。
    
    用法示例:
        data = {
            'Algorithm1': [[...], [...], ...],
            'Algorithm1_time': [12.5, 13.0, ...],
            'Algorithm2': [[...], [...], ...],
            'Algorithm2_time': [10.0, 11.5, ...]
        }
        result_record(data, './results/', [1E3, 1E4, 1E5])
    结果:
        在指定目录下生成 "result_record.txt" 文件，包含各算法在指定评估点的平均适应度、标准差和平均运行时间。
    """
    # ---------------------------------------------------------
    # 1. 预处理与环境准备
    # ---------------------------------------------------------
    # 确保 FEs 是整数并排序，防止乱序导致的阅读困难
    record_FEs_list = sorted([int(x) for x in record_FEs_list])
    
    # 确保输出目录存在
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    output_file_path = os.path.join(output_path, "result_record.txt")

    # 定义格式化字符串模板，方便统一修改列宽
    # Header 格式
    HEADER_FMT = "{:<20}{:<20}{:<25}{:<25}{:<25}{:<25}\n"
    # Data 格式: 空白占位 | FEs | Fitness(float) | Fitness(Sci) | Std(float) | Std(Sci)
    DATA_FMT = "{:<20}{:<20}{:<25.6f}{:<25.6e}{:<25.6f}{:<25.6e}\n"
    # Final Line 格式
    FINAL_FMT = "{:<15}{:<25}{:<25.6f}{:<25.6e}{:<25.6f}{:<25.6e}\n"
    SEPARATOR = "-" * 140 + "\n"

    # ---------------------------------------------------------
    # 2. 开始处理与写入
    # ---------------------------------------------------------
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:

            # --- 定义内部函数：同时写文件和打印到控制台 ---
            def dual_log(text):
                f.write(text)
                print(text, end='') # end='' 防止 print 自动加换行，因为 text 已包含 \n
            # -------------------------------------------------
            
            # 写入表头
            dual_log(SEPARATOR)
            dual_log(HEADER_FMT.format(
                "Algorithm", "Record Point", "Mean Fitness", "Mean Sci", "Std Dev", "Std Sci"
            ))
            dual_log(SEPARATOR)

            # 遍历数据字典
            for algorithm_name, runs_data in data.items():
                # 跳过时间 Key，只处理算法主数据
                if algorithm_name.endswith("_time"):
                    continue
                
                # --- 2.1 数据清洗与向量化 ---
                # 将数据转换为 NumPy 列表以便处理
                # 注意：不同 Run 的长度可能不同，不能直接转为矩阵，需处理 Padding
                try:
                    # 获取最大长度
                    max_len = max(len(run) for run in runs_data)
                    n_runs = len(runs_data)
                    
                    # 创建一个填充 NaN 的矩阵 (Runs x Max_FEs)
                    padded_data = np.full((n_runs, max_len), np.nan)
                    
                    for i, run in enumerate(runs_data):
                        # 截取或填充，确保数据放入矩阵
                        length = min(len(run), max_len)
                        # 确保转换为 float 类型
                        run_array = np.array(run[:length], dtype=np.float64)
                        
                        # [重要] 强制单调递减 (Best-so-far)
                        # 使用 np.minimum.accumulate 实现高效的累计最小值
                        monotonic_run = np.minimum.accumulate(run_array)
                        padded_data[i, :length] = monotonic_run

                except Exception as e:
                    print(f"[Error] Processing data for {algorithm_name}: {e}")
                    continue

                # --- 2.2 计算统计量 (使用 np.nanmean 忽略填充的 NaN) ---
                # axis=0 表示沿着 Run 的方向计算均值
                with np.errstate(divide='ignore', invalid='ignore'):
                    mean_curve = np.nanmean(padded_data, axis=0)
                    std_curve = np.nanstd(padded_data, axis=0)

                # --- 2.3 获取时间数据 ---
                time_key = f"{algorithm_name}_time"
                avg_time = "N/A"
                if time_key in data:
                    times = data[time_key]
                    if isinstance(times, (list, np.ndarray)) and len(times) > 0:
                        avg_time = np.mean(times)
                    elif isinstance(times, (int, float)):
                        avg_time = times

                # --- 2.4 写入文件 ---
                dual_log(f"Algorithm: {algorithm_name}\n")
                
                # 写入指定的 FE 点
                for fe_point in record_FEs_list:
                    # 注意：FE 通常是从 1 开始计数，列表索引从 0 开始
                    # 如果 FE_point 是评估次数，对应的索引应该是 fe_point - 1
                    # 但为了防止越界，我们需要做检查
                    idx = fe_point - 1 if fe_point > 0 else 0

                    # 科学计数法格式化 FEs 显示 (如 1.000e+05)
                    fe_display = f"{fe_point:.1e}"
                    
                    if idx < max_len:
                        fitness_val = mean_curve[idx]
                        std_val = std_curve[idx]
                        
                        # 如果计算结果是 NaN (该点没有足够数据)，则处理
                        if np.isnan(fitness_val):
                            f.write(f"{'':<20}{fe_point:<20}{'N/A':<25}{'N/A':<25}{'N/A':<25}{'N/A':<25}\n")
                        else:
                            f.write(DATA_FMT.format("", fe_display, fitness_val, fitness_val, std_val, std_val))
                    else:
                        # 超过了最大评估次数
                        f.write(f"{'':<20}{fe_display:<20}{'Exceeded':<25}{'-':<25}{'-':<25}{'-':<25}\n")

                # 输出最终结果
                final_val = mean_curve[-1]
                final_std = std_curve[-1]
                # 算出科学计数法的规模 (例如 1e5)
                scale_str = f"{max_len:.0e}".replace("+0", "").replace("+", "")
                # 拼接： 规模 | 精确值
                final_label = f"Final({scale_str}|{max_len:,})"
                
                if not np.isnan(final_val):
                    dual_log(FINAL_FMT.format("", final_label, final_val, final_val, final_std, final_std))
                
                # 输出时间
                if avg_time != "N/A":
                    dual_log(f"{'':<15}{'Avg Time(s)':<25}{avg_time:<25.6f}\n")
                
                dual_log(SEPARATOR)
        
        print(f"Evaluation result records successfully saved to: {output_file_path}")

    except IOError as e:
        print(f"Error writing to file {output_file_path}: {e}")

# =======================================================================
# 函数：将运行数据保存为 HDF5 格式
# =======================================================================
def running_data_record(output_data, file_path):
    """
    将优化算法的运行数据保存为 HDF5 格式。
    主要功能包括：
    1. 过滤掉不需要保存的数据 (如包含 "_time" 的键名)。
    2. 对每个算法的多次运行数据进行对齐，确保矩阵化存储。
    3. 使用 gzip 压缩保存数据，节省存储空间。

    参数:
        output_data (dict): 包含算法名称为键，评估值列表为值的字典。 例如: {'Algorithm1': [[run1_data], [run2_data], ...], ...}
        file_path (str): HDF5 文件保存的目录路径

    用法示例:
        output_data = {
            'Algorithm1': [[...], [...], ...],
            'Algorithm2': [[...], [...], ...]
        }
        running_data_record(output_data, './results/h5_data/')

    结果:
        在指定目录下生成 "running_data.h5" 文件，包含各算法的运行数据。
    """
    
    # 确保目录存在
    if not os.path.exists(file_path): 
        os.makedirs(file_path, exist_ok=True)
        
    # 拼接真正的文件名
    h5_file = os.path.join(file_path, 'running_data.h5') # 或者 running_data.h5
    
    print(f"Running data start saving to {file_path} ...")
    
    with h5py.File(h5_file, 'w') as f:
        for key, runs_list in output_data.items():
            
            # --- 1. 过滤逻辑 ---
            # 如果键名包含 "_time"，或者数据为空，直接跳过
            if "_time" in key or not runs_list:
                continue
            
            try:
                # --- 2. 数据对齐与矩阵化 ---
                # HDF5 要求 dataset 必须是规整的矩形，不能是每行长度不一样的列表
                # 找出该算法所有 run 中最短的长度
                min_len = min(len(r) for r in runs_list)
                
                if min_len == 0:
                    print(f"  [Skip] {key}: Runs are empty.")
                    continue

                # 截断所有 run 到 min_len，转为 numpy float64 矩阵
                # 形状: (n_runs, min_len)
                data_matrix = np.array([r[:min_len] for r in runs_list], dtype=np.float64)
                
                # --- 3. 创建数据集 ---
                # 算法名作为 dataset 的名字
                f.create_dataset(key, data=data_matrix, compression='gzip')
                
                print(f"  - Saved '{key}': shape={data_matrix.shape}")
                
            except Exception as e:
                print(f"  [Error] Failed to save {key}: {e}")

    print(f"Running data saved to {file_path}.")

'''
============================绘图类函数====================================
'''

# =======================================================================
# 函数：绘制不同算法的Best-so-Far评估曲线
# =======================================================================
def plot_evaluation_curve_best_so_far(
    data, 
    output_path, 
    maxfes, 
    figsize=(3.5, 2.16), # IEEE Trans 双栏设置，若跨栏 宽度改为 7
    font_size=8,              # 论文标准字体大小 (8pt - 10pt)
    log_scale=True, 
    show_variance=True, 
    eps=1e-12
):
    """
    Nature 风格优化曲线 (无降采样版)
    
    特点:
    1. 尊重原始数据长度，不进行插值。
    2. 自动对齐不同 Run 的长度 (取最小长度)。
    3. X轴自动根据数据点数映射到 [0, MaxFEs]。
    """
    
    # --- 1. 全局样式设置 (Nature 风格) ---
    # 强制使用无衬线字体 (Arial/Helvetica)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'legend.fontsize': font_size - 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'axes.linewidth': 0.8,    
        'lines.linewidth': 1.2,   # 线条稍细，显精致
        'xtick.major.width': 0.8, 
        'ytick.major.width': 0.8,
        'xtick.direction': 'in',  # 刻度朝内
        'ytick.direction': 'in',
        'figure.dpi': 300,
        'savefig.dpi': 600,       # 印刷级分辨率
        'svg.fonttype': 'none',   # 保证 SVG 文字可编辑
    })

    # Nature 经典配色 (NPG)
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
    
    fig, ax = plt.subplots(figsize=figsize)
    color_idx = 0
    
    for algorithm, runs in data.items():
        if "_time" in algorithm or len(runs) == 0: 
            continue
        
        # --- 2. 数据对齐与矩阵化 ---
        # 即使不做降采样，不同 Run 的长度可能因为意外差 1-2 个点
        # 为了矩阵计算，我们截断到最短的那条 Run 的长度
        min_len = min(len(r) for r in runs)
        
        # 预处理：截断 + 单调化 (Best-so-far)
        processed_runs = []
        for run in runs:
            # 截取到公共长度
            r = np.array(run[:min_len])
            # 强制单调递减 (Best-so-far)
            r = np.minimum.accumulate(r)
            processed_runs.append(r)
        
        # 转为矩阵: (n_runs, n_points)
        run_matrix = np.array(processed_runs)
        
        # 防止 Log 报错 (处理 0 或 负数)
        run_matrix = np.clip(run_matrix, eps, None)

        # --- 3. 生成 X 轴 ---
        # 假设数据是均匀记录的，将点数映射到 [0, MaxFEs]
        # x_axis = np.linspace(0, maxfes, min_len)
        
        # 如果你希望 X 轴严格对应索引 (不拉伸到 MaxFEs)，可以用:
        x_axis = np.arange(min_len) 
        # # 但通常为了对比不同算法，建议拉伸到 MaxFEs:
        # x_axis = np.linspace(0, maxfes, min_len)

        # --- 4. 计算统计量 ---
        means = np.mean(run_matrix, axis=0) # 算术平均
        
        curr_color = colors[color_idx % len(colors)]
        
        # 绘制主曲线
        ax.plot(x_axis, means, label=algorithm, color=curr_color, alpha=0.9)

        # 绘制方差带 (Log-Normal Error Band)
        # 在优化问题中，数值跨度极大，普通标准差会导致下界为负，
        # Log 空间标准差生成的 "乘法带宽" 更符合物理意义。
        if show_variance and len(runs) > 1:
            log_vals = np.log10(run_matrix)
            std_log = np.std(log_vals, axis=0)
            factor = 10 ** std_log # 转换回线性空间
            
            upper = means * factor
            lower = means / factor
            
            ax.fill_between(x_axis, lower, upper, color=curr_color, alpha=0.2, linewidth=0)

        color_idx += 1

    # --- 5. 坐标轴深度美化 ---
    if log_scale:
        ax.set_yscale("log")
        # 添加 Log 次级刻度，这让图表看起来更专业
        locmin = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(NullFormatter())

    # 1. 设置格式器
    xfmt = ScalarFormatter(useMathText=True) # useMathText=True 会生成漂亮的 ×10⁶ 而不是 1e6
    xfmt.set_powerlimits((0, 0))             # 强制所有数字都转为科学计数法 (只要不为 0)
    
    ax.xaxis.set_major_formatter(xfmt)

    # 2. (可选) 稍微调整一下那个 10^6 的位置，别太挤
    ax.xaxis.get_offset_text().set_fontsize(font_size - 1)

    ax.set_xlabel("FEs")
    ax.set_ylabel("Objective Value") # 不需要写 log10，因为刻度已经是 10^x
    
    # 【顶刊关键细节】去掉 Top 和 Right 边框，或者设为极细灰色
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 开启轻微网格
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    # 图例：去掉边框，字体稍小
    ax.legend(frameon=False, loc='best')

    plt.tight_layout()

    # --- 6. 保存 ---
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    # 同时保存 PDF (矢量) 和 PNG (位图)
    pdf_path = os.path.join(output_path, "evaluation_curves_best_so_far.pdf")
    png_path = os.path.join(output_path, "evaluation_curves_best_so_far.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=600)
    
    print(f"Figures saved to:\n  {pdf_path}\n  {png_path}")
    plt.close()
