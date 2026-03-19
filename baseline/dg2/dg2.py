import numpy as np
from tqdm import tqdm
from .bfs import BFS

class Delta:
    def __init__(self):
        self.delta1 = None
        self.delta2 = None

class Evaluations:
    def __init__(self):
        self.base = 0
        self.F = None
        self.fhat = None

class DG2:
    def __init__(self, fun, info):
        """
        :param fun: 目标函数 (callable)
        :param info: 字典, 包含 'dimension', 'lower', 'upper'
        """
        self.fun = fun
        self.info = info

    def run(self):
        # 1. 识别结构 (Interaction Structure Identification)
        delta, evaluations, lambda_ = self.ism()
        # 2. 生成分解矩阵 (Decomposition Structure Matrix)
        theta = self.dsm(evaluations, lambda_)
        subspaces = BFS(theta)
        return subspaces

    # ISM: Interaction Structure Matrix (加速优化版)
    def ism(self):
        dim = self.info['dimension']
        lower = self.info['lower']
        upper = self.info['upper']
        
        # --- 初始化存储容器 ---
        delta = Delta()
        evaluations = Evaluations()
        
        # 预分配矩阵，避免动态扩展带来的开销
        f_archive = np.full((dim, dim), np.nan)    # 记录双变量扰动值
        delta1 = np.full((dim, dim), np.nan)       # 记录差分1
        delta2 = np.full((dim, dim), np.nan)       # 记录差分2
        lambda_ = np.full((dim, dim), np.nan)      # 交互强度矩阵
        
        # --- 基础点计算 ---
        # p1 在论文中通常是 lower bound 或 zeros，这里沿用你的逻辑
        p_base = np.ones(dim) * lower
        f_base = self.fun(p_base)
        
        # 计算中心点偏移值 (Perturbation value)
        center_vec = 0.5 * (lower + upper) * np.ones(dim)

        # 创建一个复用的临时向量，避免在循环中反复 copy
        p_temp = np.copy(p_base) 

        # =========================================================
        # 第一步：预计算所有单变量扰动 (O(D))
        # 这一步计算 f(base + ei)
        # =========================================================
        f_single = np.zeros(dim) # 对应原代码的 fhat_archive
        
        for i in range(dim):
            original_val = p_temp[i]       # 备份
            p_temp[i] = center_vec[i]      # 扰动第 i 维
            f_single[i] = self.fun(p_temp) # 评估
            p_temp[i] = original_val       # 还原 (回溯)
            
        # =========================================================
        # 第二步：计算双变量交互 (O(D^2))
        # 这一步计算 f(base + ei + ej)
        # =========================================================
        
        # 预先计算好所有的 d1，因为 d1 = f(base+ei) - f(base)，只跟 i 有关
        # 这样在 j 循环中就不用重复计算了
        d1_arr = f_single - f_base
        
        # 使用 tqdm 显示进度
        for i in tqdm(range(dim - 1), desc="DG2 Structure Search"):
            # --- 扰动第 i 维 ---
            orig_i = p_temp[i]
            p_temp[i] = center_vec[i]
            
            d1 = d1_arr[i] # 查表获取 d1
            
            for j in range(i + 1, dim):
                # --- 扰动第 j 维 ---
                # 此时 p_temp 处于 (base + ei) 的状态，再改 j 变为 (base + ei + ej)
                orig_j = p_temp[j]
                p_temp[j] = center_vec[j]
                
                # 计算双变量扰动 fp4
                fp4 = self.fun(p_temp)
                
                # --- 还原第 j 维 ---
                # 必须立刻还原，以便下一次 j 循环使用的是正确的基准
                p_temp[j] = orig_j 
                
                # --- 记录数据 ---
                f_archive[i, j] = fp4
                f_archive[j, i] = fp4 # 对称填充
                
                fp3 = f_single[j] # 查表获取 fp3 (f(base+ej))
                d2 = fp4 - fp3
                
                delta1[i, j] = d1
                delta2[i, j] = d1 # 原逻辑保留
                
                # 计算交互强度
                lambda_[i, j] = abs(d1 - d2)
                lambda_[j, i] = lambda_[i, j] # 保持对称性
            
            # --- 还原第 i 维 ---
            # i 循环结束，还原 p_temp 到全 base 状态
            p_temp[i] = orig_i

        # --- 封装返回结果 ---
        delta.delta1 = delta1
        delta.delta2 = delta2
        evaluations.base = f_base
        evaluations.F = f_archive
        evaluations.fhat = f_single.reshape(-1, 1) # 保持原格式 (D, 1)

        return delta, evaluations, lambda_

    # DSM: Decomposition Structure Matrix (保持原逻辑)
    def dsm(self, evaluation, lambda_):
        dim = self.info['dimension']
        fhat_archive = evaluation.fhat
        f_archive = evaluation.F
        fp1 = evaluation.base

        # 构造用于计算阈值的矩阵
        F1 = np.ones((dim, dim)) * fp1
        F2 = np.tile(fhat_archive.T, (dim, 1))
        F3 = np.tile(fhat_archive, (1, dim))
        F4 = f_archive
        
        # 计算最大函数值幅度，用于确定机器精度误差界限
        FS_stack1 = np.stack((F1, F2, F3, F4), axis=2)
        Fmax = np.nanmax(FS_stack1, axis=2) # 使用 nanmax 防止 NaN 传播
        
        FS_stack2 = np.stack((F1 + F4, F2 + F3), axis=2)
        Fmax_inf = np.nanmax(FS_stack2, axis=2)

        theta = np.full((dim, dim), np.nan)
        
        # 机器 epsilon
        muM = np.finfo(float).eps / 2
        
        # 误差阈值计算公式 (参考 Omidvar et al. 2014)
        gamma = lambda n: (n * muM) / (1 - n * muM) 
        errlb = gamma(2) * Fmax_inf 
        errub = gamma(dim**0.5) * Fmax  # F7 此处要乘以 1000
        
        # 初始分类
        I1 = lambda_ <= errlb
        theta[I1] = 0
        
        I2 = lambda_ >= errub
        theta[I2] = 1
        
        # 统计以计算自适应阈值 epsilon
        I0 = (lambda_ == 0)
        c0 = np.sum(I0)
        count_seps = np.sum(~I0 & I1) # 0 < lambda <= errlb
        count_nonseps = np.sum(I2)
        reliable_calcs = count_seps + count_nonseps
        
        # 避免除以零
        if c0 + reliable_calcs == 0:
            epsilon = errlb # Fallback
        else:
            w1 = ((count_seps + c0) / (c0 + reliable_calcs)) 
            w2 = ((count_nonseps) / (c0 + reliable_calcs))
            epsilon = w1 * errlb + w2 * errub

        # 再次过滤中间区域
        AdjTemp = lambda_ > epsilon 
        idx = np.isnan(theta)
        theta[idx] = AdjTemp[idx] 
        
        # 对角线总是相关的
        theta[np.diag_indices(dim)] = 1

        return theta.astype(int)