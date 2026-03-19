from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F14: Shifted Rotated Conflicting Schwefel
# ==========================================
class F14(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 14
        self.s_size = 20
        self.dimension = 905
        self.overlap = 5
        self.s = self.readS(self.s_size)
        # 注意：F14 读取的是 OvectorVec (一组向量)，不是单个 Ovector
        self.OvectorVec = self.readOvectorVec()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.w = self.readW(self.s_size)
        self.minX = -100.0
        self.maxX = 100.0
        
        # --- 关键优化: 预计算 Conflicting 索引和对应的位移向量 ---
        self.sub_problems = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        
        for i in range(self.s_size):
            dim = self.s[i]
            start_idx = c - i * self.overlap
            end_idx = c + dim - i * self.overlap
            
            indices = self.Pvector[start_idx : end_idx]
            # 获取当前子问题的独立位移向量
            ovec_sub = self.OvectorVec[i]
            
            self.sub_problems.append((indices, ovec_sub, rot_map[dim], self.w[i]))
            c += dim

    def __call__(self, x): return self.compute(x)
    def info(self): return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}

    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        result = np.zeros(x.shape[0], dtype=x.dtype)
        
        # F14 没有全局 Shift，是在每个子空间内独立 Shift
        for indices, ovec_sub, matrix, w in self.sub_problems:
            # 1. Slice Raw X
            z = x[:, indices]
            # 2. Local Shift
            z = z - ovec_sub
            # 3. Rotate
            z = z @ matrix.T
            # 4. Transforms
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += w * self.schwefel(z)

        return result
    


