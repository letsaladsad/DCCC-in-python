from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F4: 7-subcomponent Rotated Elliptic
# ==========================================
class F4(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 4
        self.s_size = 7
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -100.0
        self.maxX = 100.0
        
        # --- 优化: 预计算子问题配置 ---
        self.sub_problems = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        for i in range(self.s_size):
            dim = self.s[i]
            indices = self.Pvector[c : c + dim]
            matrix = rot_map[dim]
            weight = self.w[i]
            self.sub_problems.append((indices, matrix, weight))
            c += dim
        self.c_end = c # 记录主要部分结束的位置，用于处理 remainder

    def __call__(self, x): return self.compute(x)
    def info(self): return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}

    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        result = np.zeros(x.shape[0], dtype=x.dtype)
        
        # 1. Global Shift
        z_global = x - self.Ovector
        
        # 2. Sub-problems Loop (Fast)
        for indices, matrix, w in self.sub_problems:
            # Extract -> Rotate -> OSZ -> Elliptic
            z_sub = z_global[:, indices] @ matrix.T 
            z_sub = self.transform_osz(z_sub)
            result += w * self.elliptic(z_sub)

        # 3. Remainder (if any)
        if self.c_end < self.dimension:
            indices = self.Pvector[self.c_end : self.dimension]
            z_rem = z_global[:, indices]
            z_rem = self.transform_osz(z_rem)
            result += self.elliptic(z_rem)

        return result
