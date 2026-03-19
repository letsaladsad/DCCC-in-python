from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F13: Shifted Rotated Conforming Schwefel (Overlapping)
# ==========================================
class F13(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 13
        self.s_size = 20
        self.dimension = 905 # Overlapping dimensions
        self.overlap = 5
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -100.0
        self.maxX = 100.0
        
        # --- 关键优化: 预计算重叠索引 ---
        self.sub_problems = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        
        for i in range(self.s_size):
            dim = self.s[i]
            # 计算重叠后的实际索引位置
            start_idx = c - i * self.overlap
            end_idx = c + dim - i * self.overlap
            
            indices = self.Pvector[start_idx : end_idx]
            
            self.sub_problems.append((indices, rot_map[dim], self.w[i]))
            c += dim

    def __call__(self, x): return self.compute(x)
    def info(self): return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}

    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        result = np.zeros(x.shape[0], dtype=x.dtype)
        
        # F13 先进行全局 Shift
        z_global = x - self.Ovector

        for indices, matrix, w in self.sub_problems:
            # Slice (Overlapping) -> Rotate -> OSZ -> ASY -> Schwefel
            z = z_global[:, indices] @ matrix.T
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += w * self.schwefel(z)
            
        return result