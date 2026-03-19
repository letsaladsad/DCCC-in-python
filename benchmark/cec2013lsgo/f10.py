from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F10: 20-subcomponent Rotated Ackley
# ==========================================
class F10(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 10
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.minX = -32.0
        self.maxX = 32.0
        
        self.sub_problems = []
        c = 0
        rot_map = {25: self.r25, 50: self.r50, 100: self.r100}
        for i in range(self.s_size):
            dim = self.s[i]
            self.sub_problems.append((self.Pvector[c:c+dim], rot_map[dim], self.w[i]))
            c += dim

    def __call__(self, x): return self.compute(x)
    def info(self): return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}

    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        result = np.zeros(x.shape[0], dtype=x.dtype)
        z_global = x - self.Ovector

        for indices, matrix, w in self.sub_problems:
            # Rotate -> OSZ -> ASY -> Lambda -> Ackley
            z = z_global[:, indices] @ matrix.T
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            z = self.Lambda(z, 10)
            result += w * self.ackley(z)

        return result
