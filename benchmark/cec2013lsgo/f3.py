from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F3: Shifted Ackley Function
# ==========================================
class F3(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 3
        self.Ovector = self.readOvector()
        self.minX = -32.0
        self.maxX = 32.0

    def __call__(self, x):
        return self.compute(x)
    
    def info(self):
        return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}

    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        # 逻辑: Shift -> OSZ -> ASY -> Lambda -> Ackley
        z = x - self.Ovector
        z = self.transform_osz(z)
        z = self.transform_asy(z, 0.2)
        z = self.Lambda(z, 10)
        result = self.ackley(z)
        
        return result
