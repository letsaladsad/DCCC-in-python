from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F1: Shifted Elliptic Function
# ==========================================
class F1(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 1
        self.Ovector = self.readOvector()
        self.minX = -100.0
        self.maxX = 100.0

    def __call__(self, x):
        return self.compute(x)

    def info(self):
        return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}
    
    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        # 逻辑: Shift -> Transform -> Elliptic
        z = x - self.Ovector
        z = self.transform_osz(z)
        result = self.elliptic(z)

        return result
