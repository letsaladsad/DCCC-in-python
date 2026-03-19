from .benchmarks import Benchmarks
import numpy as np

# ==========================================
# F12: Shifted Rosenbrock
# ==========================================
class F12(Benchmarks):
    def __init__(self):
        super().__init__()
        self.ID = 12
        self.Ovector = self.readOvector()
        self.minX = -100.0
        self.maxX = 100.0

    def __call__(self, x): return self.compute(x)
    def info(self): return {'best': 0.0, 'dimension': self.dimension, 'lower': self.minX, 'upper': self.maxX}

    def compute(self, x):
        if not isinstance(x, np.ndarray): x = np.array(x)
        if x.ndim == 1: x = x[np.newaxis, :]
        
        # Logic: Global Shift -> Rosenbrock
        z = x - self.Ovector
        result = self.rosenbrock(z)
        
        return result