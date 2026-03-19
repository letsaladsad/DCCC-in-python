import numpy as np
import math
from numba import njit

# ==========================================
# Numba JIT Accelerated Core Functions
# ==========================================

@njit(fastmath=True, cache=True) 
def jit_transform_osz(z):
    # 极速版 OSZ 变换：单核优化 + 代数化简
    # 直接修改原数组 (In-place)
    flat_z = z.ravel()
    n = flat_z.size
    
    for i in range(n):
        val = flat_z[i]
        if val == 0:
            continue

        abs_val = math.fabs(val)
        hat = math.log(abs_val)
        
        if val > 0:
            c1, c2 = 10.0, 7.9
        else:
            c1, c2 = 5.5, 3.1
            
        sin_term = math.sin(c1 * hat) + math.sin(c2 * hat)
        
        # 代数化简优化: val * exp(delta) 替代 sign * exp(log(|val|) + delta)
        flat_z[i] = val * math.exp(0.049 * sin_term)
        
    return z

@njit(fastmath=True, cache=True)
def jit_transform_asy(z, beta):
    # 极速版 ASY 变换
    n, dim = z.shape
    for i in range(n):
        for j in range(dim):
            val = z[i, j]
            if val > 0:
                sqrt_val = math.sqrt(val)
                exponent = 1 + beta * (j / (dim - 1)) * sqrt_val
                z[i, j] = val ** exponent
    return z

@njit(fastmath=True, cache=True)
def jit_lambda(z, alpha):
    # Lambda 变换
    n, dim = z.shape
    for i in range(n):
        for j in range(dim):
            # exponents = 0.5 * j / (dim - 1)
            # z[i, j] = z[i, j] * (alpha ** exponents)
            # 优化：合并幂运算
            exponent = 0.5 * j / (dim - 1)
            factor = math.pow(alpha, exponent)
            z[i, j] *= factor
    return z

# --- Basic Benchmark Functions (JIT) ---

@njit(fastmath=True, cache=True)
def jit_sphere(x):
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        sum_sq = 0.0
        for j in range(dim):
            val = x[i, j]
            sum_sq += val * val
        res[i] = sum_sq
    return res

@njit(fastmath=True, cache=True)
def jit_elliptic(x):
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        total = 0.0
        for j in range(dim):
            # 10 ** (6 * j / (dim - 1))
            coeff = math.pow(10.0, 6.0 * j / (dim - 1))
            val = x[i, j]
            total += coeff * (val * val)
        res[i] = total
    return res

@njit(fastmath=True, cache=True)
def jit_rastrigin(x):
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        total = 0.0
        for j in range(dim):
            val = x[i, j]
            # x^2 - 10*cos(2*pi*x) + 10
            total += val * val - 10.0 * math.cos(2.0 * math.pi * val) + 10.0
        res[i] = total
    return res

@njit(fastmath=True, cache=True)
def jit_ackley(x):
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        sum_sq = 0.0
        sum_cos = 0.0
        for j in range(dim):
            val = x[i, j]
            sum_sq += val * val
            sum_cos += math.cos(2.0 * math.pi * val)
        
        term1 = -20.0 * math.exp(-0.2 * math.sqrt(sum_sq / dim))
        term2 = -math.exp(sum_cos / dim)
        res[i] = term1 + term2 + 20.0 + math.e
    return res

@njit(fastmath=True, cache=True)
def jit_schwefel(x):
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        cumsum = 0.0
        sum_sq = 0.0
        for j in range(dim):
            cumsum += x[i, j]
            sum_sq += cumsum * cumsum
        res[i] = sum_sq
    return res

@njit(fastmath=True, cache=True)
def jit_rosenbrock(x):
    n, dim = x.shape
    res = np.empty(n, dtype=x.dtype)
    for i in range(n):
        total = 0.0
        # Rosenbrock 遍历到 dim-2
        for j in range(dim - 1):
            x0 = x[i, j]
            x1 = x[i, j + 1]
            t = x0 * x0 - x1
            term1 = 100.0 * t * t
            term2 = (x0 - 1.0) * (x0 - 1.0)
            total += term1 + term2
        res[i] = total
    return res

# ==========================================
# Main Class
# ==========================================

class Benchmarks:
    def __init__(self):
        self.dtype = np.float64
        self.data_dir = "./benchmark/cec2013lsgo/datafiles"
        self.dimension = 1000

        self.min_dim = 25
        self.med_dim = 50
        self.max_dim = 100

        self.ID = None
        self.s_size = 20
        self.overlap = None
        self.minX = None
        self.maxX = None
        self.Ovector = None
        self.OvectorVec = None
        self.Pvector = None
        
        self.r25 = None
        self.r50 = None
        self.r100 = None
        self.r_min_dim = None
        self.r_med_dim = None
        self.r_max_dim = None

        self.anotherz = np.zeros(self.dimension, dtype=self.dtype)
        self.anotherz1 = None
        self.best_fitness = float('inf')
        
        self.maxevals = 3000000
        self.numevals = 0

        self.output = ""
        self.output_dir = 'cec2013lsgo_py'
        self.record_evels = [120000, 600000, 3000000]

    def readOvector(self):
        d = np.zeros(self.dimension, dtype=self.dtype)
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = float(value)
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        return d

    def readOvectorVec(self):
        d = [np.zeros(self.s[i], dtype=self.dtype) for i in range(self.s_size)]
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                i = -1
                up = 0
                for line in file:
                    if c == up:
                        i += 1
                        if i < self.s_size:
                            up += self.s[i]
                    values = line.strip().split(',')
                    for value in values:
                        if i < self.s_size and c < self.dimension:
                            idx_in_group = c - (up - self.s[i])
                            if idx_in_group < len(d[i]):
                                d[i][idx_in_group] = float(value)
                        c += 1
        except FileNotFoundError:
            print(f"Cannot open the OvectorVec datafiles '{file_path}'")
        return d

    def readPermVector(self):
        d = np.zeros(self.dimension, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-p.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = int(float(value)) - 1
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        return d

    def readR(self, sub_dim):
        m = np.zeros((sub_dim, sub_dim), dtype=self.dtype)
        file_path = f"{self.data_dir}/F{self.ID}-R{sub_dim}.txt"
        try:
            with open(file_path, 'r') as file:
                i = 0
                for line in file:
                    values = line.strip().split(',')
                    for j, value in enumerate(values):
                        if i < sub_dim and j < sub_dim:
                            m[i, j] = float(value)
                    i += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        return m

    def readS(self, num):
        self.s = np.zeros(num, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-s.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    if c < num:
                        self.s[c] = int(float(line.strip()))
                        c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        return self.s

    def readW(self, num):
        self.w = np.zeros(num, dtype=self.dtype)
        file_path = f"{self.data_dir}/F{self.ID}-w.txt"
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    if c < num:
                        self.w[c] = float(line.strip())
                        c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        return self.w

    def multiply(self, vector, matrix):
        # 批量矩阵乘法，Numpy 的 @ 算符已经非常优化 (BLAS)
        if vector.ndim == 1:
            return np.dot(matrix, vector)
        return vector @ matrix.T

    # 兼容性保留的旋转函数，主要逻辑应在子类 F14 等中优化 (如 F14.py 所示)
    def rotateVector(self, i, c):
        sub_dim = self.s[i]
        indices = self.Pvector[c:c + sub_dim]
        
        if self.anotherz.ndim == 1:
             z = self.anotherz[indices]
        else:
             z = self.anotherz[:, indices]

        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            self.anotherz1 = None
        return self.anotherz1

    def rotateVectorConform(self, i, c):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        indices = self.Pvector[start_index:end_index]
        
        if self.anotherz.ndim == 1:
            z = self.anotherz[indices]
        else:
            z = self.anotherz[:, indices]

        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            self.anotherz1 = None
        return self.anotherz1

    def rotateVectorConflict(self, i, c, x):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        indices = self.Pvector[start_index:end_index]
        
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        if x.ndim == 1:
            z = x[indices] - self.OvectorVec[i]
        else:
            z = x[:, indices] - self.OvectorVec[i]

        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            self.anotherz1 = None
        return self.anotherz1

    # --- 将类方法转发到 JIT 函数 ---
    
    def sphere(self, x):
        return jit_sphere(x)

    def elliptic(self, x):
        return jit_elliptic(x)

    def rastrigin(self, x):
        return jit_rastrigin(x)

    def ackley(self, x):
        return jit_ackley(x)

    def schwefel(self, x):
        return jit_schwefel(x)

    def rosenbrock(self, x):
        return jit_rosenbrock(x)

    def transform_osz(self, z):
        return jit_transform_osz(z)

    def transform_asy(self, z, beta=0.2):
        return jit_transform_asy(z, beta)

    def Lambda(self, z, alpha=10):
        return jit_lambda(z, alpha)




    






    




    






