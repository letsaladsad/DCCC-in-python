import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def BFS(theta):
    """
    使用 Scipy 加速连通分量查找
    """
    L = theta.shape[0]
    if L == 0:
        return {'seps': [], 'nonseps': []}

    # 1. 核心加速：使用 Scipy 的 C 语言实现查找连通分量
    # directed=False 表示视作无向图（DG2 中 theta 通常是对称的）
    n_components, labels = connected_components(csr_matrix(theta), directed=False, return_labels=True)

    # 2. 快速分组：使用 argsort 替代循环 np.where
    # 这将复杂度从 O(K*N) 降为 O(N log N)
    idx_sorted = np.argsort(labels)
    sorted_labels = labels[idx_sorted]
    
    # 找到标签变化的切分点
    # diff 的位置就是分组的边界
    cut_points = np.where(sorted_labels[:-1] != sorted_labels[1:])[0] + 1
    components = np.split(idx_sorted, cut_points)

    # 3. 分离 seps (size=1) 和 nonseps (size>1)
    seps = []
    nonseps = []

    for comp in components:
        if len(comp) == 1:
            seps.append(comp[0]) # 存入 int
        else:
            nonseps.append(comp.tolist()) # 存入 list

    return {'seps': seps, 'nonseps': nonseps}