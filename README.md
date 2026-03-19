# DCCC

**基于难度与贡献度的协同进化大规模全局优化算法**  
**Difficulty and Contribution Based Cooperative Coevolution for Large-Scale Optimization**

---

## 算法简介 / Overview

DCCC 是一种用于求解大规模全局优化问题（LSGO）的协同进化算法。它基于 DG2 分解结果进一步划分子空间，并使用 SaNSDE 作为子优化器，通过动态评估子空间的贡献度与难度，自适应地选择下一轮优化的子空间。  
DCCC is a cooperative co-evolution algorithm for Large-Scale Global Optimization (LSGO). It further partitions subspaces based on DG2 decomposition, uses SaNSDE as the sub-optimizer, and dynamically selects the next subspace to optimize based on its contribution and difficulty.

算法结合以下机制 / The algorithm combines the following mechanisms:

- **DG2 分组 / DG2 grouping**：快速准确地识别问题的可分离与非可分离变量组。Fast and accurate identification of separable and non-separable variable groups.
- **分块处理 / Block processing**：将非可分离组（块大小 100）和可分离变量（块大小 20）进一步划分，降低子空间维度。Further partition non-separable groups (block size 100) and separable variables (block size 20) to reduce subspace dimensionality.
- **SaNSDE 子优化器 / SaNSDE sub-optimizer**：自适应差分进化算法，自适应选择变异策略（rand/1 或 current-to-best/2）、缩放因子 F 的分布（正态或柯西）以及交叉率 CR。Self-adaptive DE that adaptively selects mutation strategy (rand/1 or current-to-best/2), scaling factor distribution (normal or Cauchy), and crossover rate CR.
- **贡献度评估 / Contribution evaluation**：记录每次优化后分组的适应度总提升（累积贡献度 δ）。Accumulates fitness improvement (contribution δ) after each optimization.
- **难度评估 / Difficulty evaluation**：基于种群个体适应度与到最优个体距离的相关性，计算分组难度（1 - |相关系数|）。Computes group difficulty based on correlation between fitness and distance to best individual (1 - |correlation|).
- **动态选择 / Dynamic selection**：以概率 0.3 随机选择分组，否则选择贡献度 × 难度最大的分组进行下一轮优化。With probability 0.3, randomly select a group; otherwise, select the group with highest contribution × difficulty for next optimization.

> 本算法是论文 *DCCC: Difficulty and Contribution Based Cooperative Coevolution for Large-Scale Optimization* 的实现。子优化器参考了 DG2 分组算法和 SaNSDE 差分进化算法。  
> This algorithm implements the paper *DCCC: Difficulty and Contribution Based Cooperative Coevolution for Large-Scale Optimization*. The sub-optimizer is inspired by DG2 grouping and SaNSDE differential evolution.

---

## 文件结构 / File Structure

```
demo/
├── baseline/
│   ├── dccc/
│   │   └── dccc.py          # DCCC 算法主类 / DCCC main class
│   └── dg2/
│       ├── bfs.py           # BFS 辅助函数（用于DG2）/ BFS helper for DG2
│       └── dg2.py           # DG2 分组算法 / DG2 grouping algorithm
├── benchmark/                # 基准测试函数（略）/ Benchmark functions (omitted)
├── test.py                   # 主运行文件，演示 DCCC 使用 / Main runner, demonstrates DCCC usage
├── utils.py                  # 工具函数 / Utility functions
└── README.md                 # 本文件 / This file
```

`dccc.py` 内部结构 / Internal structure of `dccc.py`:

| 组件 / Component | 说明 / Description |
|---|---|
| `DCCC` | 算法主类 / Main class |
| `_build_groups()` | 基于 DG2 子空间构建分组（块大小：非可分离 100，可分离 20）/ Build groups from DG2 subspaces (block size: non-sep 100, sep 20) |
| `_init_population()` | 初始化种群（均匀随机，大小 100）/ Initialize population (uniform random, size 100) |
| `_optimizer()` | SaNSDE 子优化器，自适应变异策略、F 分布、CR，并更新贡献度 / SaNSDE sub-optimizer with adaptive strategy, F distribution, CR, and contribution update |
| `_evaluate_difficulty()` | 评估分组难度（基于适应度-距离相关性）/ Evaluate group difficulty (based on fitness-distance correlation) |
| `_selector()` | 根据贡献度和难度选择下一分组（或随机）/ Select next group based on contribution and difficulty (or random) |
| `run()` | 主优化循环，迭代至最大评估次数 / Main optimization loop until max FEs |

---

## 环境依赖 / Requirements

```
Python >= 3.8
numpy
scipy
tqdm (可选，用于进度条 / optional, for progress bar)
```

---

## 快速开始 / Quick Start

确保当前目录在 `demo/` 下，直接运行测试脚本：  
Make sure you are in the `demo/` directory and run the test script:

```bash
python test.py
```

测试脚本将依次执行 DG2 分组和 DCCC 优化，输出类似以下信息：  
The test script will run DG2 grouping followed by DCCC optimization, with output similar to:

```
DG2 Structure Search: 100%|██████████| 999/999 [00:35<00:00, 27.79it/s] 
Function ID: 1
nonseps 0
sep 1000
 Iter | Subspace |  Dim |         BestFit |      Delta
------------------------------------------------------------
    0 |        1 |  100 |    3.782571e+11 |  4.815e+08
    1 |        9 |  100 |    1.030320e+11 |  3.325e+13
...
Final best fitness: 4.520319e+2
```

**自定义运行 / Custom run:**

在 `test.py` 中修改以下变量以调整实验设置：  
Modify these variables in `test.py` to adjust the experiment:

```python
FUNCTION_ID = 1                # 函数编号 1-15 / Function ID 1-15
MAX_FES = 3000000              # 最大评估次数 / Max function evaluations
```

或者在自己的脚本中调用 DCCC，参见“接口说明”。  
Alternatively, call DCCC in your own script as shown in the “Interface” section.

---

## 算法参数 / Algorithm Parameters

以下参数可在 `dccc.py` 的 `__init__` 或相应方法中调整：  
The following parameters can be adjusted in `__init__` or corresponding methods of `dccc.py`:

| 参数 / Parameter | 默认值 / Default | 说明 / Description |
|---|---|---|
| 种群大小 / Population size | 100 | 全局种群个体数 / Number of individuals in global population |
| 最大评估次数 / Max FEs | 3e6 | 算法停止的总函数评估预算 / Total function evaluation budget |
| 非分离分组块大小 / Non-sep block size | 100 | 每个非可分离组划分的块维度 / Block dimension for non-separable groups |
| 分离分组块大小 / Sep block size | 20 | 可分离变量划分的块维度 / Block dimension for separable variables |
| 分组最大迭代次数 / Group max iterations | 100 | 每次选中分组时运行的优化迭代次数 / Iterations per selected group optimization |
| 随机选择概率 / Random selection prob | 0.3 | 随机选择下一分组的概率 / Probability of random group selection |
| 贡献度衰减因子 / Contribution decay factor | 0.8 | 累积贡献度更新时的衰减系数 / Decay factor for accumulated contribution |

---

## 输出说明 / Output

运行过程中，控制台每轮迭代打印一行信息：  
During execution, console prints one line per iteration:

- `Iter`: 迭代次数 / Iteration number
- `Subspace`: 被选中的分组索引 / Selected group index
- `Dim`: 该分组的维度 / Group dimension
- `BestFit`: 当前全局最优适应度 / Current global best fitness
- `Delta`: 本次优化带来的适应度总提升（即贡献度）/ Total fitness improvement from this optimization (contribution)

最终打印最优适应度。  
Finally, the best fitness is printed.

`run()` 方法返回 `(best_fitness, fitness_curve)`，其中 `fitness_curve` 是每轮迭代后的全局最优适应度列表，可用于绘图分析。  
The `run()` method returns `(best_fitness, fitness_curve)`, where `fitness_curve` is a list of global best fitness after each iteration, useful for plotting.

---

## 接口说明 / Interface

在您自己的代码中使用 DCCC 的示例：  
Example of using DCCC in your own code:

```python
import numpy as np
from baseline.dccc.dccc import DCCC

# 定义目标函数 / Define objective function
def sphere(x):
    return np.sum(x**2)

# 问题信息 / Problem info
info = {'dimension': 1000, 'lower': -100, 'upper': 100}

# 运行 DG2 获取子空间结构（需调用 dg2.py）/ Obtain subspace structure from DG2 (call dg2.py)
from baseline.dg2.dg2 import DG2
dg2 = DG2(sphere, info)
subspaces = dg2.run(epsilon=1e-3)   # 根据 DG2 实现调整参数 / Adjust parameters per DG2 implementation

# 创建 DCCC 优化器 / Create DCCC optimizer
optimizer = DCCC(sphere, info, subspaces)

# 运行优化 / Run optimization
best_fitness, history = optimizer.run()

print(f"Best fitness: {best_fitness}")
```

`info` 字典必须包含 `'dimension'`、`'lower'`、`'upper'` 三个键。  
`info` dict must contain keys `'dimension'`, `'lower'`, `'upper'`.

`subspaces` 字典应包含 `'seps'`（可分离变量索引列表）和 `'nonseps'`（非可分离变量组列表）。  
`subspaces` dict should contain `'seps'` (list of separable indices) and `'nonseps'` (list of lists of non-separable groups).

---

## 测试说明 / Test Description

目前未包含单元测试，但 `test.py` 提供了一个完整的端到端运行示例，可验证算法在 CEC'2013 基准函数上的表现。运行 `test.py` 应能观察到适应度逐步下降，并最终输出一个较优的数值。  
No unit tests are included currently, but `test.py` provides a complete end-to-end run example to verify algorithm performance on CEC'2013 benchmark functions. Running `test.py` should show decreasing fitness and a reasonably good final value.

未来可添加单元测试覆盖分组构建、难度计算、选择器等核心模块。  
Future work may include unit tests for core modules such as group construction, difficulty calculation, and selector.

---

## 参考文献 / References

- [1] [Author Name et al.]. *DCCC: Difficulty and Contribution Based Cooperative Coevolution for Large-Scale Optimization*. To appear in IEEE Transactions on Evolutionary Computation, 2024.
- [2] Omidvar, M. N., et al. "DG2: A faster and more accurate differential grouping for large-scale black-box optimization." *IEEE Transactions on Evolutionary Computation*, vol. 21, no. 6, pp. 929-942, 2017.
- [3] Yang, Z., et al. "SaNSDE: Self-adaptive differential evolution with neighborhood search." *2008 IEEE Congress on Evolutionary Computation*, 2008.

---
