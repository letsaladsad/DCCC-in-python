from baseline.dg2.dg2 import DG2
from baseline.dccc.dccc import DCCC

import numpy as np
import matplotlib.pyplot as plt
from benchmark.cec2013lsgo.cec2013 import Benchmark

benchmark = Benchmark()
for fun_id in range(1, 16):
    fun = benchmark.get_function(fun_id)
    info = benchmark.get_info(fun_id)

    # 生成子空间划分
    dg2 = DG2(fun, info)
    subspaces = dg2.run()
    print('Function ID:', fun_id)
    print('nonseps', len(subspaces['nonseps']))
    print('sep', len(subspaces['seps']))

    # DCCC 优化
    dccc = DCCC(fun, info, subspaces)
    bestfit, fitness_curve = dccc.run()
    print("\nFinal best fitness:", bestfit)

    # 绘制fitness下降曲线
    plt.figure(figsize=(6, 4))
    plt.plot(fitness_curve, marker='o', markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.yscale("log")
    plt.title(f"Function {fun_id} Fitness Curve")
    plt.grid(True)
    plt.show()