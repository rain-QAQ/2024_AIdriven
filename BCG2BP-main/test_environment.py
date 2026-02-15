# -*- coding: utf-8 -*-
"""
简单的环境测试脚本
用于验证 Python 环境和主要依赖是否正常工作
"""

import sys
import io

# 设置标准输出为 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("环境测试开始")
print("=" * 60)
print()

# 测试 1: Python 基础
print("1. 测试 Python 基础功能...")
try:
    result = sum([1, 2, 3, 4, 5])
    print(f"   [OK] Python 基础功能正常 (测试求和: {result})")
except Exception as e:
    print(f"   [FAIL] Python 基础功能失败: {e}")
print()

# 测试 2: NumPy
print("2. 测试 NumPy...")
try:
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print(f"   [OK] NumPy 导入成功 (版本: {np.__version__})")
    print(f"   [OK] NumPy 数组创建成功: {arr}")
except Exception as e:
    print(f"   [FAIL] NumPy 测试失败: {e}")
print()

# 测试 3: SciPy
print("3. 测试 SciPy...")
try:
    import scipy
    from scipy.io import loadmat
    print(f"   [OK] SciPy 导入成功 (版本: {scipy.__version__})")
except Exception as e:
    print(f"   [FAIL] SciPy 测试失败: {e}")
print()

# 测试 4: PyTorch
print("4. 测试 PyTorch...")
try:
    import torch
    print(f"   [OK] PyTorch 导入成功 (版本: {torch.__version__})")
    
    # 创建一个简单的张量
    tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f"   [OK] PyTorch 张量创建成功: {tensor}")
    
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"   [OK] CUDA 可用 (设备数量: {torch.cuda.device_count()})")
    else:
        print(f"   [INFO] CUDA 不可用 (将使用 CPU)")
    
    # 测试简单的运算
    result = tensor * 2
    print(f"   [OK] PyTorch 运算正常: {result}")
    
except Exception as e:
    print(f"   [FAIL] PyTorch 测试失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 测试 5: 其他常用库
print("5. 测试其他常用库...")
try:
    import matplotlib
    print(f"   [OK] Matplotlib 导入成功 (版本: {matplotlib.__version__})")
except Exception as e:
    print(f"   [WARN] Matplotlib 测试失败: {e}")

try:
    import pandas as pd
    print(f"   [OK] Pandas 导入成功 (版本: {pd.__version__})")
except Exception as e:
    print(f"   [WARN] Pandas 测试失败: {e}")

print()
print("=" * 60)
print("环境测试完成")
print("=" * 60)
