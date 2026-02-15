# -*- coding: utf-8 -*-
"""
快速测试：验证FF特征提取功能
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from util.fiducial_feature_extraction import compute_features_from_j_peaks
from scipy.signal import find_peaks

def test_ff_feature_extraction():
    """测试FF特征提取函数"""
    print("="*60)
    print("测试FF特征提取功能")
    print("="*60)
    
    # 生成模拟BCG信号
    fs = 250
    duration = 5  # 5秒
    t = np.arange(0, duration, 1/fs)
    
    # 模拟心跳信号（假设心率为70 bpm）
    hr = 70
    heart_period = 60 / hr  # 秒
    bcg = np.sin(2 * np.pi * (1/heart_period) * t) + 0.1 * np.random.randn(len(t))
    
    # 检测峰值
    distance = int(0.7 * (60 / hr) * fs)
    peaks, _ = find_peaks(bcg, distance=distance)
    
    print(f"\n模拟信号参数:")
    print(f"  采样率: {fs} Hz")
    print(f"  信号长度: {len(bcg)} 点 ({duration}秒)")
    print(f"  模拟心率: {hr} bpm")
    print(f"  检测到峰值数: {len(peaks)}")
    
    # 计算FF特征
    ff_features = compute_features_from_j_peaks(bcg, peaks, fs=fs)
    
    print(f"\nFF特征 (9维):")
    print(f"  特征维度: {ff_features.shape}")
    print(f"  特征值: {ff_features}")
    
    # 验证特征维度
    assert ff_features.shape == (9,), f"特征维度错误：期望(9,)，实际{ff_features.shape}"
    
    # 验证没有NaN
    assert not np.isnan(ff_features).any(), "特征中包含NaN值"
    
    print("\n✅ 测试通过！")
    
    # 打印特征解释
    print("\n特征说明:")
    print(f"  [0] J峰间隔均值: {ff_features[0]:.4f} 秒")
    print(f"  [1] J峰间隔标准差: {ff_features[1]:.4f} 秒")
    print(f"  [2] J峰间隔CV: {ff_features[2]:.4f}")
    print(f"  [3] J峰幅值均值: {ff_features[3]:.4f}")
    print(f"  [4] J峰幅值标准差: {ff_features[4]:.4f}")
    print(f"  [5] J峰幅值CV: {ff_features[5]:.4f}")
    print(f"  [6] 平均心率: {ff_features[6]:.2f} bpm")
    print(f"  [7] 心率变异性RMSSD: {ff_features[7]:.4f}")
    print(f"  [8] Extremum (J峰幅值均值): {ff_features[8]:.4f}")

if __name__ == '__main__':
    test_ff_feature_extraction()
