import numpy as np
from scipy.integrate import simps
from itertools import combinations


def compute_features(BCG, H_locs, I_locs, J_locs, K_locs, L_locs):
    fs = 250
    # 定义你的基准点位置
    locs = {
        "H": H_locs,
        "I": I_locs,
        "J": J_locs,
        "K": K_locs,
        "L": L_locs,
    }

    # 计算各个基准点对应的幅值
    amps = {name: BCG[loc] for name, loc in locs.items()}

    features = {}

    # Time Interval: 基准点位置两两之间的差值

    features["Time Interval"] = {f'{name1}-{name2}': np.abs(loc1 - loc2) / fs for (name1, loc1), (name2, loc2) in
                                 combinations(locs.items(), 2)}

    # Time Ratio: 每一对差值与H-L的差值的占比
    epsilon = 1e-8
    features["Time Ratio"] = {name: np.abs(diff) / (np.abs(features["Time Interval"]["H-L"]) + epsilon) for name, diff in
                              features["Time Interval"].items()}
    del features["Time Ratio"]['H-L']

    # Extremum: 基准点位置对应的信号幅值
    features["Extremum"] = amps

    # Displacement: 两两基准点位置对应的信号幅值的差
    features["Displacement"] = {f'{name1}-{name2}': np.abs(amp1 - amp2) for (name1, amp1), (name2, amp2) in
                                combinations(amps.items(), 2)}

    # AUC: 任意两个基准点之间的BCG信号与X轴之间的面积
    features["AUC"] = {}
    for (name1, loc1), (name2, loc2) in combinations(locs.items(), 2):
        area_list = []
        for i in range(min(len(loc1), len(loc2))):
            start, end = sorted([max(0, min(len(BCG) - 1, loc1[i])), max(0, min(len(BCG) - 1, loc2[i]))])
            if end - start <= 1:  # ensure that there's at least one point between start and end
                area_list.append(0)
                continue
            lower_bound = min(amps["K"][i], amps["I"][i])
            area = simps(np.abs(BCG[start:end] - lower_bound))
            area_list.append(area)
        features["AUC"][f'{name1}-{name2}'] = np.array(area_list)

    return features


def compute_features_from_j_peaks(BCG, J_locs, fs=250):
    """
    基于J峰位置计算9维特征
    
    参数:
        BCG: BCG信号数组
        J_locs: J峰位置数组（索引）
        fs: 采样率，默认250Hz
        
    返回:
        features: numpy数组 (9,)，包含：
            [0-2]: J峰间隔统计（均值、标准差、CV）
            [3-5]: J峰幅值统计（均值、标准差、CV）
            [6]: 平均心率
            [7]: 心率变异性RMSSD
            [8]: J峰幅值均值（Extremum）

        9维特征说明：
            J峰间隔均值：连续J峰间隔的平均值（秒）
            J峰间隔标准差：间隔的变化程度
            J峰间隔CV：变异系数 (std/mean)，归一化的变化指标
            J峰幅值均值：J峰处BCG信号幅值的平均
            J峰幅值标准差：幅值的变化程度
            J峰幅值CV：归一化的幅值变化
            平均心率：60 / 平均间隔 (bpm)
            心率变异性RMSSD：连续间隔差值的均方根
            Extremum：J峰幅值均值（类似原44维FF中的Extremum特征）
    """
    if len(J_locs) < 2:
        # 如果J峰数量不足，返回零向量
        return np.zeros(9)
    
    # 1. J峰间隔特征（3维）
    intervals = np.diff(J_locs) / fs  # 转换为秒
    interval_mean = np.mean(intervals)
    interval_std = np.std(intervals)
    interval_cv = interval_std / (interval_mean + 1e-8)  # 变异系数
    
    # 2. J峰幅值特征（3维）
    amplitudes = BCG[J_locs]
    amp_mean = np.mean(amplitudes)
    amp_std = np.std(amplitudes)
    amp_cv = amp_std / (np.abs(amp_mean) + 1e-8)  # 变异系数
    
    # 3. 平均心率（1维）
    hr = 60 / interval_mean  # bpm
    
    # 4. 心率变异性RMSSD（1维）
    # RMSSD: Root Mean Square of Successive Differences
    rmssd = np.sqrt(np.mean(np.diff(intervals) ** 2))
    
    # 5. Extremum - J峰幅值均值（1维）
    extremum = amp_mean
    
    # 组合为9维特征向量
    features = np.array([
        interval_mean, interval_std, interval_cv,
        amp_mean, amp_std, amp_cv,
        hr, rmssd, extremum
    ], dtype=np.float32)
    
    return features
