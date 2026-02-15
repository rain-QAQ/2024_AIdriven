# -*- coding: utf-8 -*-
"""
滤波效果对比测试脚本
比较带通滤波vs不滤波后的心率检测MAE
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from hdf5storage import loadmat
import os
from scipy import signal
import scipy.stats
from scipy.signal import savgol_filter

def bandpass_filter(signal, lowcut=0.5, highcut=20, fs=250, order=4):
    """
    带通滤波器
    
    参数:
        signal: 输入信号
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        fs: 采样率 (Hz)
        order: 滤波器阶数
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def calculate_hr_from_peaks(peak_locs, fs=250):
    """
    从J峰位置计算心率
    
    参数:
        peak_locs: J峰位置数组（索引）
        fs: 采样率
        
    返回:
        hr: 心率 (bpm)，如果峰值少于2个则返回NaN
    """
    if len(peak_locs) < 2:
        return np.nan
    
    intervals = np.diff(peak_locs) / fs  # 转换为秒
    hr = 60 / np.mean(intervals)  # 转换为bpm
    return hr


def downsample_1000_to_250(signal_data):
    """降采样从1000Hz到250Hz"""
    from scipy import signal as sp_signal
    downsampled = sp_signal.decimate(signal_data, 4, zero_phase=True)
    return downsampled

def smooth_signal(data, sample_rate, window_length=None, polyorder=3):
    '''smooths given signal using savitzky-golay filter
    '''

    if window_length == None:
        window_length = sample_rate // 10

    if window_length % 2 == 0 or window_length == 0: window_length += 1
    if window_length<=polyorder:
        polyorder = polyorder-1
    smoothed = savgol_filter(data, window_length = window_length,
                             polyorder = polyorder)
    return smoothed  

def compare_filter_effect():
    """主函数：对比滤波效果"""
    
    # 1. 加载心率CSV
    hr_csv_path = r'D:\zsy\坐垫\文献梳理及画图\论文\20250905第二次投递submitCode_npj digital medicine提交\sorted\result\pred_hr_1d_subject_level.csv'
    print(f'加载心率CSV: {hr_csv_path}')
    hr_df = pd.read_csv(hr_csv_path)
    print(f'  共{len(hr_df)}个被试')
    
    # 创建心率字典
    hr_dict = {}
    for _, row in hr_df.iterrows():
        hr_dict[str(row['subject_id'])] = {
            'true_hr': row['true_hr'],
            'pred_hr': row['pred_hr']
        }
    
    # 2. 加载BCG数据（修正路径）
    data_path = r'd:\zsy\坐垫\2024_AIdriven\BCG2BP-main\data\Preprocessed_Database_flat.mat'
    print(f'\n加载BCG数据: {data_path}')
    mat = loadmat(data_path)
    
    # 查找主数据字段
    main_key = None
    for key in mat.keys():
        if not key.startswith('__'):
            main_key = key
            print(f'  找到主数据字段: {main_key}')
            break
    
    if main_key is None:
        raise ValueError(f"未找到有效的数据字段")
    
    data_table = mat[main_key]
    subject_IDs = data_table['ID'].flatten()
    subject_NUM = len(subject_IDs)
    print(f'  总共 {subject_NUM} 个受试者')
    
    # 3. 对每个被试进行对比
    results = []
    fs = 250  # 降采样后的采样率
    window_length = 5 * fs  # 5秒窗口长度
    frame_shift = 2 * fs  # 2秒帧移
    
    for i in range(subject_NUM):
        subject_data = data_table.flatten()[i]
        
        # 提取ID
        user_id = subject_data['ID']
        if isinstance(user_id, np.ndarray):
            user_id = user_id.flatten()[0] if user_id.size > 0 else str(i)
        if isinstance(user_id, bytes):
            user_id = user_id.decode('utf-8')
        elif hasattr(user_id, 'item'):
            user_id = str(user_id.item())
        else:
            user_id = str(user_id)
        
        # add: ID映射 - 将X0132转换为X1032
        if user_id == 'X0132':
            user_id = 'X1032'
            print(f'\n处理被试 {i+1}/{subject_NUM}: X0132 (映射为 {user_id})')
        else:
            print(f'\n处理被试 {i+1}/{subject_NUM}: {user_id}')
        
        # 检查该被试是否在心率CSV中
        if user_id not in hr_dict:
            print(f'  ⚠️ 警告: {user_id} 不在心率CSV中，跳过')
            continue
        
        true_hr = hr_dict[user_id]['true_hr']
        pred_hr = hr_dict[user_id]['pred_hr']
        
        # 检查必要字段
        if 'Film0' not in subject_data.dtype.names:
            print(f'  ⚠️ 警告: 缺少Film0字段，跳过')
            continue
        
        # 提取BCG信号并降采样
        BCG_raw = subject_data['Film0'].flatten()
        BCG = downsample_1000_to_250(BCG_raw)
        print(f'  BCG信号长度: {len(BCG)} (降采样后)')
        
        # 计算distance参数（基于pred_hr）
        distance = int(0.7 * (60 / pred_hr) * fs)
        print(f'  参考心率: {pred_hr:.2f} bpm, distance: {distance}')
        
        # 在窗口内检测J峰并计算心率
        window_hrs_raw = []  # 不滤波的窗口心率列表
        window_hrs_filtered = []  # 滤波后的窗口心率列表
        
        window_start = 0
        window_end = window_start + window_length
        window_count = 0
        
        # 预先对整个信号进行滤波（方法2需要）
        # BCG_filtered = bandpass_filter(BCG, lowcut=0.5, highcut=20, fs=fs)  # 带通

        data_v       = np.nan_to_num(BCG,nan=np.nanmean(BCG))
        data_v       = signal.detrend(data_v,type='constant')
        data_v       = smooth_signal(data_v,sample_rate=fs)
        data_v       = savgol_filter(data_v,5,3,mode = 'nearest')
        BCG_filtered       = np.abs(scipy.signal.hilbert(data_v)) # 包络
        while window_end <= len(BCG):
            # 提取当前窗口的BCG段
            window_bcg_raw = BCG[window_start:window_end]
            window_bcg_filtered = BCG_filtered[window_start:window_end]
            
            # 方法1：不滤波，在窗口内检测峰值
            try:
                peaks_raw, _ = find_peaks(window_bcg_raw, distance=distance)
                if len(peaks_raw) >= 2:
                    hr_window_raw = calculate_hr_from_peaks(peaks_raw, fs)
                    if not np.isnan(hr_window_raw):
                        window_hrs_raw.append(hr_window_raw)
            except:
                pass
            
            # 方法2：滤波后，在窗口内检测峰值
            try:
                peaks_filtered, _ = find_peaks(window_bcg_filtered, distance=distance)
                if len(peaks_filtered) >= 2:
                    hr_window_filtered = calculate_hr_from_peaks(peaks_filtered, fs)
                    if not np.isnan(hr_window_filtered):
                        window_hrs_filtered.append(hr_window_filtered)
            except:
                pass
            
            # 更新窗口
            window_start += frame_shift
            window_end = window_start + window_length
            window_count += 1
        
        print(f'  总窗口数: {window_count}')
        print(f'  不滤波有效窗口: {len(window_hrs_raw)}')
        print(f'  滤波后有效窗口: {len(window_hrs_filtered)}')
        
        # 计算平均心率
        if len(window_hrs_raw) > 0:
            hr_raw = np.mean(window_hrs_raw)
            mae_raw = abs(hr_raw - true_hr)
            print(f'  不滤波: 平均HR={hr_raw:.2f} bpm, MAE={mae_raw:.2f}')
        else:
            hr_raw = np.nan
            mae_raw = np.nan
            print(f'  不滤波: 无有效窗口')
        
        if len(window_hrs_filtered) > 0:
            hr_filtered = np.mean(window_hrs_filtered)
            mae_filtered = abs(hr_filtered - true_hr)
            print(f'  滤波后: 平均HR={hr_filtered:.2f} bpm, MAE={mae_filtered:.2f}')
        else:
            hr_filtered = np.nan
            mae_filtered = np.nan
            print(f'  滤波后: 无有效窗口')
        
        # 记录结果
        results.append({
            'subject_id': user_id,
            'true_hr': true_hr,
            'pred_hr': pred_hr,
            'hr_raw': hr_raw,
            'hr_filtered': hr_filtered,
            'mae_raw': mae_raw,
            'mae_filtered': mae_filtered,
            'n_windows_raw': len(window_hrs_raw),
            'n_windows_filtered': len(window_hrs_filtered)
        })
    
    # 4. 汇总结果
    print('\n' + '='*60)
    print('汇总结果')
    print('='*60)
    
    results_df = pd.DataFrame(results)
    
    # 过滤掉NaN值
    valid_results = results_df.dropna(subset=['mae_raw', 'mae_filtered'])
    
    if len(valid_results) > 0:
        mean_mae_raw = valid_results['mae_raw'].mean()
        mean_mae_filtered = valid_results['mae_filtered'].mean()
        
        print(f"\n有效被试数量: {len(valid_results)}/{len(results_df)}")
        print(f"平均MAE（不滤波）: {mean_mae_raw:.4f} bpm")
        print(f"平均MAE（滤波）  : {mean_mae_filtered:.4f} bpm")
        print(f"MAE改善: {mean_mae_raw - mean_mae_filtered:.4f} bpm")
        
        if mean_mae_filtered < mean_mae_raw:
            print("\n✅ 结论: 滤波后的MAE更低，建议使用滤波")
        else:
            print("\n❌ 结论: 不滤波的MAE更低，建议不使用滤波")
    else:
        print("\n⚠️ 警告: 没有有效的结果数据")
    
    # 保存结果
    output_path = 'filter_comparison_results.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_path}")
    
    return results_df


if __name__ == '__main__':
    print('='*60)
    print('滤波效果对比测试')
    print('='*60)
    compare_filter_effect()

# 带通滤波0.5-2.0
# 0.6
# 平均MAE（不滤波）: 6.6856 bpm
# 平均MAE（滤波）  : 6.6872 bpm
# 0.7
# 平均MAE（不滤波）: 1.5260 bpm
# 平均MAE（滤波）  : 1.5255 bpm
# 0.8
# 平均MAE（不滤波）: 2.3423 bpm
# 平均MAE（滤波）  : 2.3478 bpm


# 以下都是0.7
# 平均MAE（不滤波）: 1.5260 bpm   
# 平均MAE（滤波）  : 1.5255 bpm 带通 
# 平均MAE（滤波）  : 0.9553 bpm 带通+包络 
# 平均MAE（滤波）  : 0.9407 bpm 平滑（smooth_signal + savgol_filter）+ Hilbert包络
# 平均MAE（滤波）  : 1.4480 bpm win后 带通 
# 平均MAE（滤波）  : 1.2564 bpm win后 带通+包络 
# 平均MAE（滤波）  : 1.1398 bpm win后 平滑（smooth_signal + savgol_filter）+ Hilbert包络 

