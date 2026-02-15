import numpy as np
import torch
from hdf5storage import loadmat, savemat
from scipy.stats import zscore
import random
import math
import pandas as pd
from sklearn import preprocessing
import os
import re
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy import signal
import time
from util.fiducial_feature_extraction import *
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data import ConcatDataset
from sklearn.model_selection import KFold


class CustomDataset(Dataset):
    # 定义数据的存储方式和访问接口
    def __init__(self, features, labels, subjects):
        self.features = features
        self.labels = labels
        self.subjects = subjects

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.subjects[idx]


class DataProcessor:
    def __init__(self, datapath, signal_type, training_scheme, params):
        self.datapath = datapath
        self.signal_type = signal_type
        self.training_scheme = training_scheme
        self.batch_size = params['batch_size']
        self.params = params
        self.all_train_loaders = []
        self.all_test_loaders = []
        self.all_fine_tune_loaders = []

        self.fs = 250
        # self.seg_len = 1250
        self.dim_spi = 4  
        self.dim_FF = 9  # change_by_zsy: 原值44，基于J峰的9维FF特征
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.finetune_N = params['finetune_N']
        self.finetune_strategy = params['finetune_strategy']  # 新增加的微调样本选择策略
        self.window_length = 5 * self.fs  # 5秒窗口长度
        self.frame_shift = 2 * self.fs  # 2秒帧移
        self.user_info_df = None  # add_by_zsy: 用于存储从CSV加载的个人信息
        self.pi_csv_path = params['pi_csv_path']  # add_by_zsy: PI CSV文件路径
        self.hr_csv_path = params['hr_csv_path']  # add_by_zsy: 1d预测的心率CSV文件路径
        self.hr_dict = {}  # add_by_zsy: 存储每个被试的心率
        
        # add_by_zsy: 用于统计基于J峰检测的心率MAE
        self.hr_predictions_per_subject = {}  # 每个被试的所有窗口预测心率列表 {user_id: [window1_hr, window2_hr, ...]}
        
        # add_by_zsy: 血压标签归一化相关参数
        self.use_normalize_bp = params['normalize_bp']  # 是否对BP标签进行归一化
        self.normalize_method = params.get('normalize_method', 'minmax')  # 归一化方式: 'minmax' 或 'zscore'
        self.bp_stats = None  # 存储归一化参数，minmax: {'DBP': {'min': x, 'max': y}, ...}，zscore: {'DBP': {'mean': x, 'std': y}, ...}

    def load_data(self):
        """加载MATLAB v7.3格式的数据"""
        
        # add_by_zsy: 读取个人信息CSV文件
        if self.pi_csv_path and os.path.exists(self.pi_csv_path):
            print(f'Loading PI CSV from: {self.pi_csv_path}')
            self.user_info_df = pd.read_csv(self.pi_csv_path, encoding='gbk')
            
            # 性别编码: M->1, F->0
            self.user_info_df['Gender_encoded'] = self.user_info_df['Gender'].map({'M': 1, 'F': 0})
            # 设置ID为索引,便于根据ID查找
            self.user_info_df.set_index('ID', inplace=True)
            print(f'  已加载个人信息CSV，共{len(self.user_info_df)}个用户')
        else:
            if self.pi_csv_path:
                print(f'⚠️ 警告: PI CSV文件不存在: {self.pi_csv_path}')
            print('  未使用个人信息特征')
        
        # add_by_zsy: 读取心率CSV文件
        if self.hr_csv_path and os.path.exists(self.hr_csv_path):
            print(f'Loading HR CSV from: {self.hr_csv_path}')
            hr_df = pd.read_csv(self.hr_csv_path)
            for _, row in hr_df.iterrows():
                self.hr_dict[str(row['subject_id'])] = row['pred_hr']
            print(f'  已加载心率数据，共{len(self.hr_dict)}个用户')
        else:
            if self.hr_csv_path:
                print(f'⚠️ 警告: HR CSV文件不存在: {self.hr_csv_path}')
            print('  未加载心率CSV，将无法计算FF特征')
        
        features_dict = {}
        labels_dict = {}
        all_people_group = []
        ids_dict = set()

        for path in self.datapath:
            print(f'Loading data from: {path}')
            
            # add_by_zsy: 使用 hdf5storage 读取 MATLAB v7.3 格式
            mat = loadmat(path)
            
            # 查找主数据字段（非__开头的字段）
            main_key = None
            for key in mat.keys():
                if not key.startswith('__'):
                    main_key = key
                    print(f'  找到主数据字段: {main_key}')
                    break
            
            if main_key is None:
                raise ValueError(f"未找到有效的数据字段在 {path}")
            
            data_table = mat[main_key]
            
            # add_by_zsy: hdf5storage 返回的是 NumPy 结构数组
            # 需要使用字典风格访问字段: data['field'] 而不是 getattr(data, 'field')
            print(f'  数据类型: {type(data_table)}')
            print(f'  数据形状: {data_table.shape}')
            
            # 检查字段名
            if hasattr(data_table, 'dtype') and data_table.dtype.names:
                print(f'  字段名: {data_table.dtype.names}')
                
                # 提取 ID 和其他字段
                subject_IDs = data_table['ID'].flatten()  # 展平为1D数组
                subject_NUM = len(subject_IDs)
                print(f'  总共 {subject_NUM} 个受试者')
                
                # 提取所有字段数据（除了ID）
                # 从结构体中提取每个受试者的数据
                all_fields_data = []
                for i in range(subject_NUM):
                    subject_data = data_table.flatten()[i]  # 获取第i个受试者
                    all_fields_data.append(subject_data)
                
                # change_by_zsy: 调用修改后的函数处理结构数组格式
                features, labels, ids, _ = self._load_subjects_data_from_struct(
                    all_fields_data, subject_NUM
                )
            print(f'Feature extracting: {path}')

            # 将每个文件中的用户数据添加到字典中，以用户ID为键
            for idx, user_id in enumerate(ids):
                # user_id = user_id[0]
                if user_id not in features_dict:
                    features_dict[user_id] = features[idx]
                    labels_dict[user_id] = labels[idx]
                else:
                    # 如果该用户ID已存在，则将新数据追加到现有数据
                    features_dict[user_id] = torch.cat((features_dict[user_id], features[idx]), dim=0)
                    labels_dict[user_id] = torch.cat((labels_dict[user_id], labels[idx]), dim=0)

                ids_dict.add(user_id)

        # 转换字典为列表
        all_people_features = [features_dict[user_id] for user_id in sorted(ids_dict)]
        all_people_labels = [labels_dict[user_id] for user_id in sorted(ids_dict)]
        all_people_ids = sorted(list(ids_dict))
        
        # add_by_zsy: 如果启用BP归一化，计算全局统计量并应用归一化
        if self.use_normalize_bp:
            method_name = 'Z-score' if self.normalize_method == 'zscore' else 'Min-Max'
            print(f'\n启用血压标签归一化 ({method_name})')
            # 计算归一化统计量
            self.compute_bp_stats(all_people_labels)
            
            # 对所有标签应用归一化
            all_people_labels_normalized = []
            for labels in all_people_labels:
                normalized_labels = self.normalize_bp(labels)
                all_people_labels_normalized.append(normalized_labels)
            all_people_labels = all_people_labels_normalized
            print('血压标签归一化完成\n')

        # NDL = Non-Domain-specific Learning（非领域特定学习）
        # 这些方法负责将数据划分为训练集、测试集和微调集，用于跨用户泛化的交叉验证。
        if self.training_scheme == 'DL':
            self._process_DL(all_people_features, all_people_labels)  # 五折
        elif self.training_scheme == 'NDL':
            if self.params['is_balance_dist']:  
                self._process_NDL2(all_people_features, all_people_labels, all_people_ids, all_people_group) # 分层抽样
            else:  
                self._process_NDL3(all_people_features, all_people_labels, all_people_ids) # 留一法
        elif self.training_scheme == 'SM':
            self._process_SM(all_people_features, all_people_labels)
        else:
            raise ValueError("Invalid training scheme")

    def load_similar_data(self, similar_group):
        mat = loadmat(self.datapath)
        subject_ID = mat['data']['ID'][0]
        subject_NUM = subject_ID.shape[0]

        all_people_features, all_people_labels, all_people_ids, all_people_group = self._load_subjects_data(mat,
                                                                                                            subject_NUM)

        # 根据similar_group,把目标用户的样本都提取出来
        similar_features = []
        similar_labels = []

        # 遍历所有用户，检查是否在 similar_group 中
        for i, user_id in enumerate(all_people_ids):
            if user_id in similar_group:
                # 如果用户在 similar_group 中，添加他们的特征和标签到列表
                # similar_features.append(all_people_features[i])
                # similar_labels.append(all_people_labels[i])

                # 相似群体随机采样10个
                if len(all_people_features[i]) > 10:
                    indices = torch.randperm(len(all_people_features[i]))[:10]
                    sampled_features = all_people_features[i][indices]
                    sampled_labels = all_people_labels[i][indices]
                else:
                    sampled_features = all_people_features[i]
                    sampled_labels = all_people_labels[i]

                similar_features.append(sampled_features)
                similar_labels.append(sampled_labels)

        # 合并所有相似用户群的特征和标签
        combined_features = torch.cat(similar_features, dim=0)
        combined_labels = torch.cat(similar_labels, dim=0)

        return combined_features, combined_labels

    def _get_user_PI(self, user_id):
        """
        add_by_zsy: 根据用户ID从CSV中提取个人信息特征
        
        参数:
            user_id: bcg数据中的用户ID字符串
        
        返回:
            torch.tensor [gender, age, weight, height] (4维)
            如果没有加载CSV或找不到用户，返回默认值[0, 0, 0, 0]
        """
        if self.user_info_df is None:  
            # 如果没有加载CSV，返回默认值
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        
        # 确保user_id是字符串类型
        user_id = str(user_id)
        
        # change_by_zsy: 先检查原始ID是否存在
        if user_id not in self.user_info_df.index:
            # 如果不存在，尝试X01->X10的格式转换
            if user_id.startswith('X01') and len(user_id) == 5:
                alternative_id = 'X10' + user_id[3:]
                if alternative_id in self.user_info_df.index:
                    user_id = alternative_id  # 使用转换后的ID
                else:
                    # 两种格式都找不到，返回默认值
                    print(f'  ⚠️ 警告: 用户 {user_id} (也尝试了{alternative_id}) 不在PI CSV中，使用默认值')
                    return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            else:
                # 不是X0开头或找不到，返回默认值
                print(f'  ⚠️ 警告: 用户 {user_id} 不在PI CSV中，使用默认值')
                return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        
        # 找到用户，提取PI特征
        user_row = self.user_info_df.loc[user_id]
        gender = user_row['Gender_encoded']
        age = user_row['Age']
        height = user_row['Height_cm']
        weight = user_row['Weight_kg']
        
        # 按照原代码中的注释顺序: gender, age, weight, height
        PI = torch.tensor([gender, age, weight, height], dtype=torch.float32)
        return PI

    def _downsample_1000_to_250(self, signal_data):
        """add_by_zsy: 降采样从1000Hz到250Hz"""
        from scipy import signal as sp_signal
        # 降采样因子为4 (1000/250 = 4)
        downsampled = sp_signal.decimate(signal_data, 4, zero_phase=True)
        return downsampled
    
    def _examineFluctuation(self, sbp, dbp):
        # Downsample the data to 1Hz (from 250Hz), taking every 250th sample
        sbp_downsampled = sbp[::250]
        dbp_downsampled = dbp[::250]

        # Check for fluctuations in the downsampled data
        # Check if the difference between consecutive samples is greater than 8
        if np.any(np.abs(np.diff(sbp_downsampled)) > 8) or np.any(np.abs(np.diff(dbp_downsampled)) > 8):
            print('相邻数据变化>8')
            print(sbp_downsampled)
            return True

        # Check if the difference between the max and min values is greater than 10
        if (np.max(sbp_downsampled) - np.min(sbp_downsampled) > 10) or (
                np.max(dbp_downsampled) - np.min(dbp_downsampled) > 10):
            print('5s内最大值-最小值超过10')
            print(sbp_downsampled)
            return True

        # Check if the variance of the data is greater than 8
        if np.std(sbp_downsampled) > 5 or np.std(dbp_downsampled) > 5:
            print('整体方差过大')
            print(sbp_downsampled)
            return True

        # If none of the conditions are met, return False
        return False
    
    def compute_bp_stats(self, all_people_labels):
        """add_by_zsy: 计算全局血压归一化统计量(Min-Max或Z-score)
        
        参数:
            all_people_labels: 列表，每个元素是一个用户的标签张量 (n_samples, 2)
                               第0列是DBP，第1列是SBP
        """
        # 合并所有标签
        all_labels = torch.cat(all_people_labels, dim=0).numpy()
        
        dbp_values = all_labels[:, 0]
        sbp_values = all_labels[:, 1]
        
        if self.normalize_method == 'zscore':
            # Z-score归一化：存储均值和标准差
            self.bp_stats = {
                'DBP': {
                    'mean': float(np.mean(dbp_values)),
                    'std': float(np.std(dbp_values))
                },
                'SBP': {
                    'mean': float(np.mean(sbp_values)),
                    'std': float(np.std(sbp_values))
                }
            }
            print('\n' + '='*60)
            print('血压归一化统计量(Z-score):')
            print(f'  DBP: mean={self.bp_stats["DBP"]["mean"]:.2f}, std={self.bp_stats["DBP"]["std"]:.2f}')
            print(f'  SBP: mean={self.bp_stats["SBP"]["mean"]:.2f}, std={self.bp_stats["SBP"]["std"]:.2f}')
            print('='*60 + '\n')
        else:
            # Min-Max归一化：存储最小值和最大值
            self.bp_stats = {
                'DBP': {
                    'min': float(np.min(dbp_values)),
                    'max': float(np.max(dbp_values))
                },
                'SBP': {
                    'min': float(np.min(sbp_values)),
                    'max': float(np.max(sbp_values))
                }
            }
            print('\n' + '='*60)
            print('血压归一化统计量(Min-Max):')
            print(f'  DBP: min={self.bp_stats["DBP"]["min"]:.2f}, max={self.bp_stats["DBP"]["max"]:.2f}')
            print(f'  SBP: min={self.bp_stats["SBP"]["min"]:.2f}, max={self.bp_stats["SBP"]["max"]:.2f}')
            print('='*60 + '\n')
    
    def normalize_bp(self, labels):
        """add_by_zsy: 使用Min-Max或Z-score方法归一化血压标签
        
        参数:
            labels: numpy array或torch.Tensor, shape (n_samples, 2)
        
        返回:
            归一化后的labels (相同类型)
        """
        if self.bp_stats is None:
            raise ValueError('BP统计量未计算，请先调用compute_bp_stats()')
        
        is_tensor = isinstance(labels, torch.Tensor)
        if is_tensor:
            labels_np = labels.numpy()
        else:
            labels_np = labels
        
        normalized = np.zeros_like(labels_np)
        
        if self.normalize_method == 'zscore':
            # Z-score归一化: (x - mean) / std
            dbp_mean = self.bp_stats['DBP']['mean']
            dbp_std = self.bp_stats['DBP']['std']
            normalized[:, 0] = (labels_np[:, 0] - dbp_mean) / (dbp_std + 1e-8)
            
            sbp_mean = self.bp_stats['SBP']['mean']
            sbp_std = self.bp_stats['SBP']['std']
            normalized[:, 1] = (labels_np[:, 1] - sbp_mean) / (sbp_std + 1e-8)
        else:
            # Min-Max归一化: (x - min) / (max - min)
            dbp_min = self.bp_stats['DBP']['min']
            dbp_max = self.bp_stats['DBP']['max']
            normalized[:, 0] = (labels_np[:, 0] - dbp_min) / (dbp_max - dbp_min + 1e-8)
            
            sbp_min = self.bp_stats['SBP']['min']
            sbp_max = self.bp_stats['SBP']['max']
            normalized[:, 1] = (labels_np[:, 1] - sbp_min) / (sbp_max - sbp_min + 1e-8)
        
        if is_tensor:
            return torch.tensor(normalized, dtype=torch.float32)
        else:
            return normalized
    
    def denormalize_bp(self, labels):
        """add_by_zsy: 反归一化血压标签到原始尺度
        
        参数:
            labels: numpy array或torch.Tensor, shape (n_samples, 2)
                   Min-Max模式：值域 [0, 1]
                   Z-score模式：标准化后的值
        
        返回:
            反归一化后的labels (相同类型)，值域恢复到原始血压范围
        """
        if self.bp_stats is None:
            # 如果没有启用归一化，直接返回原值
            return labels
        
        is_tensor = isinstance(labels, torch.Tensor)
        if is_tensor:
            labels_np = labels.numpy()
        else:
            labels_np = labels
        
        denormalized = np.zeros_like(labels_np)
        
        if self.normalize_method == 'zscore':
            # Z-score反归一化: x * std + mean
            dbp_mean = self.bp_stats['DBP']['mean']
            dbp_std = self.bp_stats['DBP']['std']
            denormalized[:, 0] = labels_np[:, 0] * dbp_std + dbp_mean
            
            sbp_mean = self.bp_stats['SBP']['mean']
            sbp_std = self.bp_stats['SBP']['std']
            denormalized[:, 1] = labels_np[:, 1] * sbp_std + sbp_mean
        else:
            # Min-Max反归一化: x * (max - min) + min
            dbp_min = self.bp_stats['DBP']['min']
            dbp_max = self.bp_stats['DBP']['max']
            denormalized[:, 0] = labels_np[:, 0] * (dbp_max - dbp_min) + dbp_min
            
            sbp_min = self.bp_stats['SBP']['min']
            sbp_max = self.bp_stats['SBP']['max']
            denormalized[:, 1] = labels_np[:, 1] * (sbp_max - sbp_min) + sbp_min
        
        if is_tensor:
            return torch.tensor(denormalized, dtype=torch.float32)
        else:
            return denormalized

    def _load_subjects_data_from_table(self, subject_IDs, filtered_data, subject_NUM):
        #提取每个受试者的原始 BCG 信号、血压标签、个人信息（PI）和特征工程特征（FF）
        # 数据结构：
        # BCG 信号：1250 维原始信号
        # 个人信息（PI）：性别、年龄、身高、体重（4 维）
        # 特征工程特征（FF）：基于 BCG 信号的手工特征（44 维）
        # Last BP：上一次测量的血压值（2 维）
        # 标签：DBP 和 SBP
        """add_by_zsy: 从 MATLAB table 格式加载数据
        
        根据用户截图，FilteredData 有 15 列：
        列索引：
          3: Film0 (使用此列作为 BCG 信号，用户指定)
          11: SBP (收缩压)
          12: DBP (舒张压)
        """
        all_people_features = []
        all_people_labels = []
        all_people_ids = []  # 新增一个列表来存储所有用户的ID
        all_people_group = []
        window_length = self.window_length  # 5秒窗口长度
        frame_shift = self.frame_shift  # 2秒帧移

        for user_index in range(subject_NUM):
            user_id = subject_IDs[user_index]
            print(f'loading {user_index}: {user_id}')
            
            # 这里的+2是留给上一时间步的收缩压和舒张压值的
            person_features = torch.empty((0, self.window_length + self.dim_spi + self.dim_FF + 2), dtype=torch.float32)
            person_labels = torch.empty((0, 2), dtype=torch.float32)

            # add_by_zsy: 从 MATLAB table 的 FilteredData 中提取数据
            user_data = filtered_data[user_index]  # N×15 矩阵
            
            # 根据用户要求：使用 Film0 (列索引 3) 作为 BCG 信号
            BCG = user_data[:, 3].reshape(-1)  # BCG 列
            sbp = user_data[:, 11].reshape(-1)  # SBP 列
            dbp = user_data[:, 12].reshape(-1)  # DBP 列
            
            # add_by_zsy: 降采样从1000Hz到250Hz
            print(f'  原始长度: BCG={len(BCG)}, SBP={len(sbp)}, DBP={len(dbp)}')
            BCG = self._downsample_1000_to_250(BCG)
            sbp = self._downsample_1000_to_250(sbp)
            dbp = self._downsample_1000_to_250(dbp)
            print(f'  降采样后: BCG={len(BCG)}, SBP={len(sbp)}, DBP={len(dbp)}')

            # delete_by_zsy: 
            '''
            gender = mat['preprocessed_Data']['gender'][0][user_index][0][0]
            age = mat['preprocessed_Data']['age'][0][user_index][0][0]
            height = mat['preprocessed_Data']['height'][0][user_index][0][0]
            weight = mat['preprocessed_Data']['weight'][0][user_index][0][0]

            PI = torch.tensor(np.array((gender, age, weight, height)), dtype=torch.float32)

            H_locs = mat['preprocessed_Data']['H_locs'][0][user_index].astype(int).reshape(-1)
            I_locs = mat['preprocessed_Data']['I_locs'][0][user_index].astype(int).reshape(-1)
            J_locs = mat['preprocessed_Data']['J_locs'][0][user_index].astype(int).reshape(-1)
            K_locs = mat['preprocessed_Data']['K_locs'][0][user_index].astype(int).reshape(-1)
            L_locs = mat['preprocessed_Data']['L_locs'][0][user_index].astype(int).reshape(-1)
            '''

            # 初始化窗口的起始和结束点
            window_start = 0
            window_end = window_start + window_length
            last_window_DBP = 0  # np.mean(dbp[window_start:window_end])  # 上一个时间步的舒张压，初始化
            last_window_SBP = 0  # np.mean(sbp[window_start:window_end])  # 上一个时间步的收缩压，初始化
            
            while window_end <= len(BCG):
                # del_by_zsy: 移除 HIJKL 相关代码(无此数据)
                '''
                # 获取当前窗口内的J_locs
                window_J_indices = np.where((J_locs >= window_start) & (J_locs < window_end))[0]

                if len(window_J_indices) == 0:
                    # 如果当前窗口内没有J_locs，跳过这个窗口
                    window_start += frame_shift
                    window_end = window_start + window_length
                    continue

                # if self._examineFluctuation(sbp[window_start:window_end],dbp[window_start:window_end]):
                #     window_start += frame_shift
                #     window_end = window_start + window_length
                #     continue

                # 根据J_locs的索引获取窗口内的所有基准点位置
                window_H_locs = H_locs[window_J_indices]
                window_I_locs = I_locs[window_J_indices]
                window_J_locs = J_locs[window_J_indices]
                window_K_locs = K_locs[window_J_indices]
                window_L_locs = L_locs[window_J_indices]

                计算当前窗口内的血压平均值
                window_DBP = np.mean(dbp[window_J_indices])
                window_SBP = np.mean(sbp[window_J_indices])
                '''

                # del_by_zsy: 删除 FF 特征计算(无 HIJKL 数据)
                '''
                # 计算当前窗口内的基准点特征
                window_features = compute_features(BCG, window_H_locs, window_I_locs, window_J_locs,
                                                   window_K_locs, window_L_locs)

                # 将所有特征平均值转换为一个长向量
                time_interval = np.vstack(list(window_features['Time Interval'].values()))
                time_ratio = np.vstack(list(window_features['Time Ratio'].values()))
                extremum = np.vstack(list(window_features['Extremum'].values()))
                displacement = np.vstack(list(window_features['Displacement'].values()))
                auc = np.vstack(list(window_features['AUC'].values()))
                tmp = np.vstack((time_interval, time_ratio, extremum, displacement, auc))
                # print(np.isnan(tmp).any())
                FF = np.mean(tmp, axis=1)
                '''

                # add_by_zsy: 简化窗口切分，不再依赖J_locs
                window_DBP = np.mean(dbp[window_start:window_end])
                window_SBP = np.mean(sbp[window_start:window_end])

                # 将BCG段、个人信息和特征合并为一个样本
                seg = BCG[window_start:window_end]
                seg_tensor = torch.tensor(seg, dtype=torch.float32)
                # FF_tensor = torch.tensor(FF, dtype=torch.float32)  # del_by_zsy
                LASTBP_tensor = torch.tensor([last_window_DBP, last_window_SBP], dtype=torch.float32)
                
                # add_by_zsy: 不使用 PI 和 FF 特征，只使用 BCG + Last_BP
                # sample_features = torch.cat((seg_tensor, PI, FF_tensor, LASTBP_tensor))
                sample_features = torch.cat((seg_tensor, LASTBP_tensor)) 

                # 将当前样本添加到个人的特征和标签中间变量
                person_features = torch.cat((person_features, sample_features.reshape(1, -1)), dim=0)  # 单个人所有窗口sample_features的拼接
                person_labels = torch.cat(
                    (person_labels, torch.tensor([[window_DBP, window_SBP]], dtype=torch.float32)), dim=0)  # 单个人所有窗口标签的拼接

                # 更新窗口的起始和结束点
                window_start += frame_shift
                window_end = window_start + window_length

                # 更新上一次血压值  by_zsy  main的配置里未启用
                last_window_DBP = window_DBP   
                last_window_SBP = window_SBP


            # 添加样本到特征和标签列表
            all_people_features.append(person_features)
            all_people_labels.append(person_labels)

            # 添加用户ID到列表 
            # change_by_zsy: 
            # all_people_ids.append(mat['preprocessed_Data']['ID'][0][user_index][0])
            all_people_ids.append(user_id)  
            # all_people_group.append(mat['data']['BCG_Group'][0][user_index][0])

        return all_people_features, all_people_labels, all_people_ids, all_people_group

    def _load_subjects_data_from_struct(self, all_fields_data, subject_NUM):
        """add_by_zsy: 根据 hdf5storage 返回的结构数组格式来加载数据
        
        参数:
            all_fields_data: 列表，每个元素是一个受试者的结构数据
            subject_NUM: 受试者数量
        
        返回:
            all_people_features, all_people_labels, all_people_ids, all_people_group
        """
        all_people_features = []
        all_people_labels = []
        all_people_ids = []
        all_people_group = []
        window_length = self.window_length  # 5秒窗口长度
        frame_shift = self.frame_shift  # 2秒帧移

        for user_index in range(subject_NUM):
            subject_data = all_fields_data[user_index]
            
            # 提取 ID
            user_id = subject_data['ID']
            if isinstance(user_id, np.ndarray):
                user_id = user_id.flatten()[0] if user_id.size > 0 else str(user_index)
            # 如果是字节串，解码为字符串
            if isinstance(user_id, bytes):
                user_id = user_id.decode('utf-8')
            # 如果是 numpy 字符串数组
            elif hasattr(user_id, 'item'):
                user_id = str(user_id.item())
            else:
                user_id = str(user_id)
            
            # print(f'loading {user_index}: {user_id}')
            
            # add_by_zsy: 提取该用户的个人信息特征
            PI = self._get_user_PI(user_id)
            
            # 初始化特征和标签张量
            person_features = torch.empty((0, self.window_length + self.dim_spi + self.dim_FF + 2), dtype=torch.float32)
            person_labels = torch.empty((0, 2), dtype=torch.float32)

            # 从结构数组中提取数据字段
            # 根据flatten_nested_table.m的逻辑，所有 FilteredData table 的列都被展开为 struct 的字段
            # 需要找到 BCG、SBP、DBP 字段
            # 1. BCG 是 subject_data 除了ID之外的第4列 Film0
            # 2. SBP、DBP 来自第12列 reBAP，窗口内的最大值是SBP，最小值是DBP
            
            # 检查可用字段（用于调试）
            # available_fields = [name for name in subject_data.dtype.names if name != 'ID']
            # print(f'  可用字段: {available_fields}')
            
            # 检查必要字段是否存在
            if 'Film0' not in subject_data.dtype.names or 'reBAP' not in subject_data.dtype.names:
                print(f'  ⚠️ 警告: 受试者 {user_id} 缺少必要字段')
                print(f'    Film0: {"存在" if "Film0" in subject_data.dtype.names else "缺失"}')
                print(f'    reBAP: {"存在" if "reBAP" in subject_data.dtype.names else "缺失"}')
                continue
            
            # 直接提取 BCG 和 reBAP 数据
            BCG = subject_data['Film0'].flatten()
            reBAP = subject_data['reBAP'].flatten()
            # print(f'  使用 Film0 作为 BCG 信号, 长度={len(BCG)}')
            # print(f'  使用 reBAP 计算血压, 长度={len(reBAP)}')

            # add_by_zsy: 在窗口切分前，检查并裁剪reBAP中的NaN数据
            # 找到reBAP中第一个NaN的位置
            nan_mask = np.isnan(reBAP)
            if nan_mask.any():
                # 找到第一个NaN的索引
                first_nan_idx = np.where(nan_mask)[0][0]
                # 裁剪掉包含NaN的部分
                reBAP = reBAP[:first_nan_idx]
                # 同步裁剪BCG
                BCG = BCG[:first_nan_idx]
                if user_index < 3:  # 只为前几个用户打印信息
                    print(f'  用户{user_id}: 裁剪掉{len(nan_mask)-first_nan_idx}个NaN样本，保留{first_nan_idx}个有效样本')
            
            # 降采样从1000Hz到250Hz
            # print(f'  原始长度: BCG={len(BCG)}')
            BCG = self._downsample_1000_to_250(BCG)
            # print(f'  降采样后: BCG={len(BCG)}')

            # add_by_zsy: 获取该用户的心率用于J峰检测
            # 支持ID映射：X0132 -> X1032
            hr_user_id = user_id
            if user_id == 'X0132':
                hr_user_id = 'X1032'
            
            user_hr = None
            if hr_user_id in self.hr_dict:
                user_hr = self.hr_dict[hr_user_id]
            else:
                # 如果找不到心率，设置默认值并发出警告
                user_hr = 70.0  # 默认心率
                if user_index < 3:  # 只为前几个用户打印警告
                    print(f'  ⚠️ 警告: 用户{user_id}未找到心率数据，使用默认值{user_hr} bpm')
            
            # add_by_zsy: 应用最优滤波方案（划分窗口前进行平滑 + Hilbert包络）用于J峰检测
            # 这是在阶段1测试中确定的最优方案（MAE=0.9407 bpm）
            from scipy.signal import savgol_filter
            import scipy.stats
            
            # 预处理：去除NaN、去趋势、平滑
            BCG_for_peak = np.nan_to_num(BCG, nan=np.nanmean(BCG))
            BCG_for_peak = signal.detrend(BCG_for_peak, type='constant')
            
            # Savitzky-Golay平滑
            window_len_sg = self.fs // 10  # 采样率的1/10
            if window_len_sg % 2 == 0 or window_len_sg == 0:
                window_len_sg += 1
            polyorder_sg = 3
            if window_len_sg <= polyorder_sg:
                polyorder_sg = window_len_sg - 1
            BCG_for_peak = savgol_filter(BCG_for_peak, window_length=window_len_sg, polyorder=polyorder_sg)
            BCG_for_peak = savgol_filter(BCG_for_peak, 5, 3, mode='nearest')
            
            # Hilbert包络
            BCG_filtered = np.abs(scipy.signal.hilbert(BCG_for_peak))

            # 窗口切分
            window_start = 0   
            window_end = window_start + window_length
            last_window_DBP = 0
            last_window_SBP = 0
            
            # add_by_zsy: 添加调试标志，第一次打印reBAP范围
            # if window_start == 0 and user_index == 0:
            #     print(f'  【调试】reBAP原始值范围: min={np.min(reBAP):.4f}, max={np.max(reBAP):.4f}')
            
            
            while window_end <= len(BCG):
                # 计算当前窗口内的血压值
                # 根据用户要求：窗口内 reBAP 的最大值是 SBP，最小值是 DBP
                # 注意：BCG已降采样到250Hz，reBAP保持1000Hz，所以索引需要乘以4
                reBAP_window_start = window_start * 4
                reBAP_window_end = window_end * 4
                window_reBAP = reBAP[reBAP_window_start:reBAP_window_end]
                window_SBP = np.max(window_reBAP)*100
                window_DBP = np.min(window_reBAP)*100
                
                # add_by_zsy: 添加数据验证，检查是否包含异常值
                # if window_start == 0 and user_index == 0:
                #     print(f'  【调试】第一个窗口血压值: SBP={window_SBP:.2f}, DBP={window_DBP:.2f}')
                
                # 验证血压值是否在合理范围内
                if np.isnan(window_SBP) or np.isnan(window_DBP) or np.isinf(window_SBP) or np.isinf(window_DBP):
                    print(f'  ⚠️ 警告: 用户{user_id}窗口{window_start}出现异常血压值 SBP={window_SBP}, DBP={window_DBP}')
                    window_start += frame_shift
                    window_end = window_start + window_length
                    continue
                
                # 检查血压值是否在生理合理范围内 (SBP: 80-200, DBP: 40-130)
                if window_SBP < 80 or window_SBP > 200 or window_DBP < 40 or window_DBP > 130:
                    print(f'  ⚠️ 警告: 用户{user_id}窗口{window_start}血压值超出正常范围 SBP={window_SBP:.2f}, DBP={window_DBP:.2f}')
                    window_start += frame_shift
                    window_end = window_start + window_length
                    continue

                # 提取BCG段
                seg = BCG[window_start:window_end].copy()  # 添加.copy()确保数组连续
                seg_tensor = torch.tensor(seg, dtype=torch.float32)
                LASTBP_tensor = torch.tensor([last_window_DBP, last_window_SBP], dtype=torch.float32)
                
                # add_by_zsy: 计算FF特征（基于J峰）
                FF_tensor = torch.zeros(self.dim_FF, dtype=torch.float32)  # 默认零向量
                
                if self.dim_FF > 0 and user_hr is not None:
                    # 在滤波后的窗口中检测J峰
                    window_bcg_filtered = BCG_filtered[window_start:window_end]
                    
                    # 计算distance参数（基于心率）
                    # 使用阶段1确定的最优参数：0.7 × (60 / hr) × fs
                    distance = int(0.7 * (60 / user_hr) * self.fs)
                    
                    try:
                        from scipy.signal import find_peaks
                        peaks, _ = find_peaks(window_bcg_filtered, distance=distance)
                        
                        # 如果找到足够的峰值，计算FF特征
                        if len(peaks) >= 2:
                            FF_features = compute_features_from_j_peaks(seg, peaks, fs=self.fs)
                            FF_tensor = torch.tensor(FF_features, dtype=torch.float32)
                            
                            # add_by_zsy: 记录基于J峰的心率（用于后续MAE统计）
                            # FF_features[6]是平均心率
                            detected_hr = FF_features[6]
                            
                            # 初始化该用户的列表
                            if user_id not in self.hr_predictions_per_subject:
                                self.hr_predictions_per_subject[user_id] = []
                            
                            # 只记录预测心率，不记录误差（MAE统计时再计算）
                            self.hr_predictions_per_subject[user_id].append(detected_hr)
                    except Exception as e:
                        # 如果特征计算失败，使用零向量
                        if user_index < 3:  # 只为前几个用户打印错误
                            print(f'  ⚠️ FF特征计算失败: {e}')
                        pass
                
                # change_by_zsy: 合并特征: BCG + PI + FF + Last_BP
                if self.dim_spi > 0 and self.user_info_df is not None:
                    if self.dim_FF > 0:
                        sample_features = torch.cat((seg_tensor, PI, FF_tensor, LASTBP_tensor))
                    else:
                        sample_features = torch.cat((seg_tensor, PI, LASTBP_tensor))
                else:
                    if self.dim_FF > 0:
                        sample_features = torch.cat((seg_tensor, FF_tensor, LASTBP_tensor))
                    else:
                        sample_features = torch.cat((seg_tensor, LASTBP_tensor))

                # 添加到个人特征和标签
                person_features = torch.cat((person_features, sample_features.reshape(1, -1)), dim=0)
                person_labels = torch.cat(
                    (person_labels, torch.tensor([[window_DBP, window_SBP]], dtype=torch.float32)), dim=0)

                # 更新窗口
                window_start += frame_shift
                window_end = window_start + window_length

                # 更新上一次血压值
                last_window_DBP = window_DBP   
                last_window_SBP = window_SBP


            # 添加到列表
            all_people_features.append(person_features)
            all_people_labels.append(person_labels)
            all_people_ids.append(user_id)

        return all_people_features, all_people_labels, all_people_ids, all_people_group


    def _process_DL(self, all_people_features, all_people_labels):
        # 存储五折交叉验证的结果
        fold_train_features = [torch.empty((0, all_people_features[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_test_features = [torch.empty((0, all_people_features[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_train_labels = [torch.empty((0, all_people_labels[0].shape[1]), dtype=torch.float32) for _ in range(5)]
        fold_test_labels = [torch.empty((0, all_people_labels[0].shape[1]), dtype=torch.float32) for _ in range(5)]

        for user_features, user_labels in zip(all_people_features, all_people_labels):
            for fold_idx, (train_idx, test_idx) in enumerate(self.kfold.split(user_features)):
                train_features = user_features[train_idx]
                train_labels = user_labels[train_idx]
                test_features = user_features[test_idx]
                test_labels = user_labels[test_idx]

                fold_train_features[fold_idx] = torch.cat((fold_train_features[fold_idx], train_features), dim=0)
                fold_train_labels[fold_idx] = torch.cat((fold_train_labels[fold_idx], train_labels), dim=0)
                fold_test_features[fold_idx] = torch.cat((fold_test_features[fold_idx], test_features), dim=0)
                fold_test_labels[fold_idx] = torch.cat((fold_test_labels[fold_idx], test_labels), dim=0)

        # 创建整体五折训练和测试Dataloader
        for train_features, train_labels, test_features, test_labels in zip(fold_train_features, fold_train_labels,
                                                                            fold_test_features, fold_test_labels):
            train_dataset = CustomDataset(train_features, train_labels)
            test_dataset = CustomDataset(test_features, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)

    def _process_NDL(self, all_people_features, all_people_labels, all_people_ids):
        # 所有用户 → KFold(5折) → 每个fold:
        # ├─ 训练用户（80%）→ 训练集
        # └─ 测试用户（20%）→ 微调集（前N个样本）+ 测试集（剩余样本）
        all_features = torch.cat(all_people_features, dim=0)
        all_labels = torch.cat(all_people_labels, dim=0)
        subjects = np.concatenate(
            [[id] * len(features) for id, features in zip(all_people_ids, all_people_features)])  # 使用用户ID而不是数字索引

        unique_ids = np.array(all_people_ids)
        for train_ids, test_ids in self.kfold.split(unique_ids):
            # 使用用户ID来创建训练和测试的索引
            train_idx = np.isin(subjects, unique_ids[train_ids])
            test_idx = np.isin(subjects, unique_ids[test_ids])

            train_features = all_features[train_idx]
            train_labels = all_labels[train_idx]
            test_features = all_features[test_idx]
            test_labels = all_labels[test_idx]
            test_subjects = subjects[test_idx]

            fine_tune_features_list = []
            fine_tune_labels_list = []
            fine_tune_subjects_list = []
            remaining_test_features_list = []
            remaining_test_labels_list = []
            remaining_test_subjects_list = []

            unique_test_subjects = np.unique(test_subjects)
            for subject in unique_test_subjects:
                subject_indices = np.where(test_subjects == subject)[0]
                if len(subject_indices) >= self.finetune_N:
                    # 新增的startEnd策略
                    if self.finetune_strategy == 'startEnd':
                        half_n = self.finetune_N // 2
                        start_indices = subject_indices[:half_n]  # 取前finetune_N/2个样本
                        end_indices = subject_indices[-half_n:]  # 取后finetune_N/2个样本
                        fine_tune_indices = np.concatenate((start_indices, end_indices))
                    elif self.finetune_strategy == 'random':
                        fine_tune_indices = np.random.choice(subject_indices, self.finetune_N, replace=False)
                    elif self.finetune_strategy == 'sequential':
                        fine_tune_indices = subject_indices[:self.finetune_N]
                    else:
                        raise ValueError("Invalid finetune strategy")

                    remaining_indices = np.setdiff1d(subject_indices, fine_tune_indices)

                    fine_tune_features_list.append(test_features[fine_tune_indices])
                    fine_tune_labels_list.append(test_labels[fine_tune_indices])
                    fine_tune_subjects_list.append(test_subjects[fine_tune_indices])

                    remaining_test_features_list.append(test_features[remaining_indices])
                    remaining_test_labels_list.append(test_labels[remaining_indices])
                    remaining_test_subjects_list.append(test_subjects[remaining_indices])
                else:
                    raise ValueError(f"Subject {subject} does not have enough samples for finetuning")

            fine_tune_features = torch.cat(fine_tune_features_list, dim=0)
            fine_tune_labels = torch.cat(fine_tune_labels_list, dim=0)
            fine_tune_subjects = np.concatenate(fine_tune_subjects_list)

            remaining_test_features = torch.cat(remaining_test_features_list, dim=0)
            remaining_test_labels = torch.cat(remaining_test_labels_list, dim=0)
            remaining_test_subjects = np.concatenate(remaining_test_subjects_list)

            # 创建微调数据集的 DataLoader
            fine_tune_dataset = CustomDataset(fine_tune_features, fine_tune_labels, fine_tune_subjects)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=self.batch_size, shuffle=False)
            self.all_fine_tune_loaders.append(fine_tune_loader)

            # 创建训练和测试数据集的 DataLoader
            train_dataset = CustomDataset(train_features, train_labels, subjects[train_idx])
            test_dataset = CustomDataset(remaining_test_features, remaining_test_labels, remaining_test_subjects)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)

    def _process_NDL2(self, all_people_features, all_people_labels, all_people_ids, all_people_groups):
        # 在 _process_NDL() 基础上增加了 分层抽样（Stratified Sampling）
        # 根据 all_people_groups 将用户分为不同组（如健康人、高血压患者等）
        # 在每个组内分别进行 K-Fold 交叉验证
        # 保证训练集和测试集中各组的比例一致
        
        # 初始化用于存储训练、测试和微调数据的结构
        fold_train_features = [[] for _ in range(5)]
        fold_test_features = [[] for _ in range(5)]
        fold_train_labels = [[] for _ in range(5)]
        fold_test_labels = [[] for _ in range(5)]
        fold_fine_tune_features = [[] for _ in range(5)]
        fold_fine_tune_labels = [[] for _ in range(5)]
        fold_fine_tune_subjects = [[] for _ in range(5)]
        fold_test_subjects = [[] for _ in range(5)]
        fold_train_subjects = [[] for _ in range(5)]
        # 分层抽样
        for group in [0, 1, 2, 3]:
            group_indices = [i for i, g in enumerate(all_people_groups) if g == group]
            group_features = [all_people_features[i] for i in group_indices]
            group_labels = [all_people_labels[i] for i in group_indices]
            group_subjects = [all_people_ids[i] for i in group_indices]

            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(group_features)):
                # 训练集
                fold_train_features[fold_idx].extend([group_features[i] for i in train_idx])
                fold_train_labels[fold_idx].extend([group_labels[i] for i in train_idx])
                fold_train_subjects[fold_idx].extend(
                    [np.array([group_subjects[i]] * len(group_features[i])) for i in train_idx])

                # 测试集和微调集
                test_and_fine_tune_features = [group_features[i] for i in test_idx]
                test_and_fine_tune_labels = [group_labels[i] for i in test_idx]
                test_and_fine_tune_subjects = [group_subjects[i] for i in test_idx]
                test_and_fine_tune_subjects = np.array(test_and_fine_tune_subjects)

                fine_tune_indices = []
                remaining_test_indices = []
                for i, subject_index in enumerate(test_and_fine_tune_subjects):
                    N = len(test_and_fine_tune_features[i])
                    if N >= self.finetune_N:
                        if self.finetune_strategy == 'random':
                            selected_indices = random.sample(range(N), self.finetune_N)
                        elif self.finetune_strategy == 'sequential':
                            selected_indices = list(range(self.finetune_N))
                        elif self.finetune_strategy == 'startEnd':
                            half_n = self.finetune_N // 2
                            selected_indices = list(range(half_n)) + list(range(N - half_n, len(N)))
                        else:
                            raise ValueError("Invalid finetune strategy")

                        fine_tune_features = [test_and_fine_tune_features[i][index] for index in selected_indices]
                        fine_tune_labels = [test_and_fine_tune_labels[i][index] for index in selected_indices]
                        fine_tune_features = torch.vstack(fine_tune_features)
                        fine_tune_labels = torch.vstack(fine_tune_labels)
                        fine_tune_subjects = np.array([subject_index] * self.finetune_N)

                        fold_fine_tune_features[fold_idx].extend([fine_tune_features])
                        fold_fine_tune_labels[fold_idx].extend([fine_tune_labels])
                        fold_fine_tune_subjects[fold_idx].extend([fine_tune_subjects])

                        remaining_indices = [i for i in range(N) if i not in selected_indices]
                        remaining_features = [test_and_fine_tune_features[i][index] for index in remaining_indices]
                        remaining_labels = [test_and_fine_tune_labels[i][index] for index in remaining_indices]
                        remaining_features = torch.vstack(remaining_features)
                        remaining_labels = torch.vstack(remaining_labels)
                        remaining_subjects = np.array([subject_index] * (N - self.finetune_N))

                        fold_test_features[fold_idx].extend([remaining_features])
                        fold_test_labels[fold_idx].extend([remaining_labels])
                        fold_test_subjects[fold_idx].extend([remaining_subjects])
                    else:
                        raise ValueError(f"User with index {subject_index} does not have enough samples for finetuning")

                # fold_fine_tune_features[fold_idx].extend([test_and_fine_tune_features[i] for i in fine_tune_indices])
                # fold_fine_tune_labels[fold_idx].extend([test_and_fine_tune_labels[i] for i in fine_tune_indices])
                # fold_test_features[fold_idx].extend([test_and_fine_tune_features[i] for i in remaining_test_indices])
                # fold_test_labels[fold_idx].extend([test_and_fine_tune_labels[i] for i in remaining_test_indices])

        # 创建 DataLoader
        self.all_train_loaders = []
        self.all_test_loaders = []
        self.all_fine_tune_loaders = []
        for i in range(5):
            train_dataset = CustomDataset(torch.cat(fold_train_features[i]), torch.cat(fold_train_labels[i]),
                                          np.concatenate(fold_train_subjects[i]))
            test_dataset = CustomDataset(torch.cat(fold_test_features[i]), torch.cat(fold_test_labels[i]),
                                         np.concatenate(fold_test_subjects[i]))
            fine_tune_dataset = CustomDataset(torch.cat(fold_fine_tune_features[i]),
                                              torch.cat(fold_fine_tune_labels[i]),
                                              np.concatenate(fold_fine_tune_subjects[i]))

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)
            self.all_fine_tune_loaders.append(fine_tune_loader)

    def _process_NDL3(self, all_people_features, all_people_labels, all_people_ids):
        # 使用 Leave-One-Out (LOO) 交叉验证，每次留一个用户作为测试集，其他所有用户作为训练集
        # 对于测试用户：前 finetune_N 个样本用于微调，剩余样本用于测试
        # 适用于个性化场景：评估模型对全新用户的泛化能力
        all_features = torch.cat(all_people_features, dim=0)
        all_labels = torch.cat(all_people_labels, dim=0)
        subjects = np.concatenate(
            [[id] * len(features) for id, features in zip(all_people_ids, all_people_features)])  # 使用用户ID而不是数字索引

        unique_ids = np.array(all_people_ids)
        for loo_id in unique_ids:
            # 留一个样本作为测试集
            test_idx = np.where(subjects == loo_id)[0]
            train_idx = np.where(subjects != loo_id)[0]

            train_features = all_features[train_idx]
            train_labels = all_labels[train_idx]
            test_features = all_features[test_idx]
            test_labels = all_labels[test_idx]
            test_subjects = subjects[test_idx]

            fine_tune_features_list = []
            fine_tune_labels_list = []
            fine_tune_subjects_list = []
            remaining_test_features_list = []
            remaining_test_labels_list = []
            remaining_test_subjects_list = []

            unique_test_subjects = np.unique(test_subjects)
            for subject in unique_test_subjects:
                subject_indices = np.where(test_subjects == subject)[0]
                if len(subject_indices) >= self.finetune_N:
                    # 新增的startEnd策略
                    if self.finetune_strategy == 'startEnd':
                        half_n = self.finetune_N // 2
                        start_indices = subject_indices[:half_n]  # 取前finetune_N/2个样本
                        end_indices = subject_indices[-half_n:]  # 取后finetune_N/2个样本
                        fine_tune_indices = np.concatenate((start_indices, end_indices))
                    elif self.finetune_strategy == 'random':
                        fine_tune_indices = np.random.choice(subject_indices, self.finetune_N, replace=False)
                    elif self.finetune_strategy == 'sequential':  # 使用
                        fine_tune_indices = subject_indices[:self.finetune_N]
                    else:
                        raise ValueError("Invalid finetune strategy")

                    remaining_indices = np.setdiff1d(subject_indices, fine_tune_indices)

                    fine_tune_features_list.append(test_features[fine_tune_indices])
                    fine_tune_labels_list.append(test_labels[fine_tune_indices])
                    fine_tune_subjects_list.append(test_subjects[fine_tune_indices])

                    remaining_test_features_list.append(test_features[remaining_indices])
                    remaining_test_labels_list.append(test_labels[remaining_indices])
                    remaining_test_subjects_list.append(test_subjects[remaining_indices])
                else:
                    raise ValueError(f"Subject {subject} does not have enough samples for finetuning")

            fine_tune_features = torch.cat(fine_tune_features_list, dim=0)
            fine_tune_labels = torch.cat(fine_tune_labels_list, dim=0)
            fine_tune_subjects = np.concatenate(fine_tune_subjects_list)

            remaining_test_features = torch.cat(remaining_test_features_list, dim=0)
            remaining_test_labels = torch.cat(remaining_test_labels_list, dim=0)
            remaining_test_subjects = np.concatenate(remaining_test_subjects_list)

            # 创建微调数据集的 DataLoader
            fine_tune_dataset = CustomDataset(fine_tune_features, fine_tune_labels, fine_tune_subjects)
            fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=self.batch_size, shuffle=False) # 将 Dataset 转换为可迭代的批次数据，用于训练
            self.all_fine_tune_loaders.append(fine_tune_loader)

            # 创建训练和测试数据集的 DataLoader
            train_dataset = CustomDataset(train_features, train_labels, subjects[train_idx])
            test_dataset = CustomDataset(remaining_test_features, remaining_test_labels, remaining_test_subjects)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            self.all_train_loaders.append(train_loader)
            self.all_test_loaders.append(test_loader)

    def _process_SM(self, all_people_features, all_people_labels):
        # ... 处理SM训练方案 ...
        pass
    
    def print_hr_mae_statistics(self):
        """
        add_by_zsy: 打印基于J峰检测的心率MAE统计信息
        使用与test_filter_effect.py相同的方式：先求所有窗口的平均心率，再与真实心率计算误差
        """
        if not self.hr_predictions_per_subject:
            print('\n' + '='*60)
            print('【心率检测统计】无心率检测数据')
            print('='*60 + '\n')
            return
        
        print('\n' + '='*60)
        print('【基于J峰检测的心率统计】')
        print('='*60)
        
        all_subject_maes = []  # 每个被试的MAE
        
        for user_id in sorted(self.hr_predictions_per_subject.keys()):
            predictions = self.hr_predictions_per_subject[user_id]
            
            # 查找真实心率（处理ID映射）
            hr_user_id = 'X1032' if user_id == 'X0132' else user_id
            
            if predictions and hr_user_id in self.hr_dict:
                # change_by_zsy: 改为test_filter_effect.py的方式
                # 先求所有窗口的平均预测心率
                avg_pred_hr = np.mean(predictions)
                true_hr = self.hr_dict[hr_user_id]
                
                # 再计算平均心率与真实心率的误差
                subject_mae = abs(avg_pred_hr - true_hr)
                all_subject_maes.append(subject_mae)
                
                # print(f'{user_id}: 真实HR={true_hr:.2f}, 检测HR={avg_pred_hr:.2f}, MAE={subject_mae:.2f} bpm (窗口数={len(predictions)})')
        
        # 总体统计
        if all_subject_maes:
            overall_mae = np.mean(all_subject_maes)
            print('='*60)
            print(f'总体: 平均HR MAE = {overall_mae:.4f} bpm ({len(self.hr_predictions_per_subject)}个被试)')
            print(f'说明: 此MAE计算方式与test_filter_effect.py一致（先求窗口平均再计算误差）')
            print('='*60 + '\n')

    def get_loaders(self):
        return self.all_train_loaders, self.all_test_loaders, self.all_fine_tune_loaders


if __name__ == '__main__':
    datapath = '../../data/s3-1_update_v3.mat'
    # datapath = '../data/night-up_35p_FF_short3.mat'
    signal_type = 'day'
    training_scheme = 'NDL'
    batch_size = 64
    finetune_N = 10
    finetune_strategy = 'sequential'  # random

    data_processor = DataProcessor(datapath, signal_type, training_scheme, batch_size, finetune_N, finetune_strategy)
    data_processor.load_data()
    train_loaders, test_loaders, fine_tune_loaders = data_processor.get_loaders()
    print("Done")
