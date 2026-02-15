# -*- coding: utf-8 -*-
"""
测试PI功能是否正确启用
"""
import sys
sys.path.insert(0, '.')

from util.data_preprocessing2 import *

# 测试配置
datapath = ['d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/Preprocessed_Database_flat.mat']
signal_type = 'day'
training_scheme = 'NDL'
params = {
    'batch_size': 32,
    'finetune_N': 3,
    'training_scheme': 'NDL',
    'finetune_strategy': 'sequential',
    'is_balance_dist': 0,
    'use_PI': True,
    'pi_csv_path': 'd:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/introduction.csv'
}

print('='*60)
print('PI功能启用测试')
print('='*60)

# 创建DataProcessor
print('\n1. 创建DataProcessor...')
processor = DataProcessor(datapath, signal_type, training_scheme, params)

# 加载数据
print('\n2. 加载数据...')
processor.load_data()

# 获取loaders
print('\n3. 获取DataLoaders...')
train_loaders, test_loaders, fine_tune_loaders = processor.get_loaders()

# 检查第一个batch的特征维度
print('\n4. 验证特征维度...')
batch = next(iter(train_loaders[0]))
features, labels, subjects = batch

print(f'\n特征维度: {features.shape}')
print(f'标签维度: {labels.shape}')
print(f'预期特征维度: [batch_size, 1256]')
print(f'  - BCG信号: 1250维 (5秒 × 250Hz)')
print(f'  - PI特征: 4维 (gender, age, weight, height)')
print(f'  - Last_BP: 2维 (last_DBP, last_SBP)')

# 验证维度
if features.shape[1] == 1256:
    print('\n✓✓✓ 测试通过！PI功能已成功启用！')
    print('    特征维度正确: 1256 = 1250(BCG) + 4(PI) + 2(LastBP)')
else:
    print(f'\n✗✗✗ 测试失败！')
    print(f'    期望维度: 1256')
    print(f'    实际维度: {features.shape[1]}')
    print(f'    差异: {features.shape[1] - 1256}')
    sys.exit(1)

# 显示第一个样本的PI特征值
print(f'\n5. 检查PI特征值...')
first_sample_PI = features[0, 1250:1254]
print(f'第一个样本的PI特征: {first_sample_PI.tolist()}')
print(f'格式说明: [gender(0/1), age, weight, height]')

# 显示受试者信息
print(f'\n6. 受试者信息...')
# subjects可能是tuple、list、numpy array或tensor，需要兼容处理
if isinstance(subjects, (list, tuple)):
    subjects_list = list(subjects)
elif hasattr(subjects, 'tolist'):
    subjects_list = subjects.tolist()
else:
    subjects_list = list(subjects)

print(f'训练集用户数: {len(set(subjects_list))}')
print(f'第一个batch的受试者ID样例: {subjects_list[:5]}')

print('\n' + '='*60)
print('✓ 所有测试完成!')
print('='*60)
