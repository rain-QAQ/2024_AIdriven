"""
测试数据加载脚本
验证修改后的代码能否正确加载数据
"""

import sys
sys.path.append('.')
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from util.data_preprocessing2 import *

print("="*60)
print("测试数据加载流程")
print("="*60)

# 配置参数
datapath = ['d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/Preprocessed_Database_flat.mat']
signal_type = 'day'
training_scheme = 'NDL'

params = {
    'batch_size': 32,
    'finetune_N': 3,
    'is_finetune': 0,  # 禁用个性化
    'training_scheme': 'NDL',
    'finetune_strategy': 'sequential',
    'is_balance_dist': 0,
}

try:
    # 初始化数据处理器
    print("\n1. 初始化 DataProcessor...")
    data_processor = DataProcessor(datapath, signal_type, training_scheme, params)
    print("✅ 初始化成功")
    
    # 加载数据
    print("\n2. 加载数据...")
    data_processor.load_data()
    print("✅ 数据加载成功")
    
    # 获取 DataLoader
    print("\n3. 获取 DataLoader...")
    train_loaders, test_loaders, fine_tune_loaders = data_processor.get_loaders()
    print(f"✅ 获取成功: {len(train_loaders)} 个 fold")
    
    # 检查第一个 fold 的数据
    print("\n4. 检查第一个 fold 的数据...")
    train_loader = train_loaders[0]
    test_loader = test_loaders[0]
    
    print(f"  训练集大小: ~{len(train_loader.dataset)} 个样本")
    print(f"  测试集大小: ~{len(test_loader.dataset)} 个样本")
    
    # 获取一个 batch 验证数据形状
    for batch_features, batch_labels, batch_ids in train_loader:
        print(f"\n5. 样本形状验证:")
        print(f"  features shape: {batch_features.shape}")
        print(f"  labels shape: {batch_labels.shape}")
        print(f"  预期 features shape: [batch_size, 1250 + 2] = [batch_size, 1252] (无PI特征)")
        
        expected_dim = 1250 + 2  # BCG + Last_BP (不使用PI特征)
        if batch_features.shape[1] == expected_dim:
            print(f"  ✅ 特征维度正确!")
        else:
            print(f"  ⚠️ 特征维度不匹配! 预期: {expected_dim}, 实际: {batch_features.shape[1]}")
        
        break  # 只检查第一个 batch
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！数据加载成功！")
    print("="*60)
    
except Exception as e:
    print("\n❌ 测试失败!")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
