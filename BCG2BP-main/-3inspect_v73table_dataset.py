"""
深入检查 v7.3 格式的 MATLAB table Dataset

输出显示：

MATLAB table 被读取为一个 uint32 数组，shape (1, 6)
不是结构数组
内容是 [[3707764736 2 1 1 42 2]]
这看起来不对。MATLAB table 在保存为 v7.3 格式时，数据被编码了。MATLAB_object_decode: 3 说明这是一个需要特殊解码的 MATLAB 对象。

看起来直接重新保存为 v7.3 并不能让 Python 直接读取 table 数据。我需要回到方案1：在 MATLAB 中将 table 转换为 struct。
"""

import h5py
import numpy as np

mat_path = 'd:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/Preprocessed_Database_v73.mat'

print("="*60)
print("深入检查 MATLAB table Dataset 结构")
print("="*60)

with h5py.File(mat_path, 'r') as f:
    table_dataset = f['Preprocessed_Database']
    
    print(f"\n对象名: Preprocessed_Database")
    print(f"类型: {type(table_dataset)}")
    print(f"MATLAB 类: {table_dataset.attrs['MATLAB_class'].decode('utf-8')}")
    
    print(f"\n基本信息:")
    print(f"  shape: {table_dataset.shape}")
    print(f"  dtype: {table_dataset.dtype}")
    print(f"  size: {table_dataset.size}")
    
    print(f"\n所有属性:")
    for attr_name in table_dataset.attrs.keys():
        attr_value = table_dataset.attrs[attr_name]
        if isinstance(attr_value, bytes):
            attr_value = attr_value.decode('utf-8')
        print(f"  {attr_name}: {attr_value}")
    
    # 读取数据
    print(f"\n读取数据:")
    data = table_dataset[()]
    print(f"  数据类型: {type(data)}")
    print(f"  数据 shape: {data.shape}")
    print(f"  数据 dtype: {data.dtype}")
    
    # 如果是结构数组，显示字段名
    if data.dtype.names:
        print(f"\n✅ 这是一个结构数组，字段名:")
        for field_name in data.dtype.names:
            print(f"  - {field_name}")
        
        # 查看第一个元素的每个字段
        if data.size > 0:
            print(f"\n第一个元素的数据:")
            first_item = data.flatten()[0]
            for field_name in data.dtype.names:
                field_value = first_item[field_name]
                print(f"\n  [{field_name}]:")
                print(f"    类型: {type(field_value)}")
                
                # 如果是引用，需要解引用
                if isinstance(field_value, h5py.h5r.Reference):
                    try:
                        deref_data = f[field_value][()]
                        print(f"    这是一个引用，解引用后:")
                        print(f"      类型: {type(deref_data)}")
                        if isinstance(deref_data, np.ndarray):
                            print(f"      shape: {deref_data.shape}")
                            print(f"      dtype: {deref_data.dtype}")
                            if field_name in ['BCG', 'SBP', 'DBP']:
                                print(f"      前10个值: {deref_data.flatten()[:10]}")
                    except Exception as e:
                        print(f"    解引用失败: {e}")
                else:
                    print(f"    值: {field_value}")
        
        print("\n" + "="*60)
        print("✅ 结构检查完成！")
        print("="*60)
    
    else:
        print(f"\n⚠️ 不是结构数组")
        print(f"数据内容: {data}")

print("\n" + "="*60)
