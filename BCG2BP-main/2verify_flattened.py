"""
验证展开后的平铺 struct 文件
"""

from hdf5storage import loadmat
import numpy as np

mat_path = 'd:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/Preprocessed_Database_flat.mat'

print("="*60)
print("验证展开后的 struct 文件")
print("="*60)

try:
    print("\n正在加载文件...")
    mat = loadmat(mat_path)
    print("✅ 文件加载成功!")
    
    print("\n顶层键:")
    for key in mat.keys():
        if not key.startswith('__'):
            print(f"  - {key}: {type(mat[key])}")
    
    # 检查 preprocessed_Data_struct
    if 'preprocessed_Data_struct' in mat:
        data = mat['preprocessed_Data_struct']
        print(f"\n✅ 找到 'preprocessed_Data_struct'")
        print(f"类型: {type(data)}")
        
        if isinstance(data, np.ndarray):
            print(f"数组形状: {data.shape}")
            print(f"受试者数量: {data.shape[0]}")
            
            # 检查字段
            if data.dtype.names:
                print(f"\n✅ 字段名:")
                for field_name in data.dtype.names:
                    print(f"  - {field_name}")
                
                # 检查第一个受试者的数据
                print(f"\n第一个受试者的数据:")
                first_subject = data[0, 0] if data.ndim > 1 else data[0]
                
                for field_name in data.dtype.names:
                    field_value = first_subject[field_name]
                    print(f"\n  [{field_name}]:")
                    
                    if isinstance(field_value, np.ndarray):
                        print(f"    shape: {field_value.shape}")
                        print(f"    dtype: {field_value.dtype}")
                        
                        # 显示 BCG 或血压数据
                        if field_name in ['BCG', 'SBP', 'DBP']:
                            flat = field_value.flatten()
                            print(f"    数据长度: {len(flat)}")
                            print(f"    前10个值: {flat[:10]}")
                            print(f"    数据范围: [{flat.min():.2f}, {flat.max():.2f}]")
                    else:
                        print(f"    值: {field_value}")
                
                print("\n" + "="*60)
                print("✅ 验证成功！数据结构正确！")
                print("\n现在可以在 data_preprocessing2.py 中使用此文件了")
                print("="*60)
            else:
                print("⚠️ 不是结构数组")
    
    else:
        print("\n❌ 未找到 'preprocessed_Data_struct' 键")
        print("可用的键:", [k for k in mat.keys() if not k.startswith('__')])

except FileNotFoundError:
    print(f"\n❌ 文件未找到: {mat_path}")
    print("\n请先在 MATLAB 中运行 flatten_nested_table.m 脚本")
    
except Exception as e:
    print(f"\n❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
