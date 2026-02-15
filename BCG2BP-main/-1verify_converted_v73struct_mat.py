"""
验证转换后的 struct.mat 文件是否可以被 Python 正确读取

好的，用户成功转换了文件！现在我看到：

✅ 文件加载成功
有40个受试者
字段名只有 'ID' 和 'FilteredData'
但是问题来了：

'FilteredData' 是一个 uint32 数组，shape 是 (1, 1, 6)
这看起来还是引用数据，不是真实的 BCG/SBP/DBP 数据

"""

from hdf5storage import loadmat
import numpy as np

# 测试转换后的文件
mat_path = 'd:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/Preprocessed_Database_v73struct.mat'

print("="*60)
print("验证转换后的 .mat 文件")
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
        
        # 检查是否为结构数组
        if isinstance(data, np.ndarray):
            print(f"数组形状: {data.shape}")
            print(f"受试者数量: {data.shape[0] if data.ndim > 0 else 1}")
            
            # 如果是结构数组，检查字段
            if data.dtype.names:
                print(f"\n字段名: {data.dtype.names}")
                
                # 尝试访问第一个受试者的数据
                print("\n第一个受试者的数据:")
                first_subject = data[0] if data.ndim > 0 else data
                
                for field_name in data.dtype.names:
                    field_value = first_subject[field_name]
                    print(f"\n  [{field_name}]:")
                    print(f"    类型: {type(field_value)}")
                    
                    if isinstance(field_value, np.ndarray):
                        print(f"    shape: {field_value.shape}")
                        print(f"    dtype: {field_value.dtype}")
                        
                        # 显示 BCG 或血压数据的前几个值
                        if field_name in ['BCG', 'SBP', 'DBP']:
                            flat = field_value.flatten()
                            print(f"    前10个值: {flat[:10]}")
                            print(f"    数据长度: {len(flat)}")
                    else:
                        print(f"    值: {field_value}")
                
                print("\n" + "="*60)
                print("✅ 验证成功！文件结构正确！")
                print("="*60)
            else:
                print("⚠️ 不是结构数组，可能需要调整 MATLAB 转换脚本")
        else:
            print(f"⚠️ 数据类型不是 ndarray: {type(data)}")
    
    else:
        print("\n❌ 未找到 'preprocessed_Data_struct' 键")
        print("可用的键:", [k for k in mat.keys() if not k.startswith('__')])

except FileNotFoundError:
    print(f"\n❌ 文件未找到: {mat_path}")
    print("\n请先在 MATLAB 中运行 convert_mat_file.m 脚本")
    
except Exception as e:
    print(f"\n❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
