# 基于J峰的FF特征实现总结

## 项目目标

在 `_load_subjects_data_from_struct` 方法中启用FF（Fiducial Features）特征，由于数据中没有预标注的HIJKL特征点，需要通过以下方式实现：

1. 从心率CSV读取每个被试的心率
2. 使用 `find_peaks` 检测J峰位置
3. 基于J峰计算9维FF特征

---

## 实施过程

### 阶段1：滤波效果对比实验 ✅

**目标**：确定J峰检测的最优滤波方案

**实验设计**：
- 创建测试脚本 [`test_filter_effect.py`](file:///d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/test_filter_effect.py)
- 加载心率CSV（包含true_hr作为参考）
- 对40个被试的BCG数据分别测试不同滤波方案
- 在5秒窗口内检测J峰并计算心率
- 与true_hr比较计算MAE

**测试方案对比**：

| 方案 | distance参数 | MAE (bpm) | 说明 |
|------|-------------|-----------|------|
| 不滤波 | 0.7 | 1.5260 | 基线方案 |
| 带通滤波 (0.5-20Hz) | 0.7 | 1.5255 | 无明显改善 |
| **带通+包络** | 0.7 | 0.9553 | 效果提升37.4% |
| **平滑+包络** | 0.7 | **0.9407** | ✅ 最优 |
| 窗口后平滑+包络 | 0.7 | 1.1398 | 较差 |

**实验结论**：
- **最优方案**：平滑（smooth_signal + savgol_filter）+ Hilbert包络
- **MAE**：0.9407 bpm（相比不滤波的1.5260 bpm，提升**38.3%**）
- **distance参数**：0.7 × (60 / hr_1d) × fs
- **关键发现**：**在划分窗口前对整个信号滤波**效果优于窗口内滤波

---

### 阶段2：实现FF特征提取 ✅

#### 1. 创建FF特征提取函数

**文件**：[`util/fiducial_feature_extraction.py`](file:///d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/util/fiducial_feature_extraction.py)

添加函数 `compute_features_from_j_peaks`：

```python
def compute_features_from_j_peaks(BCG, J_locs, fs=250):
    """
    基于J峰位置计算9维特征
    
    返回:
        features: numpy数组 (9,)，包含：
            [0-2]: J峰间隔统计（均值、标准差、CV）
            [3-5]: J峰幅值统计（均值、标准差、CV）
            [6]: 平均心率
            [7]: 心率变异性RMSSD
            [8]: J峰幅值均值（Extremum）
    """
```

**9维特征说明**：
1. **J峰间隔均值**：连续J峰间隔的平均值（秒）
2. **J峰间隔标准差**：间隔的变化程度
3. **J峰间隔CV**：变异系数 (std/mean)，归一化的变化指标
4. **J峰幅值均值**：J峰处BCG信号幅值的平均
5. **J峰幅值标准差**：幅值的变化程度
6. **J峰幅值CV**：归一化的幅值变化
7. **平均心率**：60 / 平均间隔 (bpm)
8. **心率变异性RMSSD**：连续间隔差值的均方根
9. **Extremum**：J峰幅值均值（类似原44维FF中的Extremum特征）

#### 2. 修改DataProcessor类

**文件**：[`util/data_preprocessing2.py`](file:///d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/util/data_preprocessing2.py)

**修改点1**：`__init__` 方法（第51-59行）
```python
self.dim_FF = 9  # 启用9维FF特征
self.hr_csv_path = params['hr_csv_path']
self.hr_dict = {}  # 存储每个被试的心率
```

**修改点2**：`load_data` 方法（第78-90行）
```python
# 读取心率CSV文件
if self.hr_csv_path and os.path.exists(self.hr_csv_path):
    hr_df = pd.read_csv(self.hr_csv_path)
    for _, row in hr_df.iterrows():
        self.hr_dict[str(row['subject_id'])] = row['pred_hr']
```

**修改点3**：`_load_subjects_data_from_struct` 方法

**a) 获取用户心率**（第530-543行）：
```python
# 支持ID映射：X0132 -> X1032
hr_user_id = 'X1032' if user_id == 'X0132' else user_id
user_hr = self.hr_dict.get(hr_user_id, 70.0)  # 默认70 bpm
```

**b) 应用最优滤波方案**（第545-564行）：
```python
# 预处理：去除NaN、去趋势、平滑
BCG_for_peak = np.nan_to_num(BCG, nan=np.nanmean(BCG))
BCG_for_peak = signal.detrend(BCG_for_peak, type='constant')

# Savitzky-Golay平滑（两次）
BCG_for_peak = savgol_filter(BCG_for_peak, window_length=..., polyorder=3)
BCG_for_peak = savgol_filter(BCG_for_peak, 5, 3, mode='nearest')

# Hilbert包络
BCG_filtered = np.abs(scipy.signal.hilbert(BCG_for_peak))
```

**c) 窗口内计算FF特征**（第611-635行）：
```python
# 在滤波后的窗口中检测J峰
window_bcg_filtered = BCG_filtered[window_start:window_end]
distance = int(0.7 * (60 / user_hr) * self.fs)
peaks, _ = find_peaks(window_bcg_filtered, distance=distance)

# 计算FF特征
if len(peaks) >= 2:
    FF_features = compute_features_from_j_peaks(seg, peaks, fs=self.fs)
    FF_tensor = torch.tensor(FF_features, dtype=torch.float32)
```

**d) 更新特征拼接**（第637-647行）：
```python
# 特征组合逻辑：
if dim_spi > 0 and PI可用:
    if dim_FF > 0:
        特征 = BCG + PI + FF + LastBP
    else:
        特征 = BCG + PI + LastBP
else:
    if dim_FF > 0:
        特征 = BCG + FF + LastBP
    else:
        特征 = BCG + LastBP
```

---

## 测试验证

### 单元测试

**脚本**：[`test_ff_features.py`](file:///d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/test_ff_features.py)

**测试结果**：
```
✅ 测试通过！

特征说明:
  [0] J峰间隔均值: 0.8480 秒
  [1] J峰间隔标准差: 0.0264 秒
  [2] J峰间隔CV: 0.0311
  [3] J峰幅值均值: 1.1634
  [4] J峰幅值标准差: 0.0489
  [5] J峰幅值CV: 0.0420
  [6] 平均心率: 70.75 bpm
  [7] 心率变异性RMSSD: 0.0362
  [8] Extremum: 1.1634
```

**验证项**：
- ✅ 特征维度正确 (9,)
- ✅ 无NaN值
- ✅ 计算心率接近预期（70.75 vs 70 bpm）

### 集成测试要点

**特征维度验证**：
- 不使用PI和FF：1250(BCG) + 2(LastBP) = **1252维**
- 使用PI不使用FF：1250 + 4(PI) + 2 = **1256维**
- 不使用PI使用FF：1250 + 9(FF) + 2 = **1261维**
- 使用PI和FF：1250 + 4 + 9 + 2 = **1265维**

**后续验证建议**：
1. 在main.py中设置 `hr_csv_path` 参数
2. 运行数据加载，检查日志输出
3. 验证特征维度匹配
4. 训练模型并对比MAE指标

---

## 关键技术决策

### 1. 滤波方案选择

**决策**：使用平滑+Hilbert包络，且在划分窗口前滤波

**理由**：
- 相比不滤波，MAE降低38.3%
- 相比带通滤波，平滑滤波效果更好
- 整体滤波比窗口内滤波MAE更低（0.9407 vs 1.1398）

### 2. FF特征维度

**决策**：设计9维特征（而非44维）

**理由**：
- 原44维特征依赖HIJKL五个特征点
- 只能检测J峰，因此提取J峰相关的统计特征
- 包含时域（间隔、幅值）和频域（心率、变异性）信息

### 3. ID映射处理

**决策**：支持X0132 → X1032的映射

**理由**：
- 数据中存在ID不一致问题
- 确保所有被试都能找到对应的心率数据

---

## 使用指南

### 配置参数

在 `main.py` 的 `params` 字典中添加：

```python
params = {
    # ... 现有参数 ...
    'hr_csv_path': r'D:\zsy\坐垫\文献梳理及画图\论文\20250905第二次投递submitCode_npj digital medicine提交\sorted\result\pred_hr_1d_subject_level.csv',
}
```

### 心率CSV格式

CSV文件需包含以下列：
- `subject_id`: 被试ID（字符串，如'X1001'）
- `true_hr`: 真实心率（可选，用于验证）
- `pred_hr`: 预测心率（必需，用于J峰检测）

---

## 总结

**已完成**：
- ✅ 滤波效果对比实验，确定最优方案
- ✅ 实现9维FF特征提取函数
- ✅ 集成心率CSV读取
- ✅ 集成最优滤波方案（在窗口前对整体信号滤波）
- ✅ 实现窗口内J峰检测和FF特征计算
- ✅ 更新特征拼接逻辑支持FF特征
- ✅ 单元测试验证

**特征提升**：
- 心率检测MAE从1.5260降至0.9407 bpm（提升38.3%）
- 特征维度从1252/1256增加到1261/1265
- 保持与原系统的兼容性（可选启用FF）

**下一步**：
1. 在main.py中配置 `hr_csv_path`
2. 运行完整的数据加载和训练流程
3. 对比启用FF前后的血压预测MAE
