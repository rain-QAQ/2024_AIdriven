import torch
import numpy as np
import torch.nn.functional as F


def scale_F_BCG(F_BCG_original, current_dbp, current_sbp, ref_BCG):
    if len(F_BCG_original) == 0:
        return F_BCG_original  # Return the empty array as is

    min_val = torch.min(ref_BCG)
    max_val = torch.max(ref_BCG)

    F_BCG_scaled = ((F_BCG_original - min_val) / (max_val - min_val + 1e-8)) * (current_sbp - current_dbp) + current_dbp

    return F_BCG_scaled

def predict(model, data):
    model.eval()
    pred = torch.empty(0)
    labels = torch.empty(0)

    device = next(model.parameters()).device

    for batch_samples, batch_labels, batch_IDs in data:
        batch_samples = batch_samples.to(device)

        x = batch_samples

        pre, _, _ = model(x)

        pred = torch.cat((pred, pre.detach().cpu()), 0)
        labels = torch.cat((labels, batch_labels), dim=0)

    return pred.cpu().detach().numpy(), labels.cpu().detach().numpy()

def evaluate(pred, real):
    # 计算误差
    error = pred - real

    # 计算平均绝对误差（MAE）
    mae = np.mean(np.abs(error), axis=0)
    # 计算平均误差（Mean Error）
    mean_error = np.mean(error, axis=0)
    # 计算标准差（Standard Deviation）
    std_dev = np.std(error, axis=0)
    
    # add_by_zsy: 计算PCC（皮尔逊相关系数）
    # pred和real的shape为(n_samples, 2)，其中2维分别是DBP和SBP
    
    # 改进的PCC计算，处理特殊情况
    def safe_corrcoef(x, y):
        """安全计算相关系数，处理方差为0和样本数不足的情况"""
        if len(x) < 2:
            print('样本数不足')
            return 0.0  # 样本数不足
        
        # 检查方差
        if np.std(x) == 0 or np.std(y) == 0:
            print('方差为0')
            return 0.0  # 方差为0，无相关性
        
        # 正常计算
        try:
            corr_matrix = np.corrcoef(x, y)
            return corr_matrix[0, 1]
        except:
            print('计算异常')
            return 0.0  # 任何其他异常
    
    pcc_dbp = safe_corrcoef(pred[:, 0], real[:, 0])
    pcc_sbp = safe_corrcoef(pred[:, 1], real[:, 1])
    pcc = np.array([pcc_dbp, pcc_sbp])
    
    return mae, mean_error, std_dev, pcc
