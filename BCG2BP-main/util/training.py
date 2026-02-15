import time
import copy
import math

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

class SimilarityLoss(nn.Module):  # 注意继承 nn.Module
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    # 清泳实现版
    def forward(self, out, labels):
        rho = torch.cosine_similarity(out - out.mean(), labels - labels.mean(), dim=-1).mean()
        loss = 1 - rho
        return loss

class PCCLoss(nn.Module):
    """add_by_zsy: 皮尔逊相关系数损失
    
    目标: 最大化预测值和真实值之间的相关性
    Loss = 1 - mean(PCC_DBP, PCC_SBP)
    """
    def __init__(self):
        super(PCCLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        参数:
            pred: (batch_size, 2) 预测值 [DBP, SBP]
            target: (batch_size, 2) 真实值 [DBP, SBP]
        
        返回:
            loss: 标量，PCC损失
        """
        # 分别计算DBP和SBP的PCC
        pcc_dbp = self._pearson_correlation(pred[:, 0], target[:, 0])
        pcc_sbp = self._pearson_correlation(pred[:, 1], target[:, 1])
        
        # 平均PCC，然后转换为损失 (1 - PCC)，范围[0, 2]
        avg_pcc = (pcc_dbp + pcc_sbp) / 2.0
        loss = 1.0 - avg_pcc
        
        return loss
    
    def _pearson_correlation(self, x, y):
        """计算两个向量的皮尔逊相关系数
        
        参数:
            x, y: (batch_size,) 张量
        
        返回:
            pcc: 标量，相关系数 [-1, 1]
        """
        # 中心化
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # 计算分子和分母
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
        
        # 避免除零
        pcc = numerator / (denominator + 1e-8)
        
        # 限制在[-1, 1]范围内（理论上应该自动满足，但加个clamp更安全）
        pcc = torch.clamp(pcc, -1.0, 1.0)
        
        return pcc

def scale_F_BCG(F_BCG, current_dbp, current_sbp):
    if len(F_BCG) == 0:
        return F_BCG  # Return the empty array as is

    min_val, _ = torch.min(F_BCG, dim=1, keepdim=True)
    max_val, _ = torch.max(F_BCG, dim=1, keepdim=True)

    F_BCG_scaled = ((F_BCG - min_val) / (max_val - min_val + 1e-8)) * (
                current_sbp.unsqueeze(-1) - current_dbp.unsqueeze(-1)) + current_dbp.unsqueeze(-1)

    return F_BCG_scaled

def train_DL(log, model, model_path, train_dataloader, validation_dataloader, lr, epoch, params):
    best_model = copy.deepcopy(model)

    device = next(model.parameters()).device
    dim_PI = model.dim_PI
    dim_FF = model.dim_FF

    loss_regression = nn.HuberLoss(reduction='mean', delta=5) #nn.MSELoss()
    loss_reconstruction = nn.MSELoss()
    loss_pcc = PCCLoss()  # add_by_zsy: PCC损失
    # loss_similarity1 = SimilarityLoss()#nn.CosineSimilarity()
    # loss_similarity2 = SimilarityLoss()#nn.CosineSimilarity()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4, weight_decay=1e-5)

    best_val_loss = 1000
    
    # add_by_zsy: 用于记录每个epoch的loss历史
    train_loss_history = []
    val_loss_history = []

    for e in range(epoch):
        st = time.time()
        loss_sum, loss_val_sum, train_steps, val_steps = 0, 0, 0, 0
        loss_s1, loss_s2, loss_val_s1, loss_val_s2 = 0, 0, 0, 0
        # add_by_zsy: 累积三个训练损失的总和
        reg_loss_sum, recon_loss_sum, pcc_loss_sum = 0, 0, 0

        model.train()
        for step, (features, labels, IDs) in enumerate(train_dataloader):
            train_steps = train_steps + 1

            optimizer.zero_grad()

            features = features.to(device)
            labels = labels.to(device)

            # raw_bcg = features[:, 0:-dim_PI - dim_FF - 2]
            # raw_bcg = scale_F_BCG(raw_bcg, labels[:,0], labels[:,1])
            # covar = features[:, -dim_PI - dim_FF: -2]
            x = features
            last_BP = features[:, -2:] # DBP, SBP
            
            # add_by_zsy: 添加调试信息，在第一个epoch的第一个batch打印数据统计
            # if e == 0 and step == 0:
            #     log(f'【调试】训练数据统计:')
            #     log(f'  Features shape: {features.shape}')
            #     log(f'  Features range: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}')
            #     log(f'  Labels shape: {labels.shape}')
            #     log(f'  Labels range: min={labels.min().item():.4f}, max={labels.max().item():.4f}, mean={labels.mean().item():.4f}')
            #     log(f'  Labels DBP: min={labels[:,0].min().item():.2f}, max={labels[:,0].max().item():.2f}')
            #     log(f'  Labels SBP: min={labels[:,1].min().item():.2f}, max={labels[:,1].max().item():.2f}')
            #     # 检查是否包含NaN或Inf
            #     log(f'  Features contains NaN: {torch.isnan(features).any().item()}')
            #     log(f'  Features contains Inf: {torch.isinf(features).any().item()}')
            #     log(f'  Labels contains NaN: {torch.isnan(labels).any().item()}')
            #     log(f'  Labels contains Inf: {torch.isinf(labels).any().item()}')

            regression_out, _, recon_bcg = model(x)
            
            # add_by_zsy: 添加模型输出检查
            # if e == 0 and step == 0:
            #     log(f'  Model regression_out: min={regression_out.min().item():.4f}, max={regression_out.max().item():.4f}')
            #     log(f'  Model recon_bcg: min={recon_bcg.min().item():.4f}, max={recon_bcg.max().item():.4f}')
            #     log(f'  Model outputs contain NaN: reg={torch.isnan(regression_out).any().item()}, recon={torch.isnan(recon_bcg).any().item()}')
            
            regression_loss = loss_regression(regression_out, labels)
            reconstruction_loss = loss_reconstruction(recon_bcg.squeeze(1), x[:,0:1250])
            pcc_loss = loss_pcc(regression_out, labels)  # add_by_zsy: 计算PCC损失
            
            isFig = False   # temporary
            # add_by_zsy: 添加loss检查和BCG重建可视化
            if step == 0 and isFig:  # 每个epoch的第一个batch
                if e == 0:  # 只在第一个epoch输出loss详情
                    log(f'  regression_loss: {regression_loss.item():.4f}')
                    log(f'  reconstruction_loss: {reconstruction_loss.item():.4f}')
                    log(f'  pcc_loss: {pcc_loss.item():.4f}')
                
                # 可视化BCG重建效果（每个epoch都保存）
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # 获取第一个样本
                original_bcg = x[0, 0:1250].detach().cpu().numpy()
                reconstructed_bcg = recon_bcg[0].squeeze().detach().cpu().numpy()
                
                # 创建对比图
                fig, axes = plt.subplots(3, 1, figsize=(15, 10))
                time_axis = np.arange(1250) / 250.0  # 假设250Hz采样率，转换为秒
                
                # 子图1: 原始BCG
                axes[0].plot(time_axis, original_bcg, 'b-', linewidth=1, label='Original BCG')
                axes[0].set_title(f'Original BCG Signal (Epoch {e+1})', fontsize=14, fontweight='bold')
                axes[0].set_xlabel('Time (s)', fontsize=12)
                axes[0].set_ylabel('Amplitude', fontsize=12)
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(fontsize=10)
                
                # 子图2: 重建BCG
                axes[1].plot(time_axis, reconstructed_bcg, 'r-', linewidth=1, label='Reconstructed BCG')
                axes[1].set_title(f'Reconstructed BCG Signal (Epoch {e+1})', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Time (s)', fontsize=12)
                axes[1].set_ylabel('Amplitude', fontsize=12)
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=10)
                
                # 子图3: 重叠对比
                axes[2].plot(time_axis, original_bcg, 'b-', linewidth=1, alpha=0.7, label='Original')
                axes[2].plot(time_axis, reconstructed_bcg, 'r--', linewidth=1, alpha=0.7, label='Reconstructed')
                mse = ((original_bcg - reconstructed_bcg)**2).mean()
                axes[2].set_title(f'BCG Reconstruction Comparison (Epoch {e+1}, MSE={mse:.6f})', fontsize=14, fontweight='bold')
                axes[2].set_xlabel('Time (s)', fontsize=12)
                axes[2].set_ylabel('Amplitude', fontsize=12)
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(fontsize=10)
                
                plt.tight_layout()
                
                # 保存图像（文件名包含epoch编号）
                import os
                vis_dir = '../result/bcg_reconstruction'
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                save_path = os.path.join(vis_dir, f'bcg_reconstruction_epoch{e+1:02d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                log(f'  [Epoch {e+1}] BCG重建图已保存: {save_path} (MSE={mse:.6f})')
                plt.close()
            
            # print(f'Training: Last_BP: {last_BP.shape}, Labels: {labels.shape}')


            # 更新模型
            # add_by_zsy: 添加PCC损失项（权重可调）
            pcc_weight = params['pcc_weight']  
            if params['is_similarity_loss']:
                pass # loss_total = regression_loss + si_loss1 + si_loss2
            else:
                loss_total = regression_loss + reconstruction_loss + pcc_weight * pcc_loss
                # loss_total = pcc_loss    
            
            # add_by_zsy: 检查loss是否为NaN
            if torch.isnan(loss_total):
                log(f'  ⚠️ 警告: loss_total为NaN! regression_loss={regression_loss.item()}, reconstruction_loss={reconstruction_loss.item()}')
                log(f'  regression_out: {regression_out[0]}')
                log(f'  labels: {labels[0]}')
                # 跳过这个batch
                continue
                
            loss_total.backward()
            optimizer.step()
            loss_sum += loss_total.item()
            # add_by_zsy: 累积三个损失值
            reg_loss_sum += regression_loss.item()
            recon_loss_sum += reconstruction_loss.item()
            pcc_loss_sum += pcc_loss.item()
            # loss_s1 += si_loss1.item()
            # loss_s2 += si_loss2.item()

            # 输出损失信息
            # log('Training: epoch: %d, step:%d, ls_reg: %.2f, total_loss: %.2f' \
            #     % (e, step, regression_loss.data.cpu().numpy(), loss_total))

        if validation_dataloader is not None:
            model.eval()
            for step, (features, labels, IDs) in enumerate(validation_dataloader):
                val_steps = val_steps + 1

                features = features.to(device)
                labels = labels.to(device)

                # raw_bcg = features[:, 0:-dim_PI - dim_FF]
                # covar = features[:, -dim_PI - dim_FF:]
                x = features
                last_BP = features[:, -2:]  # DBP, SBP

                regression_out, _, recon_bcg= model(x)

                regression_loss = loss_regression(regression_out, labels)
                reconstruction_loss = loss_reconstruction(recon_bcg.squeeze(1), x[:, 0:1250])
                pcc_loss = loss_pcc(regression_out, labels)  # add_by_zsy: 计算PCC损失

                # temp_zsy
                pred = data_processor.denormalize_bp(regression_out)
                labels = data_processor.denormalize_bp(labels)
                mae, mean_error, std_dev, pcc = evaluate(pred, labels)

                if params['is_similarity_loss']:
                    pass # loss_val = regression_loss.item() + si_loss1.item() + si_loss2.item()
                else:
                    loss_val = regression_loss.item() + reconstruction_loss.item()
                    # loss_val = pcc_loss.item()
                loss_val_sum += loss_val

                # log('Validation: epoch: %d, step:%d, ls_reg: %.2f, total_loss: %.2f' \
                #     % (e, step, regression_loss.data.cpu().numpy(), loss_val))

            val_loss = loss_val_sum / val_steps
        else:
            val_loss = loss_sum / train_steps
        
        # add_by_zsy: 记录每个epoch的loss
        train_loss_history.append(loss_sum / train_steps)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log('saving model')
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), model_path)


        # add_by_zsy: 输出详细的训练损失分解
        avg_reg_loss = reg_loss_sum / train_steps
        avg_recon_loss = recon_loss_sum / train_steps
        avg_pcc_loss = pcc_loss_sum / train_steps
        avg_total_loss = loss_sum / train_steps

        # log('BPEstimator: [%d/%d] loss: %.4f, val_loss: %.4f'
        #     % (e + 1, epoch, loss_sum / train_steps, val_loss))
        log('BPEstimator: [%d/%d] total_loss: %.4f, val_loss: %.4f' % (e + 1, epoch, avg_total_loss, val_loss))
        log('  ├─ regression_loss: %.4f' % avg_reg_loss)
        log('  ├─ reconstruction_loss: %.4f' % avg_recon_loss)
        log('  └─ pcc_loss: %.4f (weight=%.1f)' % (avg_pcc_loss, pcc_weight))
        log('Cunrrent ephoch run time: %.3f s' % (time.time() - st))

        torch.cuda.empty_cache()

    return best_model, train_loss_history, val_loss_history