#按 fold 进行训练/测试 + 个性化（Personalization Adapter, PA）微调/训练 + SHAP 可解释性分析的主程序。
import argparse
import os
from os.path import join as fullfile
import sys
import copy
import time

import numpy as np
# del_by_zsy: 移除未使用的d2l库导入
# import d2l
from hdf5storage import savemat, loadmat

from util import common_utils
from util.logger import Logger
from util.evaluation import *
from util.data_preprocessing2 import *
from PersonalizationAdapter import *
from util.training import *
from model.BCGNET import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--feature_spi', help='bool, features option', default=True)
parser.add_argument('--data_dir', default='../data/', type=str)
parser.add_argument('--signal', default='darma', type=str)
parser.add_argument('--num_channels', default=1, type=int)
parser.add_argument('--model', default='../saved_model/%s/%s-%s-Model.pt', type=str)  # add_by_zsy: saved_model/experiment_name/setting-fold-Model.pt
parser.add_argument('--training', default=False, action='store_true')   # 不加 --training：默认 testing mode（只 load 模型推理）；加了 --training：进入 training mode（训练+保存）
parser.add_argument('--training_scheme', default=2, type=int)  # 1=leave-one-out, 2=adversaril-transfer-learning
parser.add_argument('--experiment_name', default='test', type=str)  # add_by_zsy: 实验名称，用于创建子目录组织文件  每次运行时修改

#--model / --log / --savemat 都是带格式化占位符 %s 的路径模板，后面会用 (setting, fold_idx) 或 (setting, session, signal, ...) 去填。
# add_by_zsy: 路径中添加experiment_name子目录
parser.add_argument('--log', default='../log/%s/%s-%s-%s.log', type=str)  # log/experiment_name/setting-session-signal.log
parser.add_argument('--savemat', default='../result/%s/%s-%s-%s-%s.mat', type=str)  # result/experiment_name/...

parser.add_argument('--session', default='s3-1', type=str)
parser.add_argument('--ufrom', default=0, type=int)
parser.add_argument('--uto', default=40, type=int)
args = parser.parse_args()

common_utils.set_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

# datapath = '../data/night-up_35p_FF_short3.mat'
# datapath = '../data/s3-1_update_v3.mat'
# datapath = ['../data/s2-1_v3.mat', '../data/s2-2_v3.mat', '../data/s2-3_v3.mat', '../data/s3-1_v3.mat']
# datapath = ['../data/preprocessed_Data_labledpeaks_250Hz_40p.mat']  # mat 文件列表（允许多个文件）
# datapath = ['../data/night-up_35p_1-76_v3.mat', '../data/night-up_35p_77-148_v3.mat', '../data/night-up_35p_149-240_v3.mat', '../data/night-up_35p_242-275_v3.mat']
datapath = ['d:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/Preprocessed_Database_flat.mat']

# 用来控制 DataProcessor 的处理逻辑（例如不同数据域/不同切分方式）
signal_type = 'day'  # day
training_scheme = 'NDL'

params = {
    'finetune_strategy': 'sequential',
    'training_scheme': 'NDL',
    'batch_size': 32,
    'finetune_N': 10,  # add_by_zsy: 微调样本数量，原值3
    'is_finetune': 1,      # 是否进行个性化微调
    'is_balance_dist': 0,
    'is_similarity_loss': 0,
    'is_similar_group': 0,
    'PA_model': 'RF',  # RF or MLP×
    'use_BCG': True,
    'use_PI': True,   # 使用PI特征
    'use_FF': True,  # change_by_zsy: 原值True，禁用FF特征
    'use_LAST_BP': False,
    'grid_search': False,
    'pi_csv_path': 'd:/zsy/坐垫/2024_AIdriven/BCG2BP-main/data/introduction.csv',  # add_by_zsy: PI个人信息CSV路径
    'hr_csv_path': r'D:\zsy\坐垫\文献梳理及画图\论文\20250905第二次投递submitCode_npj digital medicine提交\sorted\result\pred_hr_1d_subject_level.csv',
    'normalize_bp': True,  # add_by_zsy: 是否对血压标签进行归一化 (Min-Max到0-1)
    'normalize_method': 'minmax',  # add_by_zsy: 归一化方式 'minmax' 或 'zscore'
    'pcc_weight': 0.0,  # add_by_zsy: PCC损失的权重
}

model_choice = 'OS_BCGNET_RFPA_GridSearch0_Huber_BCG1_PI1_FF1_seed0_N10_zscore1_e20_all_fluc'
# model_choice = 'Test_LOO'

# finetune_strategy = 'sequential'  # sequential; startEnd


# is_finetune = 1

setting = f'{training_scheme}-{signal_type}-{model_choice}'  # “实验配置”，拼在日志/模型/结果文件名里

# add_by_zsy: 使用experiment_name参数填充路径中的子目录
_logger = Logger(args.log % (args.experiment_name, setting, args.session, args.signal), overwrite=args.training)  # 把所有 log 写入文件。如果是训练模式就覆盖旧日志；测试模式追加


def log(*args, **kwargs):
    print(*args, file=_logger, **kwargs)


def main():
    data_processor = DataProcessor(datapath, signal_type, training_scheme, params)  # 负责读 .mat、切分、标准化、构造 DataLoader、组织 fold
    data_processor.load_data()
    train_loaders, test_loaders, fine_tune_loaders = data_processor.get_loaders()   # list 长度等于fold
    
    # add_by_zsy: 输出基于J峰检测的心率MAE统计
    data_processor.print_hr_mae_statistics()

    results_set = []
    # 个性化和SHAP相关变量
    all_shap_values = []
    all_shap_features = []
    
    # add_by_zsy: 用于累积所有fold的评估指标
    all_mae = []
    all_me = []
    all_std = []
    all_pcc = []  # add_by_zsy: 累积PCC指标 (DBP_PCC, SBP_PCC)
    
    # add_by_zsy: 用于收集所有fold的loss历史，用于绘制曲线
    all_train_loss = []
    all_val_loss = []

    for fold_idx, (train_loader, test_loader, fine_tune_loader) in enumerate(
            zip(train_loaders, test_loaders, fine_tune_loaders)):

        log(f"Training on Fold: {fold_idx}")

        # 定义模型路径
        # add_by_zsy: 添加experiment_name到路径中
        model_path = args.model % (args.experiment_name, setting, fold_idx)  # %是字符串占位符，填入experiment_name, setting和fold_idx
        log(f'model_path:{model_path}')
        filepath, _ = os.path.split(model_path)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        # 定义模型
        model = BCGNet()
        model.to(device)

        # 训练

        if args.training:
            log('training mode')
            _, train_loss_hist, val_loss_hist = train_DL(log, model, model_path, train_loader, test_loader, args.lr, args.epoch, params) # change_by_zsy
            # add_by_zsy: 收集loss历史
            all_train_loss.append(train_loss_hist)
            all_val_loss.append(val_loss_hist)
            
            # add_by_zsy: 每个fold训练完成后立即绘制并保存loss曲线图
            if len(train_loss_hist) > 0:
                import matplotlib
                matplotlib.use('Agg')  # 使用非交互式后端
                import matplotlib.pyplot as plt
                
                # 创建图表
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(train_loss_hist) + 1)
                
                plt.plot(epochs, train_loss_hist, 'b-', label='Training Loss', linewidth=2)
                plt.plot(epochs, val_loss_hist, 'r-', label='Validation Loss', linewidth=2)
                
                plt.title(f'Training and Validation Loss - Fold {fold_idx}', fontsize=14)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                # 保存图表到saved_model目录
                model_dir = os.path.dirname(model_path)
                loss_plot_path = os.path.join(model_dir, f'{setting}_fold{fold_idx}_loss_curves.png')
                plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
                log(f'Fold {fold_idx} Loss曲线图已保存至: {loss_plot_path}')
                plt.close()
        else:
            log('testing mode')
            model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # add_by_zsy: 训练模式下加载checkpoint
        # if args.training:
        #     model.load_state_dict(torch.load(model_path, map_location=device))
        #     log(f'✓ 已加载训练保存的最佳模型: {model_path}')

        # 获取预测结果
        pred, labels = predict(model, test_loader)

        # add_by_zsy: 如果启用了BP归一化，需要在评估前反归一化
        if data_processor.use_normalize_bp:
            pred = data_processor.denormalize_bp(pred)
            labels = data_processor.denormalize_bp(labels)
            
        mae, mean_error, std_dev, pcc = evaluate(pred, labels)  # change_by_zsy: 添加PCC返回值

        results = {
            'fold_idx': fold_idx,
            'pred': pred,
            'labels': labels,
        }
        results_set.append(results)
        
        # add_by_zsy: 累积评估指标
        all_mae.append(mae)
        all_me.append(mean_error)
        all_std.append(std_dev)
        all_pcc.append(pcc)  # add_by_zsy: 累积PCC [DBP_PCC, SBP_PCC]

        log('[Fold:%d] ' % fold_idx)
        log('MAE: (%.4f, %.4f)' % (mae[0], mae[1]))
        log('ME : (%.4f, %.4f)' % (mean_error[0], mean_error[1]))
        log('STD: (%.4f, %.4f)' % (std_dev[0], std_dev[1]))
        log('PCC: (%.4f, %.4f)' % (pcc[0], pcc[1]))  # add_by_zsy: 输出PCC (DBP, SBP)
        
        '''
        # add_by_zsy: 输出预测值和真实值的统计信息（诊断PCC为0的原因）
        log('--- 预测值统计 ---')
        log('DBP预测: min=%.2f, max=%.2f, mean=%.2f, std=%.4f' % 
            (pred[:, 0].min(), pred[:, 0].max(), pred[:, 0].mean(), pred[:, 0].std()))
        log('SBP预测: min=%.2f, max=%.2f, mean=%.2f, std=%.4f' % 
            (pred[:, 1].min(), pred[:, 1].max(), pred[:, 1].mean(), pred[:, 1].std()))
        log('--- 真实值统计 ---')
        log('DBP真实: min=%.2f, max=%.2f, mean=%.2f, std=%.4f' % 
            (labels[:, 0].min(), labels[:, 0].max(), labels[:, 0].mean(), labels[:, 0].std()))
        log('SBP真实: min=%.2f, max=%.2f, mean=%.2f, std=%.4f' % 
            (labels[:, 1].min(), labels[:, 1].max(), labels[:, 1].mean(), labels[:, 1].std()))
        
        # 输出前10个样本用于检查
        log('--- 前10个样本对比 ---')
        log('DBP: 预测 vs 真实')
        for i in range(min(10, len(pred))):
            log('  [%d] %.2f vs %.2f' % (i, pred[i, 0], labels[i, 0]))
        log('SBP: 预测 vs 真实')
        for i in range(min(10, len(pred))):
            log('  [%d] %.2f vs %.2f' % (i, pred[i, 1], labels[i, 1]))
        '''
        # 个性化训练部分
        if params['is_finetune']:
            if params['PA_model'] == 'MLP':
                pass  # PA_results_set = training_PA2(log, model, test_loader, fine_tune_loader, data_processor, params)
            else:
                PA_results_set, shap_values, shap_features = training_PA(log, model, test_loader, fine_tune_loader, data_processor, params)
                all_shap_values.append(shap_values)
                all_shap_features.append(shap_features)
            
            # add_by_zsy: 确保PA结果保存目录存在
            pa_result_path = args.savemat % (args.experiment_name, setting, args.session, args.signal, str(fold_idx) + '_PA_' + params['finetune_strategy'])
            pa_result_dir = os.path.dirname(pa_result_path)
            if not os.path.exists(pa_result_dir):
                os.makedirs(pa_result_dir)
            
            savemat(pa_result_path, {'PA_results': PA_results_set, 'Params': params})
        
        # add_by_zsy: 保存当前fold的基础模型结果
        base_result_path = args.savemat % (args.experiment_name, setting, args.session, args.signal, f'base_model_fold{fold_idx}')
        base_result_dir = os.path.dirname(base_result_path)
        if not os.path.exists(base_result_dir):
            os.makedirs(base_result_dir)
        savemat(base_result_path, {'results': results})
        log(f'Fold {fold_idx} 基础模型结果已保存至 {base_result_path}')

    # add_by_zsy: 计算并输出所有fold的平均指标
    all_mae = np.array(all_mae)
    all_me = np.array(all_me)
    all_std = np.array(all_std)
    all_pcc = np.array(all_pcc)  # add_by_zsy: 转换为numpy数组
    
    avg_mae = np.mean(all_mae, axis=0)
    avg_me = np.mean(all_me, axis=0)
    avg_std = np.mean(all_std, axis=0)
    avg_pcc = np.mean(all_pcc, axis=0)  # add_by_zsy: 分别对DBP_PCC和SBP_PCC求平均
    
    log('\n' + '='*60)
    log('所有 %d 个Fold的平均结果:' % len(all_mae))
    log('='*60)
    log('平均MAE: (%.4f, %.4f)' % (avg_mae[0], avg_mae[1]))
    log('平均ME : (%.4f, %.4f)' % (avg_me[0], avg_me[1]))
    log('平均STD: (%.4f, %.4f)' % (avg_std[0], avg_std[1]))
    log('平均PCC: (%.4f, %.4f)' % (avg_pcc[0], avg_pcc[1]))  # add_by_zsy: (DBP_PCC, SBP_PCC)
    log('='*60 + '\n')

    # SHAP可解释性分析
    shapVisual(all_shap_values, all_shap_features)   # 可解释性分析：使用 SHAP 值分析特征重要性，可视化收缩压和舒张压的 Top 12 重要特征


def shapVisual(all_shap_values, all_shap_features):

    feature_names = ['Gender','Age','Weight(kg)','Height(cm)',
                     'T(H,I)', 'T(H,J)', 'T(H,K)', 'T(H,L)', 'T(I,J)', 'T(I,K)', 'T(I,L)', 'T(J,K)', 'T(J,L)', 'T(K,L)',
                     'T(H,I)/T', 'T(H,J)/T', 'T(H,K)/T', 'T(I,J)/T', 'T(I,K)/T', 'T(I,L)/T', 'T(J,K)/T', 'T(J,L)/T',
                     'T(K,L)/T',
                     'A(H)', 'A(I)', 'A(J)', 'A(K)', 'A(L)',
                     '|A(H)-A(I)|', '|A(H)-A(J)|', '|A(H)-A(K)|', '|A(H)-A(L)|', '|A(I)-A(J)|', '|A(I)-A(K)|',
                     '|A(I)-A(L)|', '|A(J)-A(K)|', '|A(J)-A(L)|', '|A(K)-A(L)|',
                     'AUC(H,I)', 'AUC(H,J)', 'AUC(H,K)', 'AUC(H,L)', 'AUC(I,J)', 'AUC(I,K)', 'AUC(I,L)', 'AUC(J,K)',
                     'AUC(J,L)', 'AUC(K,L)']

    # 合并SHAP值
    all_shap_values_dbp = np.vstack([values[0] for values in all_shap_values])
    all_shap_values_sbp = np.vstack([values[1] for values in all_shap_values])

    # 合并特征数据
    all_feature_data = np.vstack(all_shap_features)

    # 计算所有特征的平均绝对SHAP值
    mean_abs_shap_values_dbp = np.abs(all_shap_values_dbp).mean(axis=0)
    mean_abs_shap_values_sbp = np.abs(all_shap_values_sbp).mean(axis=0)

    # 对每个类别获取最重要的12个特征的索引
    top_indices_dbp = np.argsort(-mean_abs_shap_values_dbp)[:12]
    top_indices_sbp = np.argsort(-mean_abs_shap_values_sbp)[:12]

    # 选择最重要的特征和对应的名称
    top_features_dbp = all_feature_data[:, top_indices_dbp]
    top_features_sbp = all_feature_data[:, top_indices_sbp]
    top_feature_names_dbp = [feature_names[i] for i in top_indices_dbp]
    top_feature_names_sbp = [feature_names[i] for i in top_indices_sbp]

    # 生成合并后的SHAP值的可视化图，只包含最重要的12个特征
    plt.figure(figsize=(13, 8))
    shap.summary_plot(all_shap_values_dbp[:, top_indices_dbp], top_features_dbp, feature_names=top_feature_names_dbp)
    plt.figure(figsize=(13, 8))
    shap.summary_plot(all_shap_values_sbp[:, top_indices_sbp], top_features_sbp, feature_names=top_feature_names_sbp)
    print('Visualized!')

if __name__ == '__main__':
    main()
