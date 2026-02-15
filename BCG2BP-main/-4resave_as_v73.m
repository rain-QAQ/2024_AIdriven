%% MATLAB 脚本：将文件重新保存为 v7.3 格式
%% 这样 Python 可以使用 h5py 读取

% 加载原始文件
fprintf('正在加载原始文件...\n');
load('Preprocessed_Database.mat')

% 直接保存为 v7.3 格式，保留所有变量（包括 table 对象）
fprintf('正在保存为 v7.3 格式...\n');
save('Preprocessed_Database_v73.mat', '-v7.3');

fprintf('✅ 保存成功: Preprocessed_Database_v73.mat\n');

% 显示文件大小
info = dir('Preprocessed_Database_v73.mat');
fprintf('文件大小: %.2f MB\n', info.bytes / 1024 / 1024);

fprintf('\n💡 现在可以在 Python 中使用 h5py 读取此文件\n');
