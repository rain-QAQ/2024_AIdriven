%% MATLAB 脚本：将 table 对象转换为 struct 并保存为 v7.3 格式
%% 这样 Python 就可以正确读取了

% 加载原始文件
load('Preprocessed_Database.mat')

% 查看加载的变量
fprintf('加载的变量:\n');
whos

% 将 table 转换为结构数组（推荐）
% 这种方法将 table 的每一行转换为一个 struct，保持列名作为字段名
preprocessed_Data_struct = table2struct(Preprocessed_Database);

% 保存为 v7.3 格式（Python 可以用 h5py 或 hdf5storage 读取）
save('Preprocessed_Database_v73.mat', 'preprocessed_Data_struct', '-v7.3');

fprintf('✅ 已保存为: Preprocessed_Database_v73.mat\n');
fprintf('文件大小: %.2f MB\n', dir('Preprocessed_Database_v73.mat').bytes / 1024 / 1024);

% 验证保存的数据
verify_data = load('Preprocessed_Database_v73.mat');
fprintf('受试者数量: %d\n', length(verify_data.preprocessed_Data_struct));
fprintf('字段名: ');
disp(fieldnames(verify_data.preprocessed_Data_struct));

fprintf('\n💡 现在可以在 Python 中使用以下代码加载：\n');
fprintf('  from hdf5storage import loadmat\n');
fprintf('  mat = loadmat("Preprocessed_Database_v73.mat")\n');
