%% MATLAB 脚本：展开嵌套的 table 结构
%% 将 Preprocessed_Database 中嵌套的 FilteredData table 展开为平铺的 struct

% 加载原始文件
fprintf('正在加载原始文件...\n');
load('Preprocessed_Database.mat')

% 显示 table 结构
fprintf('Table 大小: %d 行 x %d 列\n', height(Preprocessed_Database), width(Preprocessed_Database));
fprintf('列名: ');
disp(Preprocessed_Database.Properties.VariableNames);

% 获取受试者数量
num_subjects = height(Preprocessed_Database);
fprintf('受试者数量: %d\n', num_subjects);

% 创建结构数组来存储展开后的数据
preprocessed_Data_struct = struct();

% 遍历每个受试者
for i = 1:num_subjects
    fprintf('处理受试者 %d/%d...\n', i, num_subjects);
    
    % 获取 ID
    preprocessed_Data_struct(i).ID = Preprocessed_Database.ID{i};
    
    % 获取 FilteredData (这是一个嵌套的 table)
    filtered_table = Preprocessed_Database.FilteredData{i};
    
    % 检查 FilteredData 是否为 table
    if istable(filtered_table)
        % 将嵌套的 table 展开为 struct 的字段
        % 获取 FilteredData table 的所有列名
        col_names = filtered_table.Properties.VariableNames;
        
        % 将每一列作为 struct 的字段
        for j = 1:length(col_names)
            col_name = col_names{j};
            preprocessed_Data_struct(i).(col_name) = filtered_table.(col_name);
        end
    else
        % 如果不是 table，直接存储
        preprocessed_Data_struct(i).FilteredData = filtered_table;
    end
end

% 显示第一个受试者的字段
fprintf('\n第一个受试者的字段:\n');
disp(fieldnames(preprocessed_Data_struct(1)));

% 保存为 v7.3 格式
fprintf('\n正在保存为 v7.3 格式...\n');
save('Preprocessed_Database_flat.mat', 'preprocessed_Data_struct', '-v7.3');

fprintf('✅ 保存成功: Preprocessed_Database_flat.mat\n');
info = dir('Preprocessed_Database_flat.mat');
fprintf('文件大小: %.2f MB\n', info.bytes / 1024 / 1024);

fprintf('\n💡 现在可以在 Python 中加载此文件了！\n');
