image = im2uint8(imread('images/Yn.png'));
[~,~,~,~,~,eo,~] = phasecong3(image,4,6,3,'mult',1.6,'sigmaOnf',0.75,'g', 3, 'k',1); 

% eo 是 4x6 的 cell 数组
normalized_eo = cell(size(eo));  % 创建一个新的同样大小的 cell 数组用于存储结果

for i = 1:size(eo, 1)
    for j = 1:size(eo, 2)
        currentMatrix = abs(eo{i, j});  % 获取当前 cell 的绝对值
        minVal = min(currentMatrix(:));  % 计算最小值
        maxVal = max(currentMatrix(:));  % 计算最大值

        % 归一化并缩放到 [0, 1]
        normalized_eo{i, j} = (currentMatrix - minVal) / (maxVal - minVal);
    end
end

                                                                                                                   

% 遍历所有行和列
numRows = size(normalized_eo, 1);
numCols = size(normalized_eo, 2);

for i = 1:numRows
    for j = 1:numCols

        % 获取当前图像数据
        img_normalized = normalized_eo{i, j};

        % 转换为 uint8 格式，适合保存
        img_uint8 = im2uint8(img_normalized);

        % 创建文件名
        filename = sprintf('E:/code/LGHD-master/playground/result/Yn/normalized_image_%d_%d.png', i, j);

        % 保存图像到文件
        imwrite(img_uint8, filename);
    end
end



numRows = 4;
numMatrices = 6;
[numElementsRow, numElementsCol] = size(normalized_eo{1, 1}); % 获取单个矩阵的尺寸
resultMatrices = cell(numElementsRow, numElementsCol); % 创建结果存储cell

% 初始化结果cell数组
for r = 1:numElementsRow
    for c = 1:numElementsCol
        resultMatrices{r, c} = zeros(numRows, numMatrices);
    end
end


% 计算循环相减结果并将每个结果存入相应位置的新矩阵
% 外两层循环遍历normalized_eo中的行和列。
% 内两层循环遍历单个矩阵的每个元素，将相减结果按位置放到对应的新矩阵中。
for i = 1:numRows
    for j = 1:numMatrices
        nextIndex = mod(j, numMatrices) + 1;
        diffMatrix = normalized_eo{i, j} - normalized_eo{i, nextIndex};  % 计算差值
        % 将差值矩阵的每个元素放到对应的新矩阵中
        for r = 1:numElementsRow
            for c = 1:numElementsCol
                resultMatrices{r, c}(i, j) = diffMatrix(r, c);
            end
        end
    end
end


% 遍历resultMatrices中的每个矩阵
for i = 1:size(resultMatrices, 1)
    for j = 1:size(resultMatrices, 2)
        % 获取当前矩阵
        currentMatrix = resultMatrices{i, j};
        
        % 将小于0的值设为0，大于0的值设为1
        currentMatrix(abs(currentMatrix) < 0.1) = 0;
        currentMatrix(abs(currentMatrix) > 0.1) = 1;
        
        % 更新resultMatrices中的当前矩阵
        resultMatrices{i, j} = currentMatrix;
    end
end


% 初始化存储结果的数组
numRows = size(resultMatrices, 1);
numCols = size(resultMatrices, 2);
onePercentage = zeros(numRows, numCols);

% 遍历每个矩阵，计算1的数量占比
for i = 1:numRows
    for j = 1:numCols
        % 获取当前矩阵
        currentMatrix = resultMatrices{i, j};
        
        % 计算矩阵中1的数量
        numOnes = sum(currentMatrix(:) == 1);
        
        % 计算1的数量占比
        totalElements = numel(currentMatrix); % 总元素数量
        onePercentage(i, j) = numOnes / totalElements;
    end
end




% 设置颜色映射
colormapName = 'jet'; % 颜色映射名称
cmap = colormap(colormapName); % 获取颜色映射
close; % 关闭当前的figure窗口，因为我们只需要colormap数据

% 将onePercentage矩阵转换为索引图像
indexedImage = gray2ind(onePercentage, size(cmap, 1));

% 将索引图像转换为RGB图像
onePercentage_Image = ind2rgb(indexedImage, cmap);
savePath = 'E:/code/LGHD-master/playground/result/Yn/onePercentage_Image.png'
% 保存RGB图像
imwrite(onePercentage_Image, savePath);