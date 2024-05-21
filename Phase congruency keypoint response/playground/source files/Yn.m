image = im2uint8(imread('images/Yn.png'));
[~,~,~,~,~,eo,~] = phasecong3(image,4,6,3,'mult',1.6,'sigmaOnf',0.75,'g', 3, 'k',1); 

% eo �� 4x6 �� cell ����
normalized_eo = cell(size(eo));  % ����һ���µ�ͬ����С�� cell �������ڴ洢���

for i = 1:size(eo, 1)
    for j = 1:size(eo, 2)
        currentMatrix = abs(eo{i, j});  % ��ȡ��ǰ cell �ľ���ֵ
        minVal = min(currentMatrix(:));  % ������Сֵ
        maxVal = max(currentMatrix(:));  % �������ֵ

        % ��һ�������ŵ� [0, 1]
        normalized_eo{i, j} = (currentMatrix - minVal) / (maxVal - minVal);
    end
end

                                                                                                                   

% ���������к���
numRows = size(normalized_eo, 1);
numCols = size(normalized_eo, 2);

for i = 1:numRows
    for j = 1:numCols

        % ��ȡ��ǰͼ������
        img_normalized = normalized_eo{i, j};

        % ת��Ϊ uint8 ��ʽ���ʺϱ���
        img_uint8 = im2uint8(img_normalized);

        % �����ļ���
        filename = sprintf('E:/code/LGHD-master/playground/result/Yn/normalized_image_%d_%d.png', i, j);

        % ����ͼ���ļ�
        imwrite(img_uint8, filename);
    end
end



numRows = 4;
numMatrices = 6;
[numElementsRow, numElementsCol] = size(normalized_eo{1, 1}); % ��ȡ��������ĳߴ�
resultMatrices = cell(numElementsRow, numElementsCol); % ��������洢cell

% ��ʼ�����cell����
for r = 1:numElementsRow
    for c = 1:numElementsCol
        resultMatrices{r, c} = zeros(numRows, numMatrices);
    end
end


% ����ѭ������������ÿ�����������Ӧλ�õ��¾���
% ������ѭ������normalized_eo�е��к��С�
% ������ѭ���������������ÿ��Ԫ�أ�����������λ�÷ŵ���Ӧ���¾����С�
for i = 1:numRows
    for j = 1:numMatrices
        nextIndex = mod(j, numMatrices) + 1;
        diffMatrix = normalized_eo{i, j} - normalized_eo{i, nextIndex};  % �����ֵ
        % ����ֵ�����ÿ��Ԫ�طŵ���Ӧ���¾�����
        for r = 1:numElementsRow
            for c = 1:numElementsCol
                resultMatrices{r, c}(i, j) = diffMatrix(r, c);
            end
        end
    end
end


% ����resultMatrices�е�ÿ������
for i = 1:size(resultMatrices, 1)
    for j = 1:size(resultMatrices, 2)
        % ��ȡ��ǰ����
        currentMatrix = resultMatrices{i, j};
        
        % ��С��0��ֵ��Ϊ0������0��ֵ��Ϊ1
        currentMatrix(abs(currentMatrix) < 0.1) = 0;
        currentMatrix(abs(currentMatrix) > 0.1) = 1;
        
        % ����resultMatrices�еĵ�ǰ����
        resultMatrices{i, j} = currentMatrix;
    end
end


% ��ʼ���洢���������
numRows = size(resultMatrices, 1);
numCols = size(resultMatrices, 2);
onePercentage = zeros(numRows, numCols);

% ����ÿ�����󣬼���1������ռ��
for i = 1:numRows
    for j = 1:numCols
        % ��ȡ��ǰ����
        currentMatrix = resultMatrices{i, j};
        
        % ���������1������
        numOnes = sum(currentMatrix(:) == 1);
        
        % ����1������ռ��
        totalElements = numel(currentMatrix); % ��Ԫ������
        onePercentage(i, j) = numOnes / totalElements;
    end
end




% ������ɫӳ��
colormapName = 'jet'; % ��ɫӳ������
cmap = colormap(colormapName); % ��ȡ��ɫӳ��
close; % �رյ�ǰ��figure���ڣ���Ϊ����ֻ��Ҫcolormap����

% ��onePercentage����ת��Ϊ����ͼ��
indexedImage = gray2ind(onePercentage, size(cmap, 1));

% ������ͼ��ת��ΪRGBͼ��
onePercentage_Image = ind2rgb(indexedImage, cmap);
savePath = 'E:/code/LGHD-master/playground/result/Yn/onePercentage_Image.png'
% ����RGBͼ��
imwrite(onePercentage_Image, savePath);