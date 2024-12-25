clc; clear;

%% Reading the MNIST dataset
% Load the data
[trainImages, trainLabels] = loadMNISTData('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[testImages, testLabels] = loadMNISTData('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

% Display some random training images
figure;
numImagesToShow = 10;
for i = 1:numImagesToShow
    subplot(2, 5, i);
    idx = randi(size(trainImages, 3));
    imshow(trainImages(:, :, idx), []);
    title(sprintf('Label: %d', trainLabels(idx)));
end
sgtitle('Some Random Train Images');

%% Dataset Manipulation
% Flatten the images to 1D vectors (28x28 = 784 features)
% Used all pixels as train || feature extraction could be used
trainData = reshape(trainImages, [], size(trainImages, 3))';
testData = reshape(testImages, [], size(testImages, 3))';

% Normalizing the dataset
trainData = double(trainData) / 255;
testData = double(testData) / 255;

% I also tried 1/10 of the dataset due to time issues. For my pc configuration;
% For 1/10 of the dataset, Test Accuracy: 90.50%, Elapsed Time: 1.63 seconds
% For all dataset, Test Accuracy: 94.38%, Elapsed Time: 187.23 seconds

% Comment these 4 lines to try for the whole dataset
trainData = trainData(1:6000,:);
trainLabels = trainLabels(1:6000,:);
testData = testData(1:1000,:);
testLabels = testLabels(1:1000,:);

tic;
svmModel = fitcecoc(trainData,trainLabels);

% Predict the labels for the test set
predictedLabels = predict(svmModel, testData);

% Test accuracy
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

elapsedTime = toc;
fprintf('Elapsed Time: %.2f seconds\n', elapsedTime);

%% Functions

function [images, labels] = loadMNISTData(imagesPath, labelsPath)
    labels = readMNISTLabels(labelsPath);
    images = readMNISTImages(imagesPath);
end

function images = readMNISTImages(filePath)
    fid = fopen(filePath, 'rb');
    magicNumber = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNumber ~= 2051
        error('Invalid magic number for images, expected 2051, got %d', magicNumber);
    end
    
    numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fid, 1, 'int32', 0, 'ieee-be');
    
    rawData = fread(fid, numImages * numRows * numCols, 'uint8');
    images = reshape(rawData, [numRows, numCols, numImages]);
    fclose(fid);
    
    % Correct the orientation: Transpose and flip
    for i = 1:numImages
        images(:, :, i) = rot90(images(:, :, i), -1); % Rotate 90 degrees CW
        images(:, :, i) = fliplr(images(:, :, i));   % Flip left-to-right
    end
end

function labels = readMNISTLabels(filePath)
    fid = fopen(filePath, 'rb');
    magicNumber = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNumber ~= 2049
        error('Invalid magic number for labels, expected 2049, got %d', magicNumber);
    end
    
    numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels = fread(fid, numLabels, 'uint8');
    fclose(fid);
end
