function imdb = getFacesImdb(opts, net)
% --------------------------------------------------------------------

% Define the minimum face size that will be accepted by the network
minFaceArea = 3600;
% [data, labels] = getFaceData(opts, min_face_area, net);

% get faces and labels
[trainData, trainLabels] = getFaceData(opts, minFaceArea, net, 'training');
[testData, testLabels] = getFaceData(opts, minFaceArea, net, 'testing');

% get the set labels for the training data
ratio = 0.8;
trainAndValSet = getTrainAndValSet(trainLabels, ratio);

% get the set labesl for the test data
testSet = cell(1, size(testLabels,2));
[testSet{:}] = deal('test');

% create the imdb structure expected by matconvnet
data = cat(4, trainData, testData);
labels = cat(2, trainLabels, testLabels); 
set = cat(2, trainAndValSet, testSet);

imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = 1:3;
end


% --------------------------------------------------------------------
function [data, labels] = getFaceData(opts, min_face_area, net, partition)
% --------------------------------------------------------------------
    
imgNames = get_jpgs_in_subdirs(fullfile(opts.dataDir, filesep, partition));

% retrieve faces that meet the minimum size criteria
large_faces = select_large_faces(imgNames, min_face_area);

% retrieve the label for each face
labels = get_binary_gradient_labels(large_faces);

% format the data to conform to Alexnet 
averageImage = net.meta.normalization.averageImage;
[x, y, z] = size(averageImage);
data = zeros(x,y,z, numel(large_faces), 'single');

for i = 1:numel(large_faces)
    raw_face = imread(large_faces{i}{1});  
    resized_face = single(imresize(raw_face, net.meta.normalization.imageSize(1:2)));
    normalized_face = resized_face - averageImage;
    data(:,:,:,i) = normalized_face(:,:,:);
end

end

% --------------------------------------------------------------------
function set = getTrainAndValSet(labels, ratio)
% --------------------------------------------------------------------
    % returns a cell array of 'train' and 'val' labels
    % in proprortion to the given ratio.
    
    % find the splitting indices for the data
    num_labels = numel(labels);
    train_val_split = ceil(ratio * num_labels);
    
    % create a cell array to hold the set markers
    set = cell(1, num_labels);
    [set{1:train_val_split - 1}] = deal('train');
    [set{train_val_split:end}] = deal('val');
end

% --------------------------------------------------------------------
function faces = normalizeImageSizes(faces, fixed_size)
% --------------------------------------------------------------------
    for i = 1: numel(faces)
        face = imread(faces{i})
        faces{i} = imresize(face, fixed_size)
    end
end
