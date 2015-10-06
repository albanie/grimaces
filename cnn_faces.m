function [net, info] = cnn_faces(varargin)

% setup the MatConvNet toolbox and add utils
vl_setupnn;
addpath('../examples');
addpath('utils');

% set destination path for results
opts.expDir = fullfile('data', 'results');
[opts, varargin] = vl_argparse(opts, varargin);

% set path to the raw input data
opts.dataDir = fullfile('data', 'faces');

% set path to where preprocessed data will be stored
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

% set training parameters for the CNN
opts.useBnorm = false;
opts.fineTuningRate = 0.05;
opts.train.batchSize = 100;
opts.train.numEpochs = 20;
opts.train.continue = false;
opts.train.gpus = [];
opts.train.learningRate = 0.001;
opts.train.expDir = opts.expDir;
opts = vl_argparse(opts, varargin);

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

net = cnn_alexnet_init(opts);
% net = dagnn.DagNN.fromSimpleNN(net) ;
% net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
%              {'prediction','label'}, 'error') ;
         
%if the imdb file has already been created, load into memory.
% Otherwise, build it from scratch.
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath)
else
    imdb = getFacesImdb(opts, net);
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
end

% imdb = getFacesImdb(opts, net);

% 

% --------------------------------------------------------------------
%                                                     Evaluate network
% --------------------------------------------------------------------
trainIdx = find(strcmp(imdb.images.set, 'train'));
valIdx = find(strcmp(imdb.images.set, 'val'));
testIdx = find(strcmp(imdb.images.set, 'test'));

info = 'loaded the data';
info = cnn_train_face_dag(net, imdb, @getBatch, opts.train, ...
                    'train', trainIdx, ...
                    'val',  valIdx)

end


% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------


input = imdb.images.data(:,:,:,batch);
label = imdb.images.labels(1,batch);
inputs = { 'input', input, ...
           'label', label};

end



% --------------------------------------------------------------------
function imdb = getFacesImdb(opts, net)
% --------------------------------------------------------------------

% Define the minimum face size that will be accepted by the network
min_face_area = 3600;
[data, labels] = getFaceData(opts, min_face_area, net);
set = getFaceSet(labels);

imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = 1:2;

end


% --------------------------------------------------------------------
function [data, labels] = getFaceData(opts, min_face_area, net)
% --------------------------------------------------------------------
    
imgNames = get_jpgs_in_dir(fullfile(opts.dataDir, filesep));

% retrieve faces that meet the minimum size criteria
large_faces = select_large_faces(imgNames, min_face_area);


labels = get_binary_gradient_labels(large_faces);


% set fixed image size for network
% fixed_size = [120 120 3];

% averageImage = getAverageImage(large_faces, fixed_size);
averageImage = net.meta.normalization.averageImage;

[x, y, z] = size(averageImage);
data = zeros(x,y,z, numel(large_faces), 'single');



for i = 1:numel(large_faces)
    raw_face = imread(large_faces{i}{1});  
%     resized_face = single(imresize(raw_face, fixed_size(1:2)));
%     normalized_face = bsxfun(@minus, resized_face, averageImage);
    resized_face = single(imresize(raw_face, net.meta.normalization.imageSize(1:2)));
    normalized_face = resized_face - averageImage;
    data(:,:,:,i) = normalized_face(:,:,:);
end

end

% --------------------------------------------------------------------
function set = getFaceSet(labels)
% --------------------------------------------------------------------
    % define the training:validation:test ratio 
    train = 0.8;
    val = 0.2;
    
    % find the splitting indices for the data
    num_labels = numel(labels);
    train_val_split = ceil(train * num_labels);
    val_test_split = train_val_split + ceil(val * num_labels);
    
    % create a cell array to hold the set markers
    set = cell(1, num_labels);
    [set{1:train_val_split - 1}] = deal('train');
    [set{train_val_split:val_test_split -1 }] = deal('val');
    [set{val_test_split:end}] = deal('test');
end

% --------------------------------------------------------------------
function faces = normalizeImageSizes(faces, fixed_size)
% --------------------------------------------------------------------
    for i = 1: numel(faces)
        face = imread(faces{i})
        faces{i} = imresize(face, fixed_size)
    end
end

