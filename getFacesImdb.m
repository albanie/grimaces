function imdb = getFacesImdb(opts, net)
% --------------------------------------------------------------------

% Define the minimum face size that will be accepted by the network
minFaceArea = 3600;

% get faces and labels
[trainData, trainLabels] = getFaceData(opts, minFaceArea, net, 'training');
[valData, valLabels] = getFaceData(opts, minFaceArea, net, 'validation');

% get the set labels for the training data
trainSet = cell(1, size(trainLabels,2));
train_label = 1;
[trainSet{:}] = deal(train_label);

% get the set labels for the validation data
valSet = cell(1, size(valLabels,2));
val_label = 2;
[valSet{:}] = deal(val_label);

% create the imdb training/validation structure expected by matconvnet
train_val_data = cat(4, trainData, valData);
train_val_labels = cat(2, trainLabels, valLabels); 
train_val_set = cat(2, trainSet, valSet);

imdb.images.data = train_val_data;
imdb.images.labels = train_val_labels;
imdb.images.set = train_val_set;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = 1:2;

end
