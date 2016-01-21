function [data, labels] = getFaceData(opts, dagnet, partition)
% returns the data and associated labels contained in the raw dataset

imgNames = get_jpgs_in_subdirs(fullfile(opts.dataDir, filesep, partition));

% retrieve faces that meet the minimum size criteria
large_faces = select_large_faces(imgNames, opts.minFaceArea);

% retrieve the label for each face
labels = getBinaryGradientLabels(large_faces, opts);

% format the data to conform to the pretrained network
averageImage = dagnet.meta.normalization.averageImage;
imgSize = dagnet.meta.normalization.imageSize;
data = zeros(imgSize(1),imgSize(2),imgSize(3), ...
                numel(large_faces), 'single');

% Modify the average image when we are using vgg faces
newAverageImage = zeros(horzcat(imgSize(1:2),3));
if numel(dagnet.layers) == 37
    for layer = 1:3
        newAverageImage(:,:,layer) = ones(imgSize(1:2)) * averageImage(:,:,layer);
    end
    averageImage = newAverageImage;
end


for i = 1:numel(large_faces)
    rawFace = imread(large_faces{i}{1});
    resizedFace = single(imresize(rawFace, dagnet.meta.normalization.imageSize(1:2)));
    normalizedFace = resizedFace - averageImage;
    data(:,:,:,i) = normalizedFace(:,:,:);
end

end

% --------------------------------------------------------------------
function faces = normalizeImageSizes(faces, fixed_size)
% --------------------------------------------------------------------
for i = 1: numel(faces)
    face = imread(faces{i})
    faces{i} = imresize(face, fixed_size)
end
end
