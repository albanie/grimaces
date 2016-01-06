function [data, labels] = getFaceData(opts, dagnet, partition)
% returns the data and associated labels contained in the raw dataset

imgNames = get_jpgs_in_subdirs(fullfile(opts.dataDir, filesep, partition));

% retrieve faces that meet the minimum size criteria
large_faces = select_large_faces(imgNames, opts.minFaceArea);

% retrieve the label for each face
labels = getBinaryGradientLabels(large_faces, opts);

% format the data to conform to Alexnet
averageImage = dagnet.meta.normalization.averageImage;
[x, y, z] = size(averageImage);
data = zeros(x,y,z, numel(large_faces), 'single');

for i = 1:numel(large_faces)
    raw_face = imread(large_faces{i}{1});
    resized_face = single(imresize(raw_face, dagnet.meta.normalization.imageSize(1:2)));
    normalized_face = resized_face - averageImage;
    data(:,:,:,i) = normalized_face(:,:,:);
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
