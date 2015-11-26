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
function faces = normalizeImageSizes(faces, fixed_size)
% --------------------------------------------------------------------
    for i = 1: numel(faces)
        face = imread(faces{i})
        faces{i} = imresize(face, fixed_size)
    end
end
