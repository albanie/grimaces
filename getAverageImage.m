% --------------------------------------------------------------------
function averageImage = getAverageImage(faces, fixed_size)
% --------------------------------------------------------------------

resized_faces = zeros(numel(faces), fixed_size(1), fixed_size(2), fixed_size(3));

for i = 1:numel(faces)
    raw_face = imread(faces{i}{1});
    resized_face = imresize(raw_face, fixed_size(1:2));
    resized_faces(i,:,:,:) = resized_face(:,:,:);
end

% Take the mean across all faces and set image dimensions appropriately
averageImage = single(mean(resized_faces,1));
averageImage = reshape(averageImage, fixed_size);
end