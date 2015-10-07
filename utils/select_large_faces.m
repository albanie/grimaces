function large_faces = select_large_faces(img_paths, varargin)
    
    % Define the default minimum area for a face to be classed
    % as a 'large face'
    if nargin < 2
        min_face_area = 3600;
    else
        min_face_area = varargin{1};
    end
    
    large_faces = {};
    for img_path = img_paths'    
        % calculate the number of pixels in each image
        img = imread(img_path{1});
        [num_rows, num_cols, layers] = size(img);
        num_pixels = num_rows * num_cols;
        
        % check that the image meets the size criteria
        if num_pixels > min_face_area
            large_faces = [large_faces; {img_path}];
        end 
    end
    
end