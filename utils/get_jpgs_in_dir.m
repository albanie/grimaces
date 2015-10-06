function img_names = get_jpgs_in_dir(path)
    %List all files in the given directory with .jpg extension
    img_files = dir(fullfile(path, '*.jpg')); 
    
    % create a cell array of the full paths to each image
    img_names = {img_files.name}';
    paths = repmat({path}, size(img_names));
    img_names = strcat(paths, img_names);