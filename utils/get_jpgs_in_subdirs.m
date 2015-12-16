function img_names = get_jpgs_in_subdirs(path)
% returns an nx1 cell array of paths to any .jpg files
% found recursively in subdirectories of 'path'

% create a cell array of all subdirectories
subdirs = strsplit(genpath(path), ':')';

% extract the jpgs in each of these subdirectories
jpgs = cellfun(@get_jpgs_in_dir, subdirs, 'UniformOutput', false);

% flatten the cell array
img_names = vertcat(jpgs{:});
end

% -----------------------------------------
function img_names = get_jpgs_in_dir(path)
% -----------------------------------------
%List all files in the given directory with .jpg extension
img_files = dir(fullfile(path, '*.jpg'));

% create a cell array of the full paths to each image
img_names = {img_files.name}';
paths = repmat({path}, size(img_names));
img_names = strcat(paths, filesep, img_names);

end
