function labels = get_binary_gradient_labels(imgPaths)
   
    labels = cellfun(@extract_gradient, imgPaths)'; 
    
end

function label = extract_gradient(imgPath)
    
    % The gradient is encoded as the last characters in the 
    % filename, so we do some string processing to extract it.
    % The format is "stageId-HH_MM:SS:I:G.G"
    % For example, the filename "1-00_11:27:0:0.4"
    % First we extract the tail of the path to get the image name
    [~,name,~] = fileparts(imgPath{1});
    
    % Split the string using the colon delimiter and pick the last
    % element
    splits = strsplit(name, ':');
    gradient_str = splits{end};
    
    % convert the string to a number
    gradient = str2num(gradient_str);
    
    % Finally, set label based on the gradient as folows:
    if gradient < 2.235
        label = 1;
    else
        label = 2;
    end
     
end
