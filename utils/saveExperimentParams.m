function saveExperimentParams(opts, mode)
% saves the parameters that were used to setup the experiment to disk.
% generalOpts will be the same for both training and testing.
% modeOpts are the options specific to the mode.

targetDir = strcat(opts.expDir, '/experimentParams/');

% create the directory if it doesn't exist yet
if ~exist(targetDir, 'dir') 
    mkdir(targetDir);
end
    
if strcmp(mode, 'train')
    save(strcat(targetDir, 'train.mat'), 'opts');
else
    save(strcat(targetDir, 'test.mat'), 'opts');
end