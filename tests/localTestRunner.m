function localTestRunner

% setup paths and code
addpath(GetFullPath('../IO'));
addpath(GetFullPath('../utils'));
addpath(GetFullPath('../visualization'));
addpath(GetFullPath('../../matlab'));
vl_setupnn;

% run tests on each package
runtests('utils/');
% runtests('IO');
runtests('visualization');

end
