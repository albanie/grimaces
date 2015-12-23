function localTestRunner

% setup path to mock objects
addpath(GetFullPath('mockObjects'));

% setup paths to source code
addpath(GetFullPath('../IO'));
addpath(GetFullPath('../utils'));
addpath(GetFullPath('../visualization'));
addpath(GetFullPath('../dag'));
addpath(GetFullPath('../../matlab'));
vl_setupnn;

%-----------------------------------
% SLOW TESTS 
%-----------------------------------
runtests('IO'); 
runtests('visualization');

%-----------------------------------
% FAST TESTS 
%-----------------------------------
runtests('utils/');
runtests('dag');

end
