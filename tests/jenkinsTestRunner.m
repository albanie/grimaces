function jenkinsTestRunner
% This function is called directly by the Jenkins CI
% server.

% setup paths and code
addpath('/Users/Shared/MATLAB_FILE_EXCHANGE/GetFullPath_17Jan2013/')
addpath(GetFullPath('../IO'));
addpath(GetFullPath('../utils'));
addpath(GetFullPath('../../matlab'));
vl_setupnn;

% run tests on each package
runTestSuite('utils');
runTestSuite('IO');

end

function runTestSuite(packageName)

import matlab.unittest.TestSuite;
% run all tests contained within the package specified
% by packageName.
try
    suite = TestSuite.fromPackage(packageName,'IncludingSubpackages',true);
    results = run(suite);
    display(results);
catch e
    disp(getReport(e,'extended'));
    exit(1);
end

% If any test fails, exit with status 1 to alert Jenkins
if any([results.Failed])
    exit(1);
end

end