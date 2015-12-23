classdef TestFindLastCheckpoint < matlab.unittest.TestCase
    
    methods (Test)
        
        function testLastCheckpoint(testCase)                  
            % check that the last checkpoint is calculated correctly.
            opts.expDir = GetFullPath('../data/savedEpochs');
            expectedEpoch = 361;
            epoch = findLastCheckpoint(opts);
            testCase.verifyEqual(expectedEpoch, epoch);
        end
    end
end





