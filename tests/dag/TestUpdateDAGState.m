classdef TestUpdateDAGState < matlab.unittest.TestCase
    
    methods (Test)
        
        function testDAGStateUpdatedCorrectly(testCase)                  
            % check that DAG state is updated correctly
            epoch = 4;
            opts.learningRate = 0.01;
            opts.train = [1:100];
            opts.val = [101:120];
            state = struct();
            updatedState = updateDAGState(state, epoch, opts);
            
            % check each property of updatedState
            testCase.verifyEqual(updatedState.epoch, epoch);
        end
    end
end
