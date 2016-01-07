classdef TestUpdateDAGState < matlab.unittest.TestCase
    
    methods (Test)
        
        function testDAGStateUpdatedCorrectly(testCase)                  
            % check that DAG state is updated correctly
            epoch = 4;
            opts.learningRate = 0.01;
            opts.train = [1:100];
            opts.val = [101:120];
            state = struct();
            state.imdb.images.labels = horzcat(ones(1,60), ...
                                                2 * ones(1,40), ...
                                                ones(1,10), ...
                                                2 * ones(1,10));
            updatedState = updateDAGState(state, epoch, opts);
            
            % check each property of updatedState
            testCase.verifyEqual(updatedState.epoch, epoch);
        end
    end
end
