classdef TestComputeBatch < matlab.unittest.TestCase
    
    properties
        subset
        opts
        subBatchIdx
    end
 
    methods(TestMethodSetup)
        function setUpExampleParameters(testCase)
            % set up some example parameters
            testCase.subset = 1:1000;
            testCase.opts.batchSize = 256;
            testCase.opts.numSubBatches = 1;
            testCase.subBatchIdx = 1;
        end
    end
    
    methods (Test)
        
        function testFirstBatchIndices(testCase)      
            % check that batch indices are calculated 
            % correctly for the first batch.
            subsetIdx = 1;
            expectedBatch = 1:256;
            batch = computeBatch(subsetIdx, ...
                                testCase.subBatchIdx, ...
                                testCase.subset, ...
                                testCase.opts);
            testCase.verifyEqual(expectedBatch, batch);
        end
        
        function testLastBatchIndices(testCase)      
            % check that batch indices are calculated 
            % correctly for the last batch.
            subsetIdx = 769;
            expectedBatch = 769:1000;
            batch = computeBatch(subsetIdx, ...
                                testCase.subBatchIdx, ...
                                testCase.subset, ...
                                testCase.opts);
            testCase.verifyEqual(expectedBatch, batch);
        end
        
        function testDifferentSubBatchIndices(testCase)      
            % check that batch indices are calculated 
            % correctly when different subBatchIdx is used.
            subsetIdx = 1;
            subBatchIdx = 2;
            testCase.opts.numSubBatches = 2;
            expectedBatch = 2:2:256;
            batch = computeBatch(subsetIdx, ...
                                subBatchIdx, ...
                                testCase.subset, ...
                                testCase.opts);
            testCase.verifyEqual(expectedBatch, batch);
        end
    end
end