classdef TestGetLearningRate < matlab.unittest.TestCase
    
    methods (Test)
        
        function testScalarLearningRate(testCase)      
            % check that getLearningRate works in the 
            % case of a scalar learning rate.
            scalarLearningRate = 0.001;
            sampleEpoch = 27;
            opts.learningRate = scalarLearningRate;
            learningRate = getLearningRate(sampleEpoch, opts);
            expected = scalarLearningRate;
            testCase.verifyEqual(learningRate, expected);
        end
        
        function testVectorLearningRate(testCase)      
            % check that getLearningRate works for a vector 
            % learning rate.
            sampleEpoch = 39;
            opts.numEpochs = 250;
            scalarLearningRate = 0.0001;
            vectorLearningRate = ones(1, opts.numEpochs) * scalarLearningRate;
            opts.learningRate = vectorLearningRate;
            learningRate = getLearningRate(sampleEpoch, opts);
            expected = scalarLearningRate;
            testCase.verifyEqual(learningRate, expected);
            
            % check that getLearningRate throws an error
            % for invalid vector learning rates.
            vectorLearningRate = ones(1, opts.numEpochs - 1) * scalarLearningRate;
            opts.learningRate = vectorLearningRate;
            errorId = 'learningRate:size';
            testCase.verifyError(@() getLearningRate(sampleEpoch, opts), errorId);
            
        end
    end
end