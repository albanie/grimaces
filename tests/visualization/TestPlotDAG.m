classdef TestPlotDAG < matlab.unittest.TestCase
    
    properties
        stats
        opts
    end
 
    methods(TestMethodSetup)
        function setUpExampleParameters(testCase)
            % set up some example parameters
            testCase.opts.modelFigPath = '../data/testFig';
            testCase.stats.train(1).prob = 0.55;
            testCase.stats.train(1).error = 0.45;
            testCase.stats.val(1).prob = 0.65;
            testCase.stats.val(1).error = 0.6;            
        end
    end
    
    methods (Test)
        
        function testPlotDAGFirstTrainingEpoch(testCase)      
            % Test plotDAG for the first epoch during training
            epoch = 1;
            testCase.opts.testMode = false;
            plotDAG(testCase.stats, epoch, testCase.opts);
        end
        
         function testPlotDAGFirstTestingEpoch(testCase)      
            % Test plotDAG for the first epoch during testing
            epoch = 1;
            testCase.opts.testMode = true;
            plotDAG(testCase.stats, epoch, testCase.opts);
         end
         
         function testPlotDAGDuringSecondTrainingEpoch(testCase)      
            % Test plotDAG for the first epoch during testing
            epoch = 2;
            % create additional stats
            testCase.stats.train(2).prob = 0.5;
            testCase.stats.train(2).error = 0.4;
            testCase.stats.val(2).prob = 0.55;
            testCase.stats.val(2).error = 0.5;
            testCase.opts.testMode = false;
            plotDAG(testCase.stats, epoch, testCase.opts);
         end
    end
end