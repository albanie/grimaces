classdef TestStateIO < matlab.unittest.TestCase
    
    methods (Test)
        
        function testLoadState(testCase)                  
            % set path to sample data
            fileName = GetFullPath('../data/net-epoch-sample.mat');
            [dagnet, stats] = loadState(fileName);
            
            % check that dagnet is a DagNN object
            testCase.verifyTrue(isa(dagnet, 'dagnn.DagNN'));
            
            % check that stats is a struct with correct fields
            testCase.verifyTrue(isa(stats, 'struct'));
            testCase.verifyTrue(isequal(fieldnames(stats), {'train';'val'}));
        end
        
        function testSaveState(testCase)                  
            % set path to sample data
            fileName = GetFullPath('../data/net-epoch-sample.mat');
            [dagnet, stats] = loadState(fileName);
            
            % set path to where network and stats will be stored
            targetFileName = '../data/net-epoch-target.mat';
            
            saveState(targetFileName, dagnet, stats);
            
            % check that net is saved as a vanilla matlab struct, 
            % rather than a dagnet object
            s = load(targetFileName, 'net', 'stats');
            testCase.verifyTrue(isa(s.net, 'struct'));

            % check that net is a DagNN object
            testCase.verifyTrue(isa(dagnet, 'dagnn.DagNN'));
            
            % check that stats remains unchanged
            testCase.verifyTrue(isequal(stats, s.stats));
        end
    end
end