classdef TestInitMomentum < matlab.unittest.TestCase
    
    methods (Test)
        
        function testInitMomentumOnCPU(testCase)                  
            % check that momentum is correctly initialized on 
            % the CPU.
            state = struct();
            opts.gpus = [];
            mode = 'train';
            
            % mock dagnet object, with 20 param layers
            dagnet = mockDagnet();
            dagnet.params = 1:20;
            
            updatedState = initMomentum(state, dagnet, opts, mode);
            expectedMomentum = num2cell(zeros(1, 20));
            
            % check momentum of updatedState
            testCase.verifyEqual(updatedState.momentum, expectedMomentum);
        end        
    end
end
