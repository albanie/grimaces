function bestEpoch = findBestEpoch(expDir)
% returns the epoch number in which the network
% achieved the highest average precision on the 
% validation set.

lastEpoch = findLastCheckpoint(expDir);
data = load(fullfile(expDir, sprintf('net-epoch-%d.mat', lastEpoch)));
AP = [data.stats.val.averagePrecision];
[~, bestEpoch] = max(AP);
end