


dlFeatures = predict(obj.FeatureExtractionNet, dlX, 'Acceleration','auto');

[dlRPNScores, dlRPNReg] = predict(obj.RegionProposalNet, dlFeatures, 'Outputs',{'RPNClassOut', 'RPNRegOut'});

dlProposals = sequentialRegionProposal(obj, dlRPNReg, dlRPNScores, KnownBBoxes, numAdditionalProposals); %change to sequential RP using previous BBOXES

dlPooled = roiAlignPooling(obj, dlFeatures, dlProposals, obj.PoolSize);

dlFinalFeatures = predict(obj.PostPoolFeatureExtractionNet, dlPooled, dlPooled, 'Acceleration','auto');

[dlBoxReg, dlBoxScores] = predict(obj.DetectionHeads, dlFinalFeatures, 'Acceleration','auto', 'Outputs',{'detectorRegOut', 'detectorClassOut'});

outputFeatures{1} = dlProposals;
outputFeatures{2}= dlBoxReg;
outputFeatures{3}= dlBoxScores;
outputFeatures{4}= dlFeatures;