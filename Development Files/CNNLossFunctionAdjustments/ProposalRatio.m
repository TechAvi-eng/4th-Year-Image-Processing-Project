function loss = NegativeMining(YRPNClass, RPNClassificationTargets, Ratio)

%pre-allocate loss array for each minibatch
lossBatch = zeros(size(YRPNClass, 4), 1);


%repeat for each image in the minibatch
for i=[1:size(YRPNClass, 4)]
    %change array to 1d values
    Class = YRPNClass(:,:,:,i);
    Class = Class(:);

    Targets = RPNClassificationTargets(:,:,:,i);
    Targets = Targets(:);

    %sort from most confident prediction to least
    [Class idx] = sort(Class);
    Targets = Targets(idx);


    posIndex = Targets == 1;
    negIndex = Targets == 0;
    %find number of negative proposals to keep, which is given as a ratio
    %of the number of positive examples (up to the entire array), one is
    %added to ensure value cannot be zero
    numPos = sum(posIndex);
    numNeg = min((numPos+1)*Ratio, sum(negIndex) );

    posProps = Class(posIndex);

    negProps = Class(negIndex);
    negProps = Class(1:numNeg); %keep only top values, which are the most incorrect ones that are best to train one

    ClassY = extractdata([posProps; negProps]);
    TargetsY = [ ones(numPos,1); zeros(numNeg,1) ];

    %use CrossEntropy or Focal Loss to calculate final loss figure
    %lossBatch(i) = vision.internal.cnn.maskrcnn.CrossEntropy(ClassY, TargetsY);
    %lossBatch(i) = focalCrossEntropy(ClassY, TargetsY, "DataFormat", "S");
    
    %scale loss by the number of positive samples to make it comparable between examples; multiply by 10 to ensure it remains relevant in the total
    lossBatch(i) = crossentropy(ClassY, TargetsY) * length(TargetsY)./length(Targets) *10;
end


loss = sum(lossBatch);   
end
