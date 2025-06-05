clear
clc
close all


addpath('./src');



%trainClassNames = ["CellA"];
imageSize=[520 704 3];

trainCats = {'CellA'};


classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


ds = fileDatastore("../SmallDSFs/", ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data
trainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));

data = preview(trainDS) 

params = createMaskRCNNConfig(imageSize, numClasses, classNames);
disp(params);

dlnet = createMaskRCNN(numClasses, params, 'resnet50'); %or resnet 101

%set environment
if canUseGPU
    executionEnvironment = "gpu";
    numberOfGPUs = gpuDeviceCount;
    pool = parpool(numberOfGPUs);
else
    executionEnvironment = "cpu";
    pool = parpool("Processes",1);
end

    N = pool.NumWorkers;

 % Create a DataQueue

stream = parallel.pool.DataQueue;
stream.afterEach(@(x)helper.displayTrainingProgress(x,lossPlotter));


%gradient function for each parallel worker
reduceGradientsFcn = @(x, factor) gplus(factor * extractdata(x), class(extractdata(x)));
aggregateStateFcn = @(state, factor) gplus(factor*state);

%% learning parameters
initialLearnRate = 0.0001;
momemtum = 0.9;
decay = 0.0001;
velocity = [];
maxEpochs = 30;

miniBatchSize = 1


%distribute number equal to mini batch size in each gpu

if executionEnvironment == "gpu"
    miniBatchSize = miniBatchSize .* N
end
workerMiniBatchSize = floor(miniBatchSize ./ repmat(N,1,N));
remainder = miniBatchSize - sum(workerMiniBatchSize);
workerMiniBatchSize = workerMiniBatchSize + [ones(1,remainder) zeros(1,N-remainder)]



% Create the batching function. The images are concatenated along the 4th
% dimension to get a HxWxCxminiBatchSize shaped batch. The other ground truth data is
% configured a cell array of length = minibatchSize.
myMiniBatchFcn = @(img, boxes, labels, masks) deal(cat(4, img{:}), boxes, labels, masks);


mb = minibatchqueue(trainDS, 4, "MiniBatchFormat", ["SSCB", "", "", ""],...
                            "MiniBatchSize", miniBatchSize,...
                            "OutputCast", ["single","","",""],...
                            "OutputAsDlArray", [true, false, false, false],...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu","cpu"]);



%% start training loop



numEpoch = 1;
numIteration = 1;

start = tic;
doTraining=1; %turn off training in script for test purposes

if doTraining
    spmd
        % Split data
        localDs = partition(trainDS, N, labindex);
        localVelocity = velocity;
        
        iteration = 0;
        
        % Construct minibatchqueue from the partitioned datastore
        mb = minibatchqueue(localDs, 4, "MiniBatchFormat", ["SSCB", "", "", ""],...
                            "MiniBatchSize", workerMiniBatchSize(labindex),...
                            "OutputCast", ["single","","",""],...
                            "OutputAsDlArray", [true, false, false, false],...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu","cpu"]);
        

                        
        for epoch = 1:numEpochs
            % reset and shuffle minibatchqueue
            shuffle(mb);
            
            % Loop over mini-batches
            while gop(@and, hasdata(mb))
                
                iteration = iteration+1;
                % Get next mini-batch
                [X, gtBox, gtClass, gtMask] = next(mb);
                
                % Evaluate the model gradients and loss of the worker using dlfeval and the
                % modelGradients function.
                [localGradients, localLoss, localState] = dlfeval(@networkGradients, X, gtBox, gtClass, gtMask, dlnet, params);
                
                % Perform reduction on the loss
                localNormalizationFactor = (workerMiniBatchSize(labindex) ./ miniBatchSize);
                localLoss = localNormalizationFactor * extractdata(localLoss);
                loss = gplus(localLoss);
                
                % Aggregate the network state on all workers
                localState.Value = dlupdate(aggregateStateFcn,localState.Value,{localNormalizationFactor});
                dlnet.State = localState;
                
                % Perform reduction on the gradients
                localGradients.Value = dlupdate(reduceGradientsFcn, localGradients.Value, {localNormalizationFactor});
                
                learnRate = initialLearnRate/(1 + decay*iteration);

                % Update the network parameters using the SGDM optimizer.
                [dlnet.Learnables, localVelocity] = sgdmupdate(dlnet.Learnables, localGradients, localVelocity, learnRate, momentum);
                
                if labindex == 1
                    output = [epoch iteration loss [] learnRate];
                    stream.send(gather(output));
                end
            
            end

        end
    end
    net = dlnet{1};
    save('dlnet.mat','net');    
   
end