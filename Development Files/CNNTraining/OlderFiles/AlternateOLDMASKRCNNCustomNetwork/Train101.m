clear
clc
close all

addpath('./src');
load("Net101.mat");

%%
trainCats={'CellA'};
classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];




Datdir="../";

ds = fileDatastore([Datdir+"DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data
ds = fileDatastore("../SmallDSFs/", ReadFcn=@(x)cocoAnnotationMATReader(x)); %training data

%valds=fileDatastore([Datdir+"ValDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %validation data
trainClassNames = ["CellA"];
imageSize=[520 704 3];
numClasses = numel(classNames);
% Add a background class
classNames = [trainClassNames {'background'}];


params = createMaskRCNNConfig(imageSize, numClasses, classNames);



if canUseGPU
    executionEnvironment = "gpu";
else
    executionEnvironment = "cpu";
end

initialLearnRate = 0.0005;
momemtum = 0.9;
decay = 0.0001;
velocity = [];
maxEpochs = 2;
minibatchSize = 1;


myMiniBatchFcn = @(img, boxes, labels, masks) deal(cat(4, img{:}), boxes, labels, masks);
mb = minibatchqueue(ds, 4, "MiniBatchFormat", ["SSCB", "", "", ""],...
                            "MiniBatchSize", minibatchSize,...
                            "OutputCast", ["single","","",""],...
                            "OutputAsDlArray", [true, false, false, false],...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu","cpu"]);





%%

numEpoch = 1;
numIteration = 1; 


doTraining=1;

start = tic;
if doTraining
    
     % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter] = helper.configureTrainingProgressPlotter(fig);
    
    % Initialize verbose output
    helper.initializeVerboseOutput([]);
    
    % Custom training loop.
    while numEpoch < maxEpochs
    mb.reset();
    mb.shuffle();
    
        while mb.hasdata()
            % get next batch from minibatchqueue
            [X, gtBox, gtClass, gtMask] = mb.next();
        
            % Evaluate the model gradients and loss using dlfeval
            [gradients, loss, state] = dlfeval(@networkGradients, X, gtBox, gtClass, gtMask, dlnet, params);
            dlnet.State = state;
            
            % compute the learning rate for current iteration
            learnRate = initialLearnRate/(1 + decay*numIteration);
            
            if(~isempty(gradients) && ~isempty(loss))
    
                [dlnet.Learnables, velocity] = sgdmupdate(dlnet.Learnables, gradients, velocity, learnRate, momemtum);
            else
                continue;
            end
            helper.displayVerboseOutputEveryEpoch(start,learnRate,numEpoch,numIteration,loss);
                
            % Plot loss/ accuracy metric
             D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lossPlotter,numIteration,double(gather(extractdata(loss))))
            
            subplot(2,1,2)

            title("Epoch: " + numEpoch + ", Elapsed: " + string(D))
           
            drawnow
            
            numIteration = numIteration + 1;
    
        end
    numEpoch = numEpoch + 1;
    
    end
end