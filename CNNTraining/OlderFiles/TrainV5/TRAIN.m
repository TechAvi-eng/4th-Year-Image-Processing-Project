clear
clc
close all


addpath('./src');

%%


%trainClassNames = ["CellA"];
imageSize=[520 704 3];

trainCats = {'CellA'};


classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


% datdir="/home/zcemydo/Scratch/TrainV2/";
% ds = fileDatastore([datdir+"/DSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));

datdir="..";
ds = fileDatastore([datdir+"/SmallDSFs"], ReadFcn=@(x)cocoAnnotationMATReader(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));

trainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));


data = preview(trainDS) 

params = createMaskRCNNConfig(imageSize, numClasses, classNames);


    params.NumRegionsToSample = 400;
    params.NumStrongestRegions = 1000;

    params.AnchorBoxes=[
    32    16
    64    32
   8    4
   20   40
   30   15
    32    32
    64    64
   20   20
   40   40
   60   60
    16    32
    32    64
    64   128
   15   30
   20   40]




disp(params);


dlnet = createMaskRCNN(numClasses, params, 'resnet101'); %or resnet 101

%set environment
if canUseGPU
    executionEnvironment = "gpu";
else
    executionEnvironment = "cpu";
end

%% learning parameters
initialLearnRate = 0.0001;
momemtum = 0.9;
decay = 0.0001;
velocity = [];
maxEpochs = 30;

minibatchSize = 8


% Create the batching function. The images are concatenated along the 4th
% dimension to get a HxWxCxminiBatchSize shaped batch. The other ground truth data is
% configured a cell array of length = minibatchSize.
myMiniBatchFcn = @(img, boxes, labels, masks) deal(cat(4, img{:}), boxes, labels, masks);

mb = minibatchqueue(trainDS, 4, "MiniBatchFormat", ["SSCB", "", "", ""],...
                            "MiniBatchSize", minibatchSize,...
                            "OutputCast", ["single","","",""],...
                            "OutputAsDlArray", [true, false, false, false],...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu","cpu"]);


savefreq= 5; %iterations, move location in for loop to save every X epoch


%% start training loop

numEpoch = 1;
numIteration = 1;

start = tic;
doTraining=1; %turn off training in script for test purposes

if doTraining
    
     % Create subplots for the learning rate and mini-batch loss.
    
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
            
            numIteration = numIteration + 1;
            if rem(numIteration, savefreq) == 0
                    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
                    save("~/Scratch/TrainV5/NetData/Checkpoint-"+modelDateTime+".mat"); %save output with the date and time into the current directotry
            end
    

        end
    numEpoch = numEpoch + 1;
    
    end
end

