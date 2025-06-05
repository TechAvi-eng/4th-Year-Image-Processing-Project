clear
clc
close all


addpath('./src');

%%


%trainClassNames = ["CellA"];
imageSize=[520 704 1];






 datdir="..";
 ds = fileDatastore([datdir+"/CatDSFs"], ReadFcn=@(x)MATReader1C(x)); %training datatrainDS = transform(ds, @(x)helper.preprocessData(x, imageSize));


trainDS = ds.shuffle;


data = preview(trainDS) 






%dlnet = createMaskRCNN(numClasses, params, 'resnet50'); %or resnet 101
%load("NET101-4D.mat")
load("1ChannelNet.mat")

trainCats= {'SHSY5Y', 'SKOV3', 'SkBr3', 'Huh7', 'BV2', 'MCF7', 'A172', 'BT474'}
classNames = trainCats;
numClasses = numel(classNames);
% Add a background class
classNames = [classNames {'background'}];


%set environment
if canUseGPU
    executionEnvironment = "gpu";
    gpuDevice(1)
else
    executionEnvironment = "cpu";
end

params.ImageSize=[520 704 1];
%disp(params);

%% learning parameters
initialLearnRate = 0.001
momemtum = 0.9
decay = 0.0025
velocity = []
maxEpochs = 30

minibatchSize = 1


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


savefreq= 400; %iterations, move location in for loop to save every X epoch
disp(params)

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
                    savenet=gather(dlnet);
                    save("~/Scratch/TrainV6/NetData101/Checkpoint-"+modelDateTime+".mat", "savenet"); %save output with the date and time into the current directotry
            end
    

        end
    numEpoch = numEpoch + 1;
    
    end
end

dlnet=gather(dlnet);
modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
                    save("~/Scratch/TrainV6/NetData101/FINAL"+modelDateTime+".mat"); %save output with the date and time into the current directotry
