function loss = NegativeMining(YRPNClass, RPNClassificationTargets, Ratio)

    % Reshape inputs: (N, B) where N = H*W*C, B = batch size
    N = size(YRPNClass,1) * size(YRPNClass,2) * size(YRPNClass,3);
    B = size(YRPNClass, 4);
    
    Class = extractdata(reshape(YRPNClass, N, B));
    Targets = reshape(RPNClassificationTargets, N, B);
    
    % Sort each batch individually (descending: most confident → least confident)
    [Class, idx] = sort(Class, 1, 'descend'); 
    Targets = Targets(idx + (0:(B-1))*N); % Reorder targets accordingly

    % Compute positive and negative indices per batch
    posIndex = (Targets == 1);
    negIndex = (Targets == 0);

    % Compute the number of negative samples to keep per batch
    numPos = sum(posIndex, 1); % Number of positive samples per batch
    numNeg = min((numPos + 1) .* Ratio, sum(negIndex, 1)); % Limit negatives

    % Extract positive proposals
    posProps = zeros(N, B);
    posProps(posIndex) = Class(posIndex); % Keep only positive class values

    % Extract negative proposals (hardest negatives)
    negProps = zeros(N, B);
    
    % Create masks to keep only the hardest negative samples per batch
    negMask = cumsum(negIndex, 1) <= numNeg; % Keeps top `numNeg` values
    negProps(negIndex & negMask) = Class(negIndex & negMask);

    % Flatten each batch separately and remove zeros
    posProps = posProps(posProps ~= 0);
    negProps = negProps(negProps ~= 0);

    %Label smoothing parameter epsilon, to improve performance by making
    %predictions slightly less confident
    eps = 0.05;

    % Concatenate final selection per batch
    ClassY = [posProps; negProps];
    TargetsY = [ones(sum(numPos), 1)*(1-eps); ones(sum(numNeg), 1)*eps];

    % Compute cross-entropy loss batch-wise, normalize for number of
    % examples used

    loss = -mean(TargetsY.*log(ClassY) + (1-TargetsY).*log(1- ClassY));
    
end
