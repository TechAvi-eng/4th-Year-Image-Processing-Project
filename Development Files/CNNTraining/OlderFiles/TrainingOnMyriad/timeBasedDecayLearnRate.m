classdef timeBasedDecayLearnRate < deep.LearnRateSchedule
    % timeBasedDecayLearnRate Time-based decay learning rate schedule

    properties
        % Schedule properties
    
        Decay
    end

    methods
        function schedule = timeBasedDecayLearnRate(decay) 
            % timeBasedDecayLearnRate Time-based decay learning rate
            % schedule
            %   schedule = timeBasedDecayLearnRate(decay) creates a
            %   time-based decay learning rate schedule with the specified
            %   decay.
    
    
            % Set schedule properties.
            schedule.FrequencyUnit = "iteration";
            schedule.NumSteps = Inf;
            schedule.Decay = decay;
        end

        function [schedule,learnRate] = update(schedule,initialLearnRate,iteration,~)
            % UPDATE Update learning rate schedule
            %   [schedule,learnRate] = update(schedule,initialLearnRate,iteration,~)
            %   calculates the learning rate for the specified iteration
            %   and also returns the updated schedule object.

            % Calculate learning rate.
            decay = schedule.Decay;
            learnRate = initialLearnRate / (1 + decay*(iteration-1));
        end
    end
end