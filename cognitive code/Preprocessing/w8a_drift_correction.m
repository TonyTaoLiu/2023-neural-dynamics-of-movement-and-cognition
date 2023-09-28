function [baseline_activity,baseline_spike_count] = w8a_drift_correction(SpikeData, hits, words_per_trials)
%% w8a_drift_correction
% This script applies a firing rate drift correction to the firing rate of single units. It outputs a baseline activity matrix of rates with structure units x (hits)trials.
% This structute is not flexible. It can be used to correct any matrix of firing rates that has that structure. Only rates can be corrected, not spike counts.
% This function was created to correct an observed drift in average firing rate of neurons over the session that might
% be explained by changing electrode isolation or unknown biological effects. When this drift correlates with an
% independent variable that also changes over time, as in a block design for example, the decoding accuracy of a decoder
% could be inflated by the artificial difference in rate between the blocks created by the drift. Conversely, when a
% independent variable is not correlated with the drift, the drift can increase the standard deviation of the firing
% rates over the session and decrease decoding accuracy.
% Created by Testard & Tremblay 09/2017

%% Estimate the trends in firing rates of units over time
bin_size = 240; % size of a bin is 4 minutes, in seconds
step = 60; % a bin is calculated every 60 seconds, looking 2 min back, and two forward
session_length = ceil(words_per_trials(end).timestamps(end));% in sec; find the last word to get a sense of the length of the session
time_points = step:step:session_length-step; % Set the center time points at which a bin needs to be calculated. The number of time points won't match the number of 60 second steps in your session

session_time_bins(1, 1) = time_points(1)-step; % Make the first bin half the size to capture the first minute
session_time_bins(1, 2) = time_points(1)+step;
for i = 2:length(time_points)-1 % Find the bin time limits; over which to calculate firing rate.
    session_time_bins(i, 1) = time_points(i) - bin_size/2;
    session_time_bins(i, 2) = time_points(i) + bin_size/2;
end
session_time_bins(length(time_points), 1) = time_points(length(time_points))-step;% Make the last bin half the size to capture the last minute
session_time_bins(length(time_points), 2) = time_points(length(time_points))+step;

unit=1; %Initialize the unit #
for i = 1:length(fields(SpikeData)) %For all channels
    
    for j = 2:length(SpikeData.(['Channel_' num2str(i)])) %For all units
        
        for k = 1:length(session_time_bins) % for all time points
            temp = find(SpikeData.(['Channel_' num2str(i)]){j} > session_time_bins(k, 1) & SpikeData.(['Channel_' num2str(i)]){j} < session_time_bins(k, 2)); % find spike times in each bin
            if k == 1 || k == length(session_time_bins) % if its the first or last minute, the bin_size is half.
                Average_Spiking(unit,k) = length(temp)/(bin_size/2); % Calculate firing rate for each bin;
                Average_Spike_Count(unit,k) = length(temp);
            else
                Average_Spiking(unit,k) = length(temp)/(bin_size); % Calculate firing rate for each bin;
                Average_Spike_Count(unit,k) = length(temp);
            end
            temp = [];
        end
        unit = unit+1;
        
    end
    
end

% figure
% plot(Average_Spiking')

%% Half-session comparison plots
% Plots half session one against the other to find major trends across neurons

% session_length = ceil(SpikeData.Channel_1{1, 2}(end));%in sec; find the last spike timestamp
% half = session_length/2;
% bins = [0 half; half session_length];
% 
% unit=1; %Initialize the unit #
% for i = 1:length(fields(SpikeData)) %For all channels
% 
%     for j = 2:length(SpikeData.(['Channel_' num2str(i)])) %For all units
% 
%         for k = 1:2 % for each half
%             temp = find(SpikeData.(['Channel_' num2str(i)]){j} > bins(k, 1) & SpikeData.(['Channel_' num2str(i)]){j} < bins(k, 2)); % find spike times in each bin
%             Spikes(unit,k) = length(temp)/half; % Calculate firing rate for each bin;
%             temp = [];
%         end
%         unit = unit+1;
% 
%     end
% 
% end
% 
% figure
% plot(Spikes(:,:)')
% xlim([0 3])
% figure
% histogram(Spikes(:,2) - Spikes(:,1), 30) %Consider making these differences relative to mean firing rate (COV)
% xlabel('Firing rate difference between 1st and 2nd half of the session')
%
%% Assign baseline firing rate for each unit, for each trial

%Find trial start times for all hit trials
for i = 1:length(hits)
    trial_start(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 9);
end

%Assign each hit trial to a time bin over the session
for i = 1:length(hits) % for all hit trials
    trial_session_epoch(i) = round(trial_start(i)/60); % Find in which minute of the session this trial occured.
    if trial_session_epoch(i) == 0 % If it was before the first 30 seconds, assign it to the first minute
        trial_session_epoch(i) = 1;
    elseif trial_session_epoch(i) > size(Average_Spiking,2) % If the minute of this trial is later than the latest bin calculated
        trial_session_epoch(i) = size(Average_Spiking,2); %Then assign this trial to the latest bin calculated. In other words, the time resolution for the last minutes of the session is slightly reduced for the correction.
    end
end

%Assign average firing rate to each unit, for each trial
for unit = 1:size(Average_Spiking,1) % For each unit
    for trial = 1:length(hits) % For each hit trial
        baseline_activity(unit, trial) = Average_Spiking(unit, trial_session_epoch(trial)); % This matrix has a structure Unit x Trial, each entry has the baseline firing rate of the unit i at trial j
        baseline_spike_count(unit,trial) = Average_Spike_Count(unit, trial_session_epoch(trial));
    end
end

% figure
% plot(baseline_activity')

% %% Apply the correction to the input matrix (trial x unit) (We preferred to keep this outside the function.
% 
% Drift_corrected_firing_rates_matrix = Input_firing_rate_matrix - baseline_activity';

% End of script
end
