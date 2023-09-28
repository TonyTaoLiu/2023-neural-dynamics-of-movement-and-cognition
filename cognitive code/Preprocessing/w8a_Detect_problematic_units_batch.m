%% w8a_Detect_problematic_units
% This script is an algorithm to detect units that should be excluded from the population analyses and units to which we should not apply any drift
% correction. This script allows implementing the policy I have established to deal with
% the drifting problem. For an explanation of the drift problem see w8a_drift_correction.m. The policy is the following:
% 1) Eliminate neurons that have a large drift, as defined by a 25% change in average firing rate between the first half and the second half of the session.
% 2) Exclude sessions that have more than 15% of their neurons exhibiting a large drift, as defined in step 1.
% 3) Exclude neurons that have a floor from step 4). A floor being defined as a neural response that goes to 0 Hz for at least 5 msec.
% 4) Drift correct all other neurons by subtracting momentaneous baseline FR from spontaneous activity.
% This script should be read before any further analysis. It outputs a .mat file which contains 2 structures:
% 1. units_to_discard : units that either have a strong drift that should be excluded from analysis, despite drift correction.
% 2. omit_correction : units that have a non-biological selectivity due to a sub-optimal drift correction (i.e. floor problem).
% This .mat file will be used by further scripts and will be located within the session folder of each respective sessions.
% Created by C. Testard 12/2017 Revised by SDT 07/2019

%% Set Batch Implementation
%Select the monkey
disp(' ');
Monkey = input('Which monkey to analyze? Krieger (1) or Brad (2) or Luca (3) ? : ');
disp(' ');

%Select experiment directory
disp(' '); disp('Please select the experiment directory')
PathName = uigetdir('', 'Please select the experiment directory');
cd(PathName)
if Monkey == 1
    neural_dir = dir('K*');
elseif Monkey == 2
    neural_dir = dir('B*');
elseif Monkey == 3
    neural_dir = dir('L*');
end

Bad_sessions = {}; %Initialize Bad_Sessions array to log in name of sessions with too many drifting neurons.

for Session = 1:length(neural_dir) %for all sessions for this monkey
    
    session_name = neural_dir(Session).name;
    flag_eyedata = 0; %Possibility to load and check eye data.
    
    %% Load and organize neural data
    [SpikeData, Timestamp_words, Words, BHV, head_position, words_per_trials, block, Waves, task] = w8a_Load_Organize_Data_batch(PathName, session_name, flag_eyedata);
    
    tic
    
    %% Create SDF per target to detect floor problems
    
    %Identify hit trials
    hits = find(BHV.TrialError == 0);
    %Identify the trial numbers part of each block
    Trials_block1 = find(BHV.BlockNumber == block(1));
    Trials_block2 = find(BHV.BlockNumber == block(2));
    %Identify the hit trials in each block
    hit_block1 = hits(ismember(hits, Trials_block1));
    hit_block2 = hits(ismember(hits, Trials_block2));
    %Pool the hits from both blocks (Corrupts chronology of trials. Shouldn't matter for SDF)
    hits = [hit_block1; hit_block2];
    
    %Get the timestamp of cue and target onset in the neural data for each hit trial (Not in chronological order)
    for i = 1:length(hits)
        Cue_onset_neural(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 12);
        Targets_onset(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 25);
        Response(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 40);
    end
    
    Cue_onset_neural = Cue_onset_neural';
    Targets_onset = Targets_onset';
    Response = Response';
    
    %% Create matrix for rasters aligned to three events
    %Define your analysis periods (in seconds)
    Analysis_period_lenght = 5000; %msec
    Cue_window_length = 3000; %msec
    Target_window_length = 750;
    Response_window_length = 1250;
    
    %Define the timestamps limits for the analysis period for each trial
    Analysis_period_cue = [(Cue_onset_neural - 1.5) (Cue_onset_neural + 1.5)];
    Analysis_period_target = [(Targets_onset - .5) (Targets_onset + .25)];
    Analysis_period_response = [(Response - .25) (Response + 1)];
    
    %Create structure with rasters per trials for all units
    hWaitbar = waitbar(0, 'Creating Raster Matrix');
    unit=1;
    for i = 1:length(fields(SpikeData)) %For all channels
        waitbar(i/length(fields(SpikeData)), hWaitbar)
        
        for j = 2:length(SpikeData.(['Channel_' num2str(i)])) %For all units
            
            for k = 1:length(hits) %For all hit trials
                
                Unit_rasters{unit}(k,:) = zeros(1,Analysis_period_lenght); %Fill the line with zeros to initiate raster for that trial
                temp_cue = find(SpikeData.(['Channel_' num2str(i)]){j} > Analysis_period_cue(k,1) & SpikeData.(['Channel_' num2str(i)]){j} < Analysis_period_cue(k,2)); %Find indicies of spikes during this analysis period
                tick_cue = SpikeData.(['Channel_' num2str(i)]){j}(temp_cue) - Analysis_period_cue(k,1); %Align those spikes to cue onset -1500ms for that trial
                tick_cue = ceil(tick_cue*1000); %Convert spikes timings (in raster time) to miliseconds
                Unit_rasters{unit}(k, tick_cue) = 1; %Fill in spikes in the raster
                temp_target = find(SpikeData.(['Channel_' num2str(i)]){j} > Analysis_period_target(k,1) & SpikeData.(['Channel_' num2str(i)]){j} < Analysis_period_target(k,2)); %Find indicies of spikes during this analysis period
                tick_target = SpikeData.(['Channel_' num2str(i)]){j}(temp_target) - Analysis_period_target(k,1); %Align those spikes to target - 500ms for that trial
                tick_target = ceil(tick_target*1000) + Cue_window_length; %Convert spikes timings (in raster time) to miliseconds, add the cue analysis period length so as to concatenate both matrices.
                Unit_rasters{unit}(k, tick_target) = 1; %Fill in spikes in the raster
                temp_response = find(SpikeData.(['Channel_' num2str(i)]){j} > Analysis_period_response(k,1) & SpikeData.(['Channel_' num2str(i)]){j} < Analysis_period_response(k,2)); %Find indicies of spikes during this analysis period
                tick_response = SpikeData.(['Channel_' num2str(i)]){j}(temp_response) - Analysis_period_response(k,1); %Align those spikes to response - 250ms for that trial
                tick_response = ceil(tick_response*1000) + Cue_window_length + Target_window_length; %Convert spikes timings (in raster time) to miliseconds, add the cue analysis period length so as to concatenate both matrices.
                Unit_rasters{unit}(k, tick_response) = 1; %Fill in spikes in the raster
                clear tick_cue temp_cue tick_target temp_target temp_response tick_response
            end
            Electrode_ID(unit) = i;
            unit = unit+1;
        end
        
    end
    
    close(hWaitbar)
    
    %% Compute SDF
    %Define kernel
    sigma = .045; %Define width of kernel (in sec)
    edges = -3*sigma:.001:3*sigma;
    kernel = normpdf(edges,0,sigma); %Use a gaussian function
    kernel = kernel*.001; %Time 1/1000 so the total area under the gaussian is 1
    
    % Compute Spike Density Function for all hit trials
    hWaitbar = waitbar(0, 'Convolving rasters');
    for i = 1:length(Unit_rasters) % for all units
        waitbar(i/length(Unit_rasters), hWaitbar)
        for j = 1:size(Unit_rasters{1,i},1) % for all hit trials
            
            sdf = conv(Unit_rasters{1,i}(j,:),kernel); %Convolve the gaussian
            center = ceil(length(edges)/2);
            SDF{1,i}(j,:) = sdf(center:length(sdf)-(center-1)).*1000; %Substract the irrelevant edges due to convolution operation
            clear sdf
            
        end
    end
    close(hWaitbar)
    clear Unit_rasters
    
    
    %Divide hit trials by their cue
    star = strfind(BHV.TimingFileByCond, 'visual_star.m'); star_conditions = find(not(cellfun('isempty', star)));
    circle = strfind(BHV.TimingFileByCond, 'visual_circle.m'); circle_conditions = find(not(cellfun('isempty', circle)));
    star_trials = find(ismember(BHV.ConditionNumber, star_conditions)); %Indices of star trials among all trials (index equivalent to the trial number)
    circle_trials = find(ismember(BHV.ConditionNumber, circle_conditions)); %Indices of star trials among all trials (index equivalent to the trial number)
    
    cue_hit_trials{1} = find(ismember(hits, star_trials) & ismember(hits, hit_block1)); %Indices of hits vector that are both star trials and block1 trials. (index NOT equivalent to the trial number)
    cue_hit_trials{2} = find(ismember(hits, circle_trials) & ismember(hits, hit_block1));
    cue_hit_trials{3} = find(ismember(hits, star_trials) & ismember(hits, hit_block2));
    cue_hit_trials{4} = find(ismember(hits, circle_trials) & ismember(hits, hit_block2));
    
    % Compute Mean SDF for correct trials of each cue condition
    for j = 1:4 %For each cue condition
        for i = 1:length(SDF) %For all units
            Mean_SDF{j,i} = mean(SDF{1,i}(cue_hit_trials{j},:));
            Error_SDF{j,i} = std(SDF{1,i}(cue_hit_trials{j},:))/((size(SDF{1,i}(cue_hit_trials{j},:),1))^(1/2));
        end
    end

    
    %Detect units with floor problems 
    flag_floor = [];
    for unit = 1:size(Mean_SDF,2) %for each unit
        for condition = 1:size(Mean_SDF,1) %for each task condition
            if length(find(Mean_SDF{condition,unit} == 0)) > 5 %if there is a floor at 0Hz of at least 5 msec for any of the task conditions.
                flag_floor(unit) = 1; %flag this unit as having a floor
            end
        end
    end
    
    Floor_neurons = find(flag_floor);
    
    
    %% Detect units that have a massive drift, either positive or negative
    %Estimate the trends in firing rates of units over time
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
                else
                    Average_Spiking(unit,k) = length(temp)/(bin_size); % Calculate firing rate for each bin;
                end
                temp = [];
            end
            unit = unit+1;
            
        end
        
    end
    
    
    %Attempt based on differences in mean between halfs of the session
    Num_bins = length(Average_Spiking(1,:));
    Half = floor(Num_bins/2); %Number of drift bins in one half of a session
    for unit = 1:size(Average_Spiking,1)
        First_half = mean(Average_Spiking(unit,1:Half));
        Second_half = mean(Average_Spiking(unit,Half:Num_bins));
        
        if (First_half - Second_half)/First_half > .25 || (First_half - Second_half)/First_half < -.25 % A 25% change in firing rate is the threshold for the definition of large drift.
            flag_large_drift(unit) = 1;
        else
            flag_large_drift(unit) = 0;
        end
        
    end
    
    Num_flagged(Session,1) = length(find(flag_large_drift)); %Absolute number of neurons flagged
    Num_flagged(Session,2) = length(find(flag_large_drift)) / size(Average_Spiking,1); %Proportion of neurons flagged in session
    Big_drift_neurons = find(flag_large_drift);
 
    if Num_flagged(Session,2) > .15 %If this session has more than 15% of large drift neuron, it should be flagged and eliminated from further analyses.
        Bad_sessions = [Bad_sessions;session_name];
    end
    
    
    cd([PathName '/' session_name]) %Save the two arrays in the session folder for later consultation by scripts.
    save(['ProblematicUnits_' session_name], 'Big_drift_neurons', 'Floor_neurons')
    
    clearvars -except Bad_sessions Session PathName neural_dir Num_flagged
    
end

%Output warning message for sessions that have too many neurons with large drift. These should be excluded from analyses.
if ~isempty(Bad_sessions)
    for i = 1:size(Bad_sessions)
    warning(['Session ' Bad_sessions{i} ' have a large number of drifting neurons (>15%) and should be excluded from neural analyses'])
    end
else
    disp('No session should be excluded from analyses based on neural drift')
end
    
    
    
    
    