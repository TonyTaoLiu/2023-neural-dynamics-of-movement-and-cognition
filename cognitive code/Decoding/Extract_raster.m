%Select the monkey
Monkey = input('Which monkey to analyze? Krieger (1) or Brad (2) or Luca (3) ? : ');
%Select experiment directory
% cd('/Users/SeB/Documents/University/Postdoc/Petrides/Projects/Wireless project/Sessions')
% cd('E:\MSc Project\monkey dataset\cognitive')
disp(' '); disp('Please select the experiment directory')
PathName = uigetdir('', 'Please select the experiment directory');
cd(PathName)
if Monkey == 1
    neural_dir = dir('K*');
    Monk = 'Krieger';
elseif Monkey == 2
    neural_dir = dir('B*');
    Monk = 'Brad';
elseif Monkey == 3
    neural_dir = dir('L*');
    Monk = 'Luca';
end

%% Set Parameters for extracting windows of data
% Choose the variable of interest.
trial_type = input('For target press 1, for cue press 2, for location press 3, for target chosen press 4: ');
% Take out noise correlations?
shuffle_noise_out = 0; % input('For taking out noise correlations press 1, else press 0:  ');
% Compute chance performance by shuffling labels?
Chance = 0; %input('For computing chance performance press 1, else press 0:  ');

for Session = 1:length(neural_dir) %for all sessions for this monkey
    
    session_name = neural_dir(Session).name;
    flag_eyedata = 0; %Possibility to load and check eye data.
    
    %% Load data and organize trial events.
    [SpikeData, Timestamp_words, Words, BHV, head_position, words_per_trials, block, Waves, task] = w8a_Load_Organize_Data_batch(PathName, session_name, flag_eyedata);
    
    %Load problematic units structure
    load([PathName '/' session_name '/ProblematicUnits_' session_name '.mat'])
    
    %Identify hit trials
    hits = find(BHV.TrialError == 0);
    
    if task == 1 || task == 3 %If main task or interleaved blocks
        %Identify the trial numbers part of each block
        Trials_block1 = find(BHV.BlockNumber == block(1));
        Trials_block2 = find(BHV.BlockNumber == block(2));
        %Identify the hit trials in each block
        hit_block1 = hits(ismember(hits, Trials_block1));
        hit_block2 = hits(ismember(hits, Trials_block2));
        %Pool the hits from both blocks (Corrupts chronology of trials. Shouldn't matter for SDF)
        hits = [hit_block1; hit_block2];
    end
    
    %Get the timestamp of cue and target onset in the neural data for each hit trial (Not in chronological order)
    for i = 1:length(hits)
        Cue_onset_neural(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 12);
        Targets_onset(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 25);
        Response(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 40);
        if task == 6
            Delay_onset(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 16);
        end
    end
    Cue_onset_neural = Cue_onset_neural';
    Targets_onset = Targets_onset';
    Response = Response';
    if task == 6
        Delay_onset = Delay_onset';
    end
    
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
    
    %% Create labels vector as a function of condition of interest
    if trial_type == 1 % Target
        star = strfind(BHV.TimingFileByCond, 'visual_star.m'); star_conditions = find(not(cellfun('isempty', star)));
        circle = strfind(BHV.TimingFileByCond, 'visual_circle.m'); circle_conditions = find(not(cellfun('isempty', circle)));
        group_trials{1} = find(ismember(BHV.ConditionNumber(hits), star_conditions)); % indices of star trials among hit trials vector (hits)
        group_trials{2} = find(ismember(BHV.ConditionNumber(hits), circle_conditions));
        
    elseif trial_type == 2 % Cue
        star = strfind(BHV.TimingFileByCond, 'visual_star.m'); star_conditions = find(not(cellfun('isempty', star)));
        circle = strfind(BHV.TimingFileByCond, 'visual_circle.m'); circle_conditions = find(not(cellfun('isempty', circle)));
        star_trials = find(ismember(BHV.ConditionNumber, star_conditions)); %Find all star trials, hits or not
        circle_trials = find(ismember(BHV.ConditionNumber, circle_conditions)); %Find all circle trials, hits or not
        group_trials{1} = find(ismember(hit_block1, star_trials)); %Identify indices of hit trials that belong to block1 and to star trials (indices are the same for hits and hit_block1 vector
        group_trials{2} = find(ismember(hit_block1, circle_trials));
        group_trials{3} = find(ismember([zeros(length(hit_block1),1);hit_block2], star_trials)); %Identify hit trials that belong to block2 and to star trials (indices are NOT the same for hits and hit_block2 vector. Need to add an empty vector in place of hit_block1 to make the indices match)
        group_trials{4} = find(ismember([zeros(length(hit_block1),1);hit_block2], circle_trials));
        
    elseif trial_type == 3 % Location
        if task == 1 || task == 3
            %Identify indices of hit trials for each target position. (assumes a condition file with 16 conditions per block)
            group_trials = cell(1,8);
            for i = 1:length(block) %Positions start with 1 at midnight (top) and run clockwise.
                group_trials{5} = [group_trials{5}; find(BHV.ConditionNumber(hits) == block(i)*16-15); find(BHV.ConditionNumber(hits) == block(i)*16-4)]; %Target Low; Condition number 1 and 12 of each block
                group_trials{4} = [group_trials{4}; find(BHV.ConditionNumber(hits) == block(i)*16-14); find(BHV.ConditionNumber(hits) == block(i)*16-3)]; %Target Low-right; Condition number 2 and 13 of each block
                group_trials{2} = [group_trials{2}; find(BHV.ConditionNumber(hits) == block(i)*16-13); find(BHV.ConditionNumber(hits) == block(i)*16-2)]; %Target Up-right; Condition number 3 and 14 of each block
                group_trials{1} = [group_trials{1}; find(BHV.ConditionNumber(hits) == block(i)*16-12); find(BHV.ConditionNumber(hits) == block(i)*16-7)]; %Target Up; Condition number 4 and 9 of each block
                group_trials{8} = [group_trials{8}; find(BHV.ConditionNumber(hits) == block(i)*16-11); find(BHV.ConditionNumber(hits) == block(i)*16-6)]; %Target Up-left; Condition number 5 and 10 of each block
                group_trials{6} = [group_trials{6}; find(BHV.ConditionNumber(hits) == block(i)*16-10); find(BHV.ConditionNumber(hits) == block(i)*16-5)]; %Target Low-left; Condition number 6 and 11 of each block
                group_trials{3} = [group_trials{3}; find(BHV.ConditionNumber(hits) == block(i)*16-9); find(BHV.ConditionNumber(hits) == block(i)*16)];    %Target right; Condition number 7 and 16 of each block
                group_trials{7} = [group_trials{7}; find(BHV.ConditionNumber(hits) == block(i)*16-8); find(BHV.ConditionNumber(hits) == block(i)*16-1)];  %Target left; Condition number 8 and 15 of each block
            end
        end
        
    elseif trial_type == 4 % Target chosen (hits and misses)
        hits = find(BHV.TrialError == 0);
        fail = find(BHV.TrialError == 6);
        non_ignored = find(BHV.TrialError ~= 8);
        
        %Divide trials by target selection
        star = strfind(BHV.TimingFileByCond, 'visual_star.m'); star_conditions = find(not(cellfun('isempty', star)));
        circle = strfind(BHV.TimingFileByCond, 'visual_circle.m'); circle_conditions = find(not(cellfun('isempty', circle)));
        star_hit_trials = find(ismember(BHV.ConditionNumber(hits), star_conditions));
        star_fail_trials = find(ismember(BHV.ConditionNumber(fail), circle_conditions));
        star_trials = [hits(star_hit_trials); fail(star_fail_trials)];
        circle_hit_trials = find(ismember(BHV.ConditionNumber(hits), circle_conditions));
        circle_fail_trials = find(ismember(BHV.ConditionNumber(fail), star_conditions));
        circle_trials = [hits(circle_hit_trials); fail(circle_fail_trials)];
        
        group_trials{1} = star_trials; % Including both hits and misses
        group_trials{2} = circle_trials;
    end
    
    %Create the label matrix.
    for i = 1:length(group_trials)
        Labels(group_trials{i}) = i;
    end
    Labels = Labels';
    
    clear BHV % Save some memory
    %% Calculate baseline firing rate drift to correct for it later on
    [baseline_activity,baseline_spike_count] = w8a_drift_correction(SpikeData, hits, words_per_trials); %Calls drift correction function to obtain baseline FR of each trial for each neuron
    
    %% Transform data to the format of an SVM package
    % Create input matrix and label matrix inputing the firing rate of neurons during the delay epoch for hit trials.
    window_size = 0.03; %Size of the integration window in seconds
    Time_points = 15:30:4965; %Time points will be calculated (sliding window)
    % Time_points = 5:10:4995;
    epoch_num = 0;
    total_epoch = length(Time_points);
    
    hWaitbar = waitbar(0, ['Computing within-epoch decoding accuracy for session ' num2str(Session) '/' num2str(length(neural_dir))]);
    
    for epoch = Time_points %For each time point
        
        epoch_num = epoch_num +1;
        
        waitbar(epoch_num/total_epoch, hWaitbar)
        
        clear Input_matrix spike_count
        % Create the input matrix for the epoch of interest
        for i = 1:length(Unit_rasters) %For all units
            for k = 1:length(hits) %For all hit trials
                
                spikes = find(Unit_rasters{1,i}(k,epoch-(window_size*1000/2)+1:epoch+(window_size*1000/2)) == 1); %Find indicies of spikes during this analysis window
                Input_matrix(k,i) = length(spikes)/window_size; %Fill in matrix with firing rate in Hz for period of interest, for that trial
                spike_count(k,i) = length(spikes); % Fill in matrix with spike for period of interest, for that trial
                clear spikes
            end
        end
        
        %Perform drift correction and exclusion of big drift neurons
        for unit = 1:size(Input_matrix,2) % for each neuron
            if ~ismember(unit, Floor_neurons) % If this neuron is not part of the floor neurons to which no correction should be applied
                Input_matrix(:,unit) = Input_matrix(:,unit) - baseline_activity(unit,:)'; %Apply correction only to that unit
                %spike_count(:,unit) = spike_count(:,unit);
            end
        end
        Input_matrix(:,Big_drift_neurons) = []; %Eliminate big drift neurons from subsequent analyses
        spike_count(:,Big_drift_neurons) = []; %Eliminate big drift neurons from subsequent analyses
        spike_count_mat(:,:,epoch_num) = spike_count; % spike matrix is not reducing baseline activity
        spikes_matrix(:,:,epoch_num) = Input_matrix; % firing rate matrix is reducing baseline activity
        
    end %Loop over time points
    
    close(hWaitbar)
    
%     Extracted_spikes.full_raster = Unit_rasters; % optional: raw spike trains 
    % Extracted_spikes.Spikes = spikes_matrix; % optional: firing rate matrix reducing baseline activity
    Extracted_spikes.Spikes_count = spike_count_mat; % spike matrix removing big drift neurons
    Extracted_spikes.Labels = Labels;
    save('Extracted_spikes_cue_drift_30.mat','Extracted_spikes');
    
    clear Extracted_spikes spikes_matrix
    clearvars -except Monk saving_folder neural_dir trial_type shuffle_noise_out Chance Session PathName 
    
end %Loop over sessions