%% w8a_SVM_batch_Biased_CAT
% This script runs a SVM on a sliding window over the timecourse of trials in the biased CAT tasks. The SVM
% computes the classification accuracy % predicting tasks or behavioural variables based on neural population activity. The script uses
% LIBSVM to run a nonlinear SVM algorithm (RBF kernel). It automatically balances classes that have unequal number of observations, and
% processes a baseline firing rate drift correction. Large drift neurons are eliminated. Lazy neurons are not eliminated. The LDA is
% deactivated by default. There is no hyperparameter search manually implemented.
% SDT 03/2020

%Select the monkey
Monkey = input('Which monkey to analyze? Krieger (1) or Brad (2) or Luca (3) ? : ');
%Select experiment directory
cd('/Users/SeB/Documents/University/Postdoc/Petrides/Projects/Wireless project/Sessions')
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

%% Set Parameters for SVM analyses
% Choose the variable of interest.
trial_type = input('For target press 1, for cue press 2, for location press 3, for target chosen press 4, for cue presented press 5: ');
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
    exception_4cues = 0; %This is a flag for Luca's sessions that have one block but 4 cues.
    
    if length(unique(BHV.BlockNumber)) == 1
        warning('This session has only one block, not two. There is one pair of cues, not two. Cue and target selectivity are therefore identical')
        if session_name == 'L180618' | session_name == 'L190618' | session_name == 'L210618'
            Trials_block1 = find(BHV.ConditionNumber == 9 | BHV.ConditionNumber == 10);
            Trials_block2 = find(BHV.ConditionNumber == 11 | BHV.ConditionNumber == 12);
            %Identify the hit trials in each block
            hit_block1 = hits(ismember(hits, Trials_block1));
            hit_block2 = hits(ismember(hits, Trials_block2));
            %Pool the hits from both blocks (Corrupts chronology of trials. Shouldn't matter for SDF)
            hits = [hit_block1; hit_block2];
            exception_4cues = 1;
        end
    elseif length(unique(BHV.BlockNumber)) == 2
        %Identify the trial numbers part of each block
        block = unique(BHV.BlockNumber); %Identify blocks played in this session
        Trials_block1 = find(BHV.BlockNumber == block(1));
        Trials_block2 = find(BHV.BlockNumber == block(2));
        %Identify the hit trials in each block
        hit_block1 = hits(ismember(hits, Trials_block1));
        hit_block2 = hits(ismember(hits, Trials_block2));
        %Pool the hits from both blocks (Corrupts chronology of trials. Shouldn't matter for SDF)
        hits = [hit_block1; hit_block2];
    elseif length(unique(BHV.BlockNumber)) > 2
        error('This session has more than 2 blocks played. You will need to specify manually which are the two blocks to analyse')
    end
    
    %Get the timestamp of cue and target onset in the neural data for each hit trial (Not in chronological order)
    for i = 1:length(hits)
        cue_length(i) = round((words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 14) - words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 12))*1000); %Identify time of cue presentation epoch
        delay_length(i) = round((words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 25) - words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 14))*1000); %Identify time of cue presentation epoch
        Cue_onset_neural(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 12);
        Targets_onset(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 25);
        Response(i) = words_per_trials(hits(i)).timestamps(words_per_trials(hits(i)).words == 40);
    end
    Cue_onset_neural = Cue_onset_neural';
    Targets_onset = Targets_onset';
    Response = Response';
    
    cue_presentation_length = round(mean(cue_length));
    if ~isempty(find(abs(diff(cue_length))>100)) % If the cue presentation length varies by more than 100 msec across all trials, issue a warning that mutliple cue lengths were selected during this session
        warning('The cue lenght in this session is not a constant. This is problematic for ploting')
    end
    
    min_delay_length = min(delay_length);
    max_delay_length = max(delay_length);
    mean_delay_length = round(mean(delay_length));
    half_delay_length = round(mean_delay_length/2);
    median_reaction_time = round(median(Response - Targets_onset)*1000);
    half_reaction_time = round(median_reaction_time/2);
    
    %Define your analysis periods (in seconds)
    Cue_window_length = 1500 + cue_presentation_length + half_delay_length; %msec
    Target_window_length = half_delay_length + half_reaction_time;
    Response_window_length = half_reaction_time+1000;
    Analysis_period_lenght = Cue_window_length+Target_window_length+Response_window_length; %msec
    
    %Define the timestamps limits for the analysis period for each trial
    Analysis_period_cue = [(Cue_onset_neural - 1.5) (Cue_onset_neural + (Cue_window_length - 1500)/1000)];
    Analysis_period_target = [(Targets_onset - half_delay_length/1000) (Targets_onset + half_reaction_time/1000)];
    Analysis_period_response = [(Response - half_reaction_time/1000) (Response + 1)];
    
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
        star = strfind(BHV.TimingFileByCond, 'visual_star'); star_conditions = find(not(cellfun('isempty', star)));
        circle = strfind(BHV.TimingFileByCond, 'visual_circle'); circle_conditions = find(not(cellfun('isempty', circle)));
        group_trials{1} = find(ismember(BHV.ConditionNumber(hits), star_conditions)); % indices of star trials among hit trials vector (hits)
        group_trials{2} = find(ismember(BHV.ConditionNumber(hits), circle_conditions));
        
    elseif trial_type == 2 % Cue
        if length(unique(BHV.BlockNumber)) == 2 | exception_4cues %If there are two pairs of cues
            star = strfind(BHV.TimingFileByCond, 'visual_star'); star_conditions = find(not(cellfun('isempty', star)));
            circle = strfind(BHV.TimingFileByCond, 'visual_circle'); circle_conditions = find(not(cellfun('isempty', circle)));
            star_trials = find(ismember(BHV.ConditionNumber, star_conditions)); %Find all star trials, hits or not
            circle_trials = find(ismember(BHV.ConditionNumber, circle_conditions)); %Find all circle trials, hits or not
            group_trials{1} = find(ismember(hit_block1, star_trials)); %Identify indices of hit trials that belong to block1 and to star trials (indices are the same for hits and hit_block1 vector
            group_trials{2} = find(ismember(hit_block1, circle_trials));
            group_trials{3} = find(ismember([zeros(length(hit_block1),1);hit_block2], star_trials)); %Identify hit trials that belong to block2 and to star trials (indices are NOT the same for hits and hit_block2 vector. Need to add an empty vector in place of hit_block1 to make the indices match)
            group_trials{4} = find(ismember([zeros(length(hit_block1),1);hit_block2], circle_trials));
        else
            warning('This session does not have two pairs of cues. Cue decoding is skipped.')
            clearvars -except Monk saving_folder neural_dir trial_type shuffle_noise_out Chance Session SVM_results SVM_results_chance SVM_results_shuffle ConfMat_results PathName
            continue
        end
        
    elseif trial_type == 3 % Location
        if session_name == 'K010818' | session_name == 'K020818'
            %Define target position
            position1_trials = find(BHV.ConditionNumber(hits) == 17 | BHV.ConditionNumber(hits) == 21 | BHV.ConditionNumber(hits) == 20 | BHV.ConditionNumber(hits) == 24); %Up
            position5_trials = find(BHV.ConditionNumber(hits) == 18 | BHV.ConditionNumber(hits) == 22 | BHV.ConditionNumber(hits) == 19 | BHV.ConditionNumber(hits) == 23); %down
            group_location(position1_trials) = 1; group_location(position5_trials) = 5;
        elseif session_name == 'K080818'
            position1_trials = find(BHV.ConditionNumber(hits) == 21 | BHV.ConditionNumber(hits) == 26 | BHV.ConditionNumber(hits) == 29 | BHV.ConditionNumber(hits) == 34); %Up
            position5_trials = find(BHV.ConditionNumber(hits) == 22 | BHV.ConditionNumber(hits) == 25 | BHV.ConditionNumber(hits) == 30 | BHV.ConditionNumber(hits) == 33); %down
            position3_trials = find(BHV.ConditionNumber(hits) == 23 | BHV.ConditionNumber(hits) == 28 | BHV.ConditionNumber(hits) == 31 | BHV.ConditionNumber(hits) == 36); %Right
            position7_trials = find(BHV.ConditionNumber(hits) == 24 | BHV.ConditionNumber(hits) == 27 | BHV.ConditionNumber(hits) == 32 | BHV.ConditionNumber(hits) == 35); %Left
            group_location(position1_trials) = 1; group_location(position5_trials) = 5; group_location(position3_trials) = 3; group_location(position7_trials) = 7;
        elseif session_name == 'K120818'
            position1_trials = find(BHV.ConditionNumber(hits) == 37 | BHV.ConditionNumber(hits) == 42); %Up
            position5_trials = find(BHV.ConditionNumber(hits) == 38 | BHV.ConditionNumber(hits) == 41); %down
            position3_trials = find(BHV.ConditionNumber(hits) == 39 | BHV.ConditionNumber(hits) == 44); %Right
            position7_trials = find(BHV.ConditionNumber(hits) == 40 | BHV.ConditionNumber(hits) == 43); %Left
            group_location(position1_trials) = 1; group_location(position5_trials) = 5; group_location(position3_trials) = 3; group_location(position7_trials) = 7;
        elseif session_name == 'K130818'
            position1_trials = find(BHV.ConditionNumber(hits) == 37 | BHV.ConditionNumber(hits) == 42 | BHV.ConditionNumber(hits) == 45 | BHV.ConditionNumber(hits) == 50); %Up
            position5_trials = find(BHV.ConditionNumber(hits) == 38 | BHV.ConditionNumber(hits) == 41 | BHV.ConditionNumber(hits) == 46 | BHV.ConditionNumber(hits) == 49); %down
            position3_trials = find(BHV.ConditionNumber(hits) == 39 | BHV.ConditionNumber(hits) == 44 | BHV.ConditionNumber(hits) == 47 | BHV.ConditionNumber(hits) == 52); %Right
            position7_trials = find(BHV.ConditionNumber(hits) == 40 | BHV.ConditionNumber(hits) == 43 | BHV.ConditionNumber(hits) == 48 | BHV.ConditionNumber(hits) == 51); %Left
            group_location(position1_trials) = 1; group_location(position5_trials) = 5; group_location(position3_trials) = 3; group_location(position7_trials) = 7;
        end
        
        if  session_name == 'L180618' | session_name == 'L190618' | session_name == 'L210618'
            position2_trials = find(BHV.ConditionNumber(hits) == 9 | BHV.ConditionNumber(hits) == 11); %Up-right
            position6_trials = find(BHV.ConditionNumber(hits) == 10 | BHV.ConditionNumber(hits) == 12); %down-left
            group_location(position2_trials) = 2; group_location(position6_trials) = 6;
        end
        
        Unique_locations = unique(group_location);
        for i = 1:length(Unique_locations)
            group_trials{i} = find(group_location == Unique_locations(i));
        end
        
    elseif trial_type == 4 % Target chosen (hits and misses)
        hits = find(BHV.TrialError == 0);
        fail = find(BHV.TrialError == 6);
        non_ignored = find(BHV.TrialError ~= 8);
        
        %Divide trials by target selection
        star = strfind(BHV.TimingFileByCond, 'visual_star'); star_conditions = find(not(cellfun('isempty', star)));
        circle = strfind(BHV.TimingFileByCond, 'visual_circle'); circle_conditions = find(not(cellfun('isempty', circle)));
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
    [baseline_activity] = w8a_drift_correction(SpikeData, hits, words_per_trials); %Calls drift correction function to obtain baseline FR of each trial for each neuron
    
    %% Transform data to the format of an SVM package
    % Create input matrix and label matrix inputing the firing rate of neurons during the delay epoch for hit trials.
    window_size = 0.4; %Size of the integration window in seconds
    Time_points = 300:100:Analysis_period_lenght-200; %Time points over which the SVM will be calculated (sliding window)
    epoch_num = 0;
    total_epoch = length(Time_points);
    
    hWaitbar = waitbar(0, ['Computing within-epoch decoding accuracy for session ' num2str(Session) '/' num2str(length(neural_dir))]);
    
    for epoch = Time_points %For each time point
        
        epoch_num = epoch_num +1;
        
        waitbar(epoch_num/total_epoch, hWaitbar)
        
        clear Input_matrix
        % Create the SVM input matrix for the epoch of interest
        for i = 1:length(Unit_rasters) %For all units
            for k = 1:length(hits) %For all hit trials
                spikes = find(Unit_rasters{1,i}(k,epoch-(window_size*1000/2)+1:epoch+(window_size*1000/2)) == 1); %Find indicies of spikes during this analysis window
                Input_matrix(k,i) = length(spikes)/window_size; %Fill in matrix with firing rate in Hz for period of interest, for that trial
                clear spikes
            end
        end
        
        %Perform drift correction and exclusion of big drift neurons
        for unit = 1:size(Input_matrix,2) % for each neuron
            if ~ismember(unit, Floor_neurons) % If this neuron is not part of the floor neurons to which no correction should be applied
                Input_matrix(:,unit) = Input_matrix(:,unit) - baseline_activity(unit,:)'; %Apply correction only to that unit
            end
        end
        Input_matrix(:,Big_drift_neurons) = []; %Eliminate big drift neurons from subsequent analyses
        
        %% Set parameters to run SVM
        kfolds = 5; %Number of folds in crossvalidation
        ldaflag = 0; %Do you want to run an LDA before SVM
        Normalize = 1; %Do you want to standardize your data (z-score)
        
        tic
        for iteration = 1:20
            if Chance == 1 % Shuffling operations for obtaining chance performance
                Labels = Labels(randperm(length(Labels)));
            end
            [hitrate(iteration,epoch_num), C(:,:,epoch_num,iteration)] = w8a_SVM_basic_function(Input_matrix, Labels, kfolds, ldaflag, Normalize);
        end
        toc
        
    end %Loop over time points
    
    close(hWaitbar)
    
    if shuffle_noise_out == 1
        SVM_results_shuffle{Session,:} =  mean(hitrate,1);
    elseif Chance == 1
        SVM_results_chance{Session,:} =  mean(hitrate,1);
    else
        SVM_results{Session,:} =  mean(hitrate,1);
        ConfMat_results{Session,:} =  mean(C,4);
    end
    
    clearvars -except Monk saving_folder neural_dir trial_type shuffle_noise_out Chance Session SVM_results SVM_results_chance SVM_results_shuffle ConfMat_results PathName
    
end %Loop over sessions

return

%% Plotting SVM results
% From the appropriate folder, manually load one by one the structures contained SVM_results. Initiate the figure only once and run the code
% for each structure (cue, target, location).

figure
hold on

Session = 3
SVM_matrix = SVM_results{Session}*100

Time_points = 300:100:length(SVM_matrix)*100+200; %Time points over which the SVM will be calculated (sliding window)
x = Time_points;

xlim([500 5000])
ylabel('Decoding Accuracy')
xlabel('Time (ms)')

%Krieger
line([1500 1500], [0 100], 'Color', 'g') % mark cue onset
line([1500+317 1500+317], [0 100], 'Color', 'g') % mark cue offset
line([1500+575 1500+575], [0 100], 'Color', 'k', 'LineStyle', '--') % mark delay axis break
line([1500+833 1500+833], [0 100], 'Color', 'r') % mark target onset
line([1500+1089 1500+1089], [0 100], 'Color', 'k', 'LineStyle', '--') % mark reaction time axis break
line([1500+1344 1500+1344], [0 100], 'Color', 'c') % mark response onset

%Luca
line([1500 1500], [0 100], 'Color', 'g') % mark cue onset
line([1500+1017 1500+1017], [0 100], 'Color', 'g') % mark cue offset
line([1500+1500 1500+1500], [0 100], 'Color', 'k', 'LineStyle', '--') % mark delay axis break
line([1500+1989 1500+1989], [0 100], 'Color', 'r') % mark target onset
line([1500+2218 1500+2218], [0 100], 'Color', 'k', 'LineStyle', '--') % mark reaction time axis break
line([1500+2446 1500+2446], [0 100], 'Color', 'c') % mark response onset

if trial_type == 1 % If decoding target
    line([0 5000],[50 50],'Color', 'r','LineStyle', '--') % mark chance level
    %fill([x fliplr(x)], [Mean_SVM+Error_SVM, fliplr(Mean_SVM-Error_SVM)], 'b')
    plot(x, SVM_matrix, 'LineWidth', 2, 'Color', 'b');
elseif trial_type == 2 % if decoding cue
    line([0 5000],[25 25],'Color', 'r','LineStyle', '--') % mark chance level
    %fill([x fliplr(x)], [Mean_SVM+Error_SVM, fliplr(Mean_SVM-Error_SVM)], 'r')
    plot(x, SVM_matrix, 'LineWidth', 2, 'Color', 'r');
elseif trial_type == 3 % if decoding location
    line([0 5000],[12.5 12.5],'Color', 'r','LineStyle', '--') % mark chance level
    %fill([x fliplr(x)], [Mean_SVM+Error_SVM, fliplr(Mean_SVM-Error_SVM)], 'g')
    plot(x, SVM_matrix, 'LineWidth', 2, 'Color', 'g');
end

title('Decoding accuracy Biased CAT - Luca - L210618')


%% Plotting average SVM results
% From the appropriate folder, manually load one by one the structures contained SVM_results. Initiate the figure only once and run the code
% for each structure (cue, target, location).


figure
hold on

%Krieger
line([1500 1500], [0 100], 'Color', 'g') % mark cue onset
line([1500+317 1500+317], [0 100], 'Color', 'g') % mark cue offset
line([1500+575 1500+575], [0 100], 'Color', 'k', 'LineStyle', '--') % mark delay axis break
line([1500+833 1500+833], [0 100], 'Color', 'r') % mark target onset
line([1500+1089 1500+1089], [0 100], 'Color', 'k', 'LineStyle', '--') % mark reaction time axis break
line([1500+1344 1500+1344], [0 100], 'Color', 'c') % mark response onset

%Luca
line([1500 1500], [0 100], 'Color', 'g') % mark cue onset
line([1500+1017 1500+1017], [0 100], 'Color', 'g') % mark cue offset
line([1500+1500 1500+1500], [0 100], 'Color', 'k', 'LineStyle', '--') % mark delay axis break
line([1500+1989 1500+1989], [0 100], 'Color', 'r') % mark target onset
line([1500+2218 1500+2218], [0 100], 'Color', 'k', 'LineStyle', '--') % mark reaction time axis break
line([1500+2446 1500+2446], [0 100], 'Color', 'c') % mark response onset

%Restructure SVM_results into matrix to compute mean and error
for i = [3 5]
    SVM_matrix(i,:) = SVM_results{i}*100;
end

Mean_SVM = mean(SVM_matrix,1);
Error_SVM = std(SVM_matrix)/(size(SVM_matrix,1)^(1/2));

Analysis_period_lenght = 3800 %(Luca 4900)
Time_points = 300:100:Analysis_period_lenght-200; %Time points over which the SVM will be calculated (sliding window)

x = Time_points;

xlim([1000 4500])
ylabel('Decoding Accuracy (%)')
xlabel('Time from cue onset (msec)')
xticks([1000 1500 2000 2500 3000 3500 4000 4500])
xticklabels({'','0','','1000','','2000','', '3000'})
yticks([0 10 20 30 40 50 60 70 80 90 100])
yticklabels({'0','','20','','40','', '60', '', '80','','100'})

set(gca,'TickDir','out') % draw the tick marks on the outside


if trial_type == 1 % If decoding target
    line([0 5000],[50 50],'Color', 'b','LineStyle', '--') % mark chance level
    fill([x fliplr(x)], [Mean_SVM+Error_SVM, fliplr(Mean_SVM-Error_SVM)], 'b')
    plot(x, Mean_SVM, 'LineWidth', 2, 'Color', 'k');
elseif trial_type == 2 % if decoding cue
    line([0 5000],[25 25],'Color', 'r','LineStyle', '--') % mark chance level
    fill([x fliplr(x)], [Mean_SVM+Error_SVM, fliplr(Mean_SVM-Error_SVM)], 'r')
    plot(x, Mean_SVM, 'LineWidth', 2, 'Color', 'k');
elseif trial_type == 3 % if decoding location
    line([0 5000],[12.5 12.5],'Color', 'g','LineStyle', '--') % mark chance level
    fill([x fliplr(x)], [Mean_SVM+Error_SVM, fliplr(Mean_SVM-Error_SVM)], 'g')
    plot(x, Mean_SVM, 'LineWidth', 2, 'Color', 'k');
end

