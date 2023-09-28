%% w8a decoding cognition from body movements
% This script uses an SVM to decode the monkeys target choice from all body movement groups.
% It uses the average movement over the last 500 msec of the delay across all tasks.
% It uses body movements from the Churchland structure.
% Sdt 05/2022

%Select the monkey
disp(' ');
Monkey = input('Which monkey to analyze? Krieger (1) or Brad (2) or Luca (3)? : ');

%Select experiment directory
disp(' '); disp('Please select the experiment directory')
PathName = uigetdir('', 'Please select the experiment directory');
cd(PathName)
if Monkey == 1
    neural_dir = dir('K*');
    Monkey_name = 'Krieger';
elseif Monkey == 2
    neural_dir = dir('B*');
    Monkey_name = 'Brad';
elseif Monkey == 3
    neural_dir = dir('L*');
    Monkey_name = 'Luca';
end

cue_epoch = 0; %select 1 if want to run on cue epoch

for Session = 1:length(neural_dir) %for all sessions for this monkey
    
    session_name = neural_dir(Session).name;
    flag_eyedata = 0; %Possibility to load and check eye and head data.
    
    %% Load and organize neural data
    [SpikeData, Timestamp_words, Words, BHV, head_position, words_per_trials, block] = w8a_Load_Organize_Data_batch(PathName, session_name, flag_eyedata);
    
    %% Load churchland structure
    try
        load([PathName '/' session_name '/' session_name '_Churchland_data_all.mat']) %all refers to the structure that includes DLC variables
    catch %if this session doesn't have Churchland data
        disp(['Skipping session ' session_name])
        continue  %skip to next session
    end
    disp(['Loaded ' session_name '_Churchland_data_all.mat'])
    
    %% Get some trial info from BHV for verification purposes
    
    hits = find(BHV.TrialError == 0);
    star = strfind(BHV.TimingFileByCond, 'visual_star'); star_conditions = find(not(cellfun('isempty', star)));
    circle = strfind(BHV.TimingFileByCond, 'visual_circle'); circle_conditions = find(not(cellfun('isempty', circle)));
    star_hit_trials = find(ismember(BHV.ConditionNumber(hits), star_conditions));
    circle_hit_trials = find(ismember(BHV.ConditionNumber(hits), circle_conditions));
    
    %(REMEMBER: the first head data point is aligned to first word == 9, which can be more than 10 msec after eye data collection starts!)
    for i = 1:length(hits)
        trialstart(i) = BHV.CodeTimes{1,hits(i)}(BHV.CodeNumbers{1,hits(i)} == 9);
        cue_onset_BHV(i) = BHV.CodeTimes{1,hits(i)}(BHV.CodeNumbers{1,hits(i)} == 12) - trialstart(i); %trial start time is substracted from other time points to account for shorter head timeseries starting at word == 9
        cue_off_BHV(i) = BHV.CodeTimes{1,hits(i)}(BHV.CodeNumbers{1,hits(i)} == 14) - trialstart(i);
        targets_onset_BHV(i) = BHV.CodeTimes{1,hits(i)}(BHV.CodeNumbers{1,hits(i)} == 25) - trialstart(i);
        response_BHV(i) = BHV.CodeTimes{1,hits(i)}(BHV.CodeNumbers{1,hits(i)} == 40) - trialstart(i);
    end
    
    
    %% Get timestamps of delay epoch from long msec time vector in Churchland structure
    delay_length = 500; %select delay length, going back from target onset
    
    Target_onsets = find(Task_events.Targets_onset); %identify target onsets timestamps (only present when a target comes up)
    Cue_onsets = find(Task_events.Cue_onset); %identify cue onsets timestamps
    trials = [find(Task_events.Trial_start)' find(Task_events.Trial_end)']; %Find trial onsets and end timestamps (all trials are included here)
    
    j = 1;
    for i = 1:length(trials) %for all trials
        
        T_onsets = Target_onsets(Target_onsets > trials(i,1) & Target_onsets < trials(i,2)); % find target onset timestamp within each trial interval
        cue_on = Cue_onsets(Cue_onsets > trials(i,1) & Cue_onsets < trials(i,2)); % find target onset timestamp within each trial interval
        if isempty(T_onsets) %if there are none (in case of ignored trial)
            T_on(j) = NaN; %leave a blank space to keep numbering commensurate
            C_on(j) = NaN; %leave a blank space to keep numbering commensurate
        else
            T_on(j) = T_onsets;
            C_on(j) = cue_on;
        end
        j = j + 1;
    end
    
    Target_onsets_hits = T_on(Trial_Info.Outcome == 0)'; % select only target onsets of hit trials
    Cue_onsets_hits = C_on(Trial_Info.Outcome == 0)'; % select only target onsets of hit trials
    Target_chosen_hits = Trial_Info.Target(Trial_Info.Outcome == 0)'; % select target chosen on hit trials
    
    Delay_epochs = [Target_onsets_hits-delay_length Target_onsets_hits]; %define boundaries of delay epochs for all hit trials
    if cue_epoch == 1
        Delay_epochs = [Cue_onsets_hits-300 Cue_onsets_hits+200]; %beginning of cue ; shouldnt have much predictability
    end
    
    %Verification
    if length(Target_onsets_hits) ~= length(targets_onset_BHV)
        error('Number of hit trials doesnt correspond with BHV')
    end
    
    %% Extract different movement effectors. Take average position during delay and compare across target conditions
    
    %Cheek
    for i = 1:length(Delay_epochs) %for all trials
        cheek(i,1) = nanmean(Tracking.head.Xpos(Delay_epochs(i,1):Delay_epochs(i,2))); % take the mean value of this motor effector during the delay.
        cheek(i,2) = nanmean(Tracking.head.Ypos(Delay_epochs(i,1):Delay_epochs(i,2)));
        cheek(i,3) = nanmean(Tracking.head.Zpos(Delay_epochs(i,1):Delay_epochs(i,2)));
        cheek(i,4) = nanmean(Tracking.head_velocity(Delay_epochs(i,1):Delay_epochs(i,2)));
        cheek(i,5) = nanmean(Tracking.head_distance_traveled(Delay_epochs(i,1):Delay_epochs(i,2)));
        cheek(i,6) = nanmean(Tracking.head_acceleration(Delay_epochs(i,1):Delay_epochs(i,2)));
        cheek(i,7) = nanmean(Tracking.head_grid_ID(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    %Eyes
    for i = 1:length(Delay_epochs)
        eyes(i,1) = nanmean(Tracking.eyes.Xpos(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,2) = nanmean(Tracking.eyes.Ypos(Delay_epochs(i,1):Delay_epochs(i,2)));
        if isfield(Tracking, 'calibrated_eyes') %Is absent in head fixed sessions
            eyes(i,3) = nanmean(Tracking.calibrated_eyes.Xpos(Delay_epochs(i,1):Delay_epochs(i,2)));
            eyes(i,4) = nanmean(Tracking.calibrated_eyes.Ypos(Delay_epochs(i,1):Delay_epochs(i,2)));
        end
        eyes(i,5) = nanmean(Tracking.pupil_size(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,6) = nanmean(Tracking.eye_distance_traveled(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,7) = nanmean(Tracking.eye_velocity(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,8) = nanmean(Tracking.eye_acceleration(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,9) = nanmean(Tracking.eye_eccentricity(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,10) = nanmean(Tracking.eye_azimuth(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,11) = nanmean(Tracking.eye_direction(Delay_epochs(i,1):Delay_epochs(i,2)));
        eyes(i,12) = nanmean(Tracking.eye_grid_ID(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    % SVD video
    for i = 1:length(Delay_epochs)
        SVD(i,1) = nanmean(Tracking.SVD_raw_video(Delay_epochs(i,1):Delay_epochs(i,2)));
        SVD(i,2) = nanmean(Tracking.SVD_ME_video(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    %Right hand
    for i = 1:length(Delay_epochs)
        R_hand(i,1) = nanmean(Tracking.right_hand.x(Delay_epochs(i,1):Delay_epochs(i,2)));
        R_hand(i,2) = nanmean(Tracking.right_hand.y(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    if Monkey == 1
        %Left hand
        for i = 1:length(Delay_epochs)
            L_hand(i,1) = nanmean(Tracking.left_hand.x(Delay_epochs(i,1):Delay_epochs(i,2)));
            L_hand(i,2) = nanmean(Tracking.left_hand.y(Delay_epochs(i,1):Delay_epochs(i,2)));
        end
    end
    
    if Monkey == 3
        %Tail
        for i = 1:length(Delay_epochs)
            Tail(i,1) = nanmean(Tracking.tail.x(Delay_epochs(i,1):Delay_epochs(i,2)));
            Tail(i,2) = nanmean(Tracking.tail.y(Delay_epochs(i,1):Delay_epochs(i,2)));
        end
    end
    
    %Nose
    for i = 1:length(Delay_epochs)
        Nose(i,1) = nanmean(Tracking.nose.x(Delay_epochs(i,1):Delay_epochs(i,2)));
        Nose(i,2) = nanmean(Tracking.nose.y(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    %Eyebrow
    if isfield(Tracking, 'eyebrow') %Is absent in head fixed sessions
        for i = 1:length(Delay_epochs)
            Eyebrow(i,1) = nanmean(Tracking.eyebrow.x(Delay_epochs(i,1):Delay_epochs(i,2)));
            Eyebrow(i,2) = nanmean(Tracking.eyebrow.y(Delay_epochs(i,1):Delay_epochs(i,2)));
        end
    end
    
    %Head tilt & direction DLC
    if isfield(Tracking, 'head_tilt') %Is absent in head fixed sessions
        for i = 1:length(Delay_epochs)
            Head_DLC(i,1) = nanmean(Tracking.head_tilt(Delay_epochs(i,1):Delay_epochs(i,2)));
            Head_DLC(i,2) = nanmean(Tracking.head_direction(Delay_epochs(i,1):Delay_epochs(i,2)));
        end
    end
    
    %Head_ME
    for i = 1:length(Delay_epochs)
        Head_ME(i,1) = nanmean(Motion_energy.head(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    %Arm_ME
    for i = 1:length(Delay_epochs)
        Arm_ME(i,1) = nanmean(Motion_energy.arm(Delay_epochs(i,1):Delay_epochs(i,2)));
    end
    
    
    
    %% Decoding
    if Monkey == 1
        if ~isfield(Tracking, 'eyebrow') %Is absent in head fixed sessions
            body_list = {cheek, eyes, SVD, R_hand, L_hand, Nose, Head_ME, Arm_ME};
            body_label = ["cheek" "eyes" "SVD" "R_hand" "L_hand" "Nose" "Head_ME" "Arm_ME"];
        else
            body_list = {cheek, eyes, SVD, R_hand, L_hand, Nose, Eyebrow, Head_DLC, Head_ME, Arm_ME};
            body_label = ["cheek" "eyes" "SVD" "R_hand" "L_hand" "Nose" "Eyebrow" "Head_DLC" "Head_ME" "Arm_ME"];
        end
    elseif Monkey == 3
        if ~isfield(Tracking, 'eyebrow') %Is absent in head fixed sessions
            body_list = {cheek, eyes, SVD, R_hand, Tail, Nose, Head_ME, Arm_ME};
            body_label = ["cheek" "eyes" "SVD" "R_hand" "Tail" "Nose" "Head_ME" "Arm_ME"];
        else
            body_list = {cheek, eyes, SVD, R_hand, Tail, Nose, Eyebrow, Head_DLC, Head_ME, Arm_ME};
            body_label = ["cheek" "eyes" "SVD" "R_hand" "Tail" "Nose" "Eyebrow" "Head_DLC" "Head_ME" "Arm_ME"];
        end
    end
    
    %Body part accuracy
    counter = 1;
    for body_part = body_list
        
        Input_matrix = body_part{1};
        Labels = Target_chosen_hits;
        kfolds = 5;
        ldaflag = 0;
        Normalize = 1;
        
        %remove rows with NaN that otherwise bring decoding accuracy to chance
        Input_matrix_nonan = Input_matrix;
        Input_matrix_nonan(any(isnan(Input_matrix), 2),:) = [];
        Labels_nonan = Labels;
        Labels_nonan(any(isnan(Input_matrix), 2),:) = [];
        
        for i = 1:30
            [hitrate(i)] = w8a_SVM_basic_function(Input_matrix_nonan, Labels_nonan, kfolds, ldaflag, Normalize);
        end
        
        Accuracy.(body_label(counter))(Session) = mean(hitrate);
        
        counter = counter + 1;
        
    end
    
    %Full body accuracy
    if Monkey == 1
        if ~isfield(Tracking, 'eyebrow') %Is absent in head fixed sessions
            body_list = [cheek eyes SVD R_hand L_hand Nose Head_ME Arm_ME];
            body_label = ["Full_body"];
        else
            body_list = [cheek eyes SVD R_hand L_hand Nose Eyebrow Head_DLC Head_ME Arm_ME];
            body_label = ["Full_body"];
        end
    elseif Monkey == 3
        if ~isfield(Tracking, 'eyebrow') %Is absent in head fixed sessions
            body_list = [cheek eyes SVD R_hand Tail Nose Head_ME Arm_ME];
            body_label = ["Full_body"];
        else
            body_list = [cheek eyes SVD R_hand Tail Nose Eyebrow Head_DLC Head_ME Arm_ME];
            body_label = ["Full_body"];
        end
    end
    
    Input_matrix = body_list;
    Labels = Target_chosen_hits;
    kfolds = 5;
    ldaflag = 0;
    Normalize = 1;
    
    %remove rows with NaN that otherwise bring decoding accuracy to chance
    Input_matrix_nonan = Input_matrix;
    Input_matrix_nonan(any(isnan(Input_matrix), 2),:) = [];
    Labels_nonan = Labels;
    Labels_nonan(any(isnan(Input_matrix), 2),:) = [];
    
    for i = 1:30
        [hitrate(i)] = w8a_SVM_basic_function(Input_matrix_nonan, Labels_nonan, kfolds, ldaflag, Normalize);
    end
    
    Accuracy.(body_label)(Session) = mean(hitrate);
    
    
    clearvars -except Session PathName neural_dir Monkey_name Accuracy Monkey cue_epoch
    
end

if Monkey == 1
    cheek = mean(nonzeros(Accuracy_K.cheek))
    eyes = mean(nonzeros(Accuracy_K.eyes))
    SVD = mean(nonzeros(Accuracy_K.SVD))
    R_hand = mean(nonzeros(Accuracy_K.R_hand))
    L_hand = mean(nonzeros(Accuracy_K.L_hand))
    Nose = mean(nonzeros(Accuracy_K.Nose))
    Eyebrow = mean(nonzeros(Accuracy_K.Eyebrow))
    Head_DLC = mean(nonzeros(Accuracy_K.Head_DLC))
    Head_ME = mean(nonzeros(Accuracy_K.Head_ME))
    Arm_ME = mean(nonzeros(Accuracy_K.Arm_ME))
    Full_body = mean(nonzeros(Accuracy_K.Full_body))
end

if Monkey == 3
    cheek = mean(nonzeros(Accuracy_L.cheek))
    eyes = mean(nonzeros(Accuracy_L.eyes))
    SVD = mean(nonzeros(Accuracy_L.SVD))
    R_hand = mean(nonzeros(Accuracy_L.R_hand))
    Tail = mean(nonzeros(Accuracy_L.Tail))
    Nose = mean(nonzeros(Accuracy_L.Nose))
    Eyebrow = mean(nonzeros(Accuracy_L.Eyebrow))
    Head_DLC = mean(nonzeros(Accuracy_L.Head_DLC))
    Head_ME = mean(nonzeros(Accuracy_L.Head_ME))
    Arm_ME = mean(nonzeros(Accuracy_L.Arm_ME))
    Full_body = mean(nonzeros(Accuracy_L.Full_body))
end

%% For plotting circles of size proportional to accuracy on monkkey drawings
% 2pt is 50%, 7pt is 100%
(cheek-.5) * 10 + 2
(eyes-.5) * 10 + 2
(R_hand-.5) * 10 + 2
(L_hand-.5) * 10 + 2
(Tail-.5) * 10 + 2
(Nose-.5) * 10 + 2
(Eyebrow-.5) * 10 + 2
(Head_ME-.5) * 10 + 2
(Arm_ME-.5) * 10 + 2
(Full_body-.5) * 10 + 2

%% Plot tables comparing dedoding across twist tasks

bar(compare_K')
ylim([.5 1])
set(gca,'xticklabel',{'cheek', 'eyes' 'R hand' 'L hand' 'Nose' 'Head ME' 'Arm ME' 'Full body'})
ylabel('Decoding accuracy (%)')
title('Krieger decoding target from movements during the delay')

bar(compare_L')
ylim([.5 1])
set(gca,'xticklabel',{'cheek', 'eyes' 'R hand' 'Tail' 'Nose' 'Eyebrow' 'Head ME' 'Arm ME' 'Full body'})
ylabel('Decoding accuracy (%)')
title('Luca decoding target from movements during the delay')
legend('Free', 'Fixed', 'Biased')

