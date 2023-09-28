%% w8a_Load_Organize_Data
% This script loads, organizes, and sanity checks behavioral and neural data from experiment w8a
% Make sure you spike sort your .nev file and extracts the SpikeData structure using Neuroshare before running the script
% Requires the .mat SpikeData structure and the .mat BHV file from Monkeylogic
% Optionally, you can load the EDF-converted .asc file from Eyelink and the analog data from the .ns2 file for checking eye
% data and extracting head movement matrices.
% Created by Testard and Tremblay 01/2017

%% Load behavioral and neural data
if ismac
    cd('~/Documents/University/Postdoc/Petrides/Projects/Wireless project/Sessions')
elseif ispc
    cd('/')
end
disp(' '); ifeyedata = input('Is there any eye data from EDF you want to analyze? 1 for yes / 2 for no; followed by return key: '); disp(' ')
disp('Select the BHV, analog, nev and problematic units files');
[FileName,PathName] = uigetfile('*.mat','Select the BHV, analog, and nev files','Multiselect', 'on');
load([PathName FileName{2}]) %Load BHV
disp(' '); disp(['Loaded ' PathName FileName{2}]); disp(' ')
load([PathName FileName{3}]) %Load NEV
disp(' '); disp(['Loaded ' PathName FileName{3}]); disp(' ')
% load([PathName FileName{4}]) % Load units which require special treatment. Either they will be discarded or drift correction will not be applied to them. 
% disp(' '); disp(['Loaded ' PathName FileName{4}]); disp(' ')
session_name = PathName(end-7:end-1); % Output session name to facilitate saving later on.

tic

%% HAS THE BHV BEEN RECORDED CORRECTLY?

if any(find(diff(BHV.TrialNumber) ~= 1))
    error('There was an error in the writing or reading of the BHV')
end 

%% ARE THERE ANY incomplete trials in BR?

if length(find(Words == 9)) >= length(BHV.TrialNumber) || length(find(Words == 18)) >= length(BHV.TrialNumber)
else
    error('There are missing words of an incomplete last trial in the Blackrock words')
end

%% IS THERE THE SAME NUMBER OF TRIALS IN ML(Monkeylogic) AND BR (BlackRock)?

if length(find(Words == 9)) >= length(BHV.TrialError)
else
    error('Unequal number of trials in BR and ML')
end

%% DETECT AND DELETE WRONG WORDS & MAKE SURE ALL THE TRIAL NUMBERS ARE CORRECT
% Because our neural recording system, Cereplex, does not accept a strobe bit, and because our DAQ card, the PCIe-6321
% from National Instrument uses software timing rather than hardware timing to synchronise its three digital ports,
% words can sometime be misread if the activation of the digital ports is not perfectly synchronous. This problem
% occurs only when the digital word sent is above 256, or 8 bits.

% Create a BHV Words vector to be able to compare easily Blackrock and ML words.
Words_BHV = [];
for i = 1:length(BHV.CodeNumbers)
    Words_BHV = [Words_BHV; BHV.CodeNumbers{1,i}];
end

% Use the timing method, based on the fact that there is an asynchrony
% between the two ports resulting in a submillisecond (0.2ms) separation between
% two words, which does not occur otherwise.
Diff = diff(Timestamp_words);
Outlier_indices = find(Diff<0.0005); %Fastest interval between two real words is 1 msec. Intervals below 1msec indicate bad word.
num_outliers = length(Outlier_indices);
Outliers = Words(Outlier_indices);

% Delete those outliers (this will delete the first word of a pair of word with too short time interval)
for i = 1:length(Outlier_indices)
    Words(Outlier_indices(i)) = 0;
    Timestamp_words(Outlier_indices(i)) = 0;
end
Words(Words == 0) = [];
Timestamp_words(Timestamp_words == 0)=[];

%Check if, after deleting bad words, you now have the same number of words in BR and ML
if length(Words_BHV) == length(Words)
else
    error('Unequal number of words between BR and ML')
end

% Now some words that remain are not the correct value because bits on the first port had time to deactivate before bits
% on the second port activated. Copy Words Identity from ML and paste them in BR.
Words(:) = Words_BHV;

%Check if there are the same number of words and timestamp words in the BR data
if length(Words) == length(Timestamp_words)
else
    error('Unequal number of words and timestamps of words in BR following word correction')
end

clearvars -except SpikeData Words Timestamp_words BHV ifeyedata PathName FileName omit_correction units_to_discard session_name


%% CHECK SYNCHRONY BETWEEN ML and BR SYSTEMS
% Comparing the time difference between start (word 9) and end (word 18) of each trial in ML vs BR

% Compute trial length for each trial in the monkeylogic clock
for i = 1:length(BHV.CodeTimes)
    Trial_length_BHV(i,1) = (BHV.CodeTimes{1,i}(BHV.CodeNumbers{1,i} == 18) - BHV.CodeTimes{1,i}(BHV.CodeNumbers{1,i} == 9)); %in ms
end

% Compute trial length for each trial in the blackrock clock
Trial_length_BR = round(Timestamp_words(Words(:) == 18)*1000 - Timestamp_words(Words(:) == 9)*1000); %in ms

% Compare blackrock and ML clocks
Within_trial_Diff_between_clocks = abs(Trial_length_BR - Trial_length_BHV);
max_trial_length_diff_between_clocks = max(Within_trial_Diff_between_clocks); % in ms
mean_trial_length_diff = mean(Within_trial_Diff_between_clocks);

if max_trial_length_diff_between_clocks > 2
    error('Synchrony of ML and BR clocks is not acceptable. Please check vector "Within_trial_Diff_between_clocks"')
end

clearvars -except SpikeData Timestamp_words Words BHV ifeyedata PathName FileName omit_correction units_to_discard session_name

%% Organize Words on a trial basis
starts = find(Words == 9);
ends = find(Words == 18);
for x = 1:length(BHV.TrialError)
    words_per_trials(x).words = Words(starts(x):ends(x));
    words_per_trials(x).timestamps = Timestamp_words(starts(x):ends(x));
end

% Make sure no words in the BHV file got timestamped at 0 msec since analog sampling of that trial (by-product of
% rounding)
timestamp_bug = 0;
for x = 1:length(BHV.TrialError)
    if BHV.CodeTimes{1,x}(1) == 0
        BHV.CodeTimes{1,x}(1) = 1;
        disp(['Trial # ' num2str(x) ' has the first word (9) timestamped at 0 msec. Modified to 1 to avoid later indexation bugging'])
        timestamp_bug = 1;
    end
end
if timestamp_bug == 1
    disp('Resaving BHV file...')
    cd(PathName)
    save(FileName{2}, 'BHV')
    disp('Resaving done')
end


%% WAS THE TTL CORRECTLY SENT by ML AND RECEIVED BY THE EDF FILE

% 0. Transform the ASC file such that it can be read by matlab function "textread":
% 1. replace "..." by "[space]"
% 2. replace ".0" by ",0"; Do this for all numbers 1-9
% 3. replace "." by "0"
% 4. replace back ",0" by ".0"; do this for all numbers 1-9
% 5. replace "C" by "3" and "M" by "13". Do the same for all eyelink remote mode letter codes

if ifeyedata == 1
    cd(PathName)
    disp(' '); disp('Select text file containing Eyelink eye data');
    [edf_file, path_EDF] = uigetfile('*.asc','Select text file containing Eyelink eye data');
    if exist(['EDF_' session_name '.txt'], 'file') || exist([session_name '.txt'], 'file') % If you already created that big corrected txt file
    else
        fid1 = fopen([path_EDF, edf_file],'r');
        fid2 = fopen([path_EDF,'EDF_' edf_file(1:end-4),'.txt'],'w');
        
        while ~feof(fid1) %Until all lines of this file are read
            tline=fgetl(fid1); % Read the line
            newline = strrep(tline, '...', ' '); % Replace the string '...' with a space ' '
            newline2 = strrep(newline, '.0', ',0'); % Replace the string '.0' with ',0'
            newline3 = strrep(newline2, '.1', ',1');
            newline4 = strrep(newline3, '.2', ',2');
            newline5 = strrep(newline4, '.3', ',3');
            newline6 = strrep(newline5, '.4', ',4');
            newline7 = strrep(newline6, '.5', ',5');
            newline8 = strrep(newline7, '.6', ',6');
            newline9 = strrep(newline8, '.7', ',7');
            newline10 = strrep(newline9, '.8', ',8');
            newline11 = strrep(newline10, '.9', ',9');
            newline12 = strrep(newline11, '.', '0');
            newline13 = strrep(newline12, ',0', '.0');
            newline14 = strrep(newline13, ',1', '.1');
            newline15 = strrep(newline14, ',2', '.2');
            newline16 = strrep(newline15, ',3', '.3');
            newline17 = strrep(newline16, ',4', '.4');
            newline18 = strrep(newline17, ',5', '.5');
            newline19 = strrep(newline18, ',6', '.6');
            newline20 = strrep(newline19, ',7', '.7');
            newline21 = strrep(newline20, ',8', '.8');
            newline22 = strrep(newline21, ',9', '.9');
            newline23 = strrep(newline22, 'C', '3');
            newline24 = strrep(newline23, 'M', '13');
            newline25 = strrep(newline24, 'A', '1');
            newline26 = strrep(newline25, 'L', '12');
            newline27 = strrep(newline26, 'N', '14');
            newline28 = strrep(newline27, 'F', '6');
            newline29 = strrep(newline28, 'T', '20');
            newline30 = strrep(newline29, 'B', '2');
            newline31 = strrep(newline30, 'R', '18');
            newline32 = strrep(newline31, 'I', '9');
            fprintf(fid2, [newline32, '\r\n']); % Write the result to output file
        end
        
        fclose(fid1); % Close input file
        fclose(fid2); % CLose output file
        
    end
    
    % 2. Once all these changes have been made you can import the text file into Matlab:
    if exist([session_name '.txt'], 'file')
        [Eye_Data] = textread([session_name '.txt'],'','headerlines',9);
    else
        [Eye_Data] = textread(['EDF_' session_name '.txt'],'','headerlines',9);
    end
    if strcmp(session_name, 'B090917')
    load('Eye_data_fixed.mat') % This EDF file was missing a few lines from trial 184 to 190. I copied lines from previous trials to fill the gap. The TTL are fixed lower.
    end
 
    disp(' '); disp(['Loaded ' path_EDF,edf_file(1:end-4),'.txt']); disp(' ')
    
    % 3. Identify the TTL ON timestamp in both BR and EDF data streams EyeLink TTL ON times
    TTL_ON = diff(Eye_Data(:, 5)); % find all the indices where there was a change in the TTL signal.
    EDF_TTL_times = Eye_Data(find(TTL_ON == 1) +1, 1); % find the timestamps of the TTL ON in EDF file. +1 because of how the "diff" function works.
    if strcmp(session_name, 'B090917')
    load('EDF_fixed.mat') % This EDF file was missing a few TTLs from trial 184 to 190. I traced them back on the BR file and corrected the EDF_TTL_times vector
    end
    % Blackrock TTL ON times
    BR_TTL_times = round(Timestamp_words(Words == 23)*1000); % find the timestamps of TTL ON in Blackrock computer. *1000 to convert sec into milliseconds. Round to have a millisecond resolution as in EDF file.
    
    % Sanity checks
    % Check if number of TTL in EDF and Blackrock are the same
    if length(EDF_TTL_times) == length(BR_TTL_times)
    else
        error('Uneven number of TTL eyedata in Blackrock and EDF files')
    end
    
    %Check the synchrony of TTL by comparing time elapsed between first and last one of the session.
    if (EDF_TTL_times(end) - EDF_TTL_times(1)) - (BR_TTL_times(end) - BR_TTL_times(1)) <= 15 %msec
    else
        error('Asynchrony between TTL eyedata in Blackrock and EDF files')
    end
    
    % If the mean asynchrony between consecutive TTLs is more than 2 msec
    if  mean(abs((diff(EDF_TTL_times) - diff(BR_TTL_times)))) < 2
    else
        error('Asynchrony between consecutive TTL eyedata in Blackrock and EDF files')
    end
    
    clearvars -except AnalogData words_per_trials SpikeData Timestamp_words Words BHV Eye_Data BR_TTL_times EDF_TTL_times PathName FileName omit_correction units_to_discard session_name
    
    %% Organize the head data per trial
    
    % 1. Find the corresponding EDF timestamps for start and end words for each trial
    BR_trialstarts = round(Timestamp_words(Words == 9)*1000); %find the Blackrock timestamps of "start trial" words
    EDF_trialstarts = EDF_TTL_times - (BR_TTL_times - BR_trialstarts); %From the EDF TTL times, look back the time interval [9-23] to get EDf starts
    BR_ends = round(Timestamp_words(Words == 18)*1000); % Same as above but for end of trial.
    EDF_endtrial = EDF_TTL_times + (BR_ends - BR_TTL_times); %From the EDF TTL times, look forward the time interval [23-18] to get EDf ends
    
    % Sanity check
    if length(EDF_trialstarts) == length(EDF_endtrial)
    else
        error('Uneven number of Start and End of trial words in EDF file')
    end
    
    % 2. Make sure that all EDF timestamp equivalents are even because eyelink's sampling rate is 500Hz, or one sample every 2ms.
    for c = 1:length(EDF_trialstarts)
        if mod(EDF_trialstarts(c), 2) == 0 % If timestamp is even (the reminder of a division by 2 will be 0), leave as is
            EDF_trialstarts(c) = EDF_trialstarts(c);
        else
            EDF_trialstarts(c) = EDF_trialstarts(c) - 1; % If timestamp is odd (the reminder of a division by 2 will be non-0), deduct 1 to make it even.
        end
    end
    
    for d = 1:length(EDF_endtrial) % Same process here as above.
        if mod(EDF_endtrial(d), 2) == 0
            EDF_endtrial(d) = EDF_endtrial(d);
        else
            EDF_endtrial(d) = EDF_endtrial(d) - 1;
        end
    end
    
    % 3. Find the indices of the EDF start timestamps
    for a = 1:length(EDF_trialstarts)
        EDF_starts_indices(a) = find(Eye_Data(:,1) == EDF_trialstarts(a));
    end
    
    for b = 1:length(EDF_endtrial)
        EDF_ends_indices(b) = find(Eye_Data(:,1) == EDF_endtrial(b));
    end
    
    % 4. Organize head x&y positions per trial
    for i = 1:length(EDF_trialstarts)
        Head_per_trial(i).Xpos = Eye_Data((EDF_starts_indices(i):EDF_ends_indices(i)),6);
        Head_per_trial(i).Ypos = Eye_Data((EDF_starts_indices(i):EDF_ends_indices(i)),7);
        Head_per_trial(i).Zpos = Eye_Data((EDF_starts_indices(i):EDF_ends_indices(i)),8);
    end
    
    % 5. Go from a 2ms resolution to a 1ms resolution to facilitate
    % analysis later on (as everything is to the milisecond precision)
    for i = 1: length(EDF_trialstarts) %for each trial
        k=0;
        for j = 1:length(Head_per_trial(i).Xpos) % for each sample
            k = k+2;
            head_position{1,i}(k,1) = Head_per_trial(i).Xpos(j);
            head_position{1,i}(k-1,1) = Head_per_trial(i).Xpos(j);
            
            head_position{1,i}(k,2) = Head_per_trial(i).Ypos(j);
            head_position{1,i}(k-1,2) = Head_per_trial(i).Ypos(j);
            
            head_position{1,i}(k,3) = Head_per_trial(i).Zpos(j);
            head_position{1,i}(k-1,3) = Head_per_trial(i).Zpos(j);
        end
    end
    
    clearvars -except AnalogData words_per_trials SpikeData Timestamp_words Words BHV head_position PathName FileName omit_correction units_to_discard session_name
    
    %%  EyeLink Performance: assess how many valid eye data points are collected in this head-free paradigm.
    
    % 1. Overall Session
    
    % Assuming that any datapoint whose x < -4.8 and/or whose y < -4.8 is "missed", because Eyelink returns -5V,-5V for
    % each missed eye sample
    Xerror_value = -4.8;
    Yerror_value = -4.8;
    
    for trial= 1:length(BHV.TrialError) %For each trial
        total_datapoints(trial) = length(BHV.AnalogData{1, trial}.EyeSignal); % Find the total amount of eye datapoints during the trial.
        missed_datapoints(trial) = length(find(BHV.AnalogData{1,(trial)}.EyeSignal(:,1) <= Xerror_value & BHV.AnalogData{1,(trial)}.EyeSignal(:,2) <= Yerror_value));% Find the amount of missed eye datapoints during the trial.
        eyelink_performance(trial) = 1 - missed_datapoints(trial)/total_datapoints(trial); %Find what the eyelink tracking performance is during the trial.
    end
    
    mean_performance_overall = mean(eyelink_performance);
    
    % 2. During the Delay (between words 14-25)
    
    non_ignored = find(BHV.TrialError ~= 8); %Find non_ignored trials
    
    for i = 1:length(non_ignored) %For all non_ignored trial
        delay_period_indices = [BHV.CodeTimes{1, non_ignored(i)}(BHV.CodeNumbers{1, non_ignored(i)}==14) BHV.CodeTimes{1, non_ignored(i)}(BHV.CodeNumbers{1, non_ignored(i)}==25)]; %Find the delay period indices limits in the eye data, knowing that there is one eye sample per msec with 1KHz sampling rate.
        eye_delay = [BHV.AnalogData{1, non_ignored(i)}.EyeSignal(delay_period_indices(1):delay_period_indices(2),1), BHV.AnalogData{1, non_ignored(i)}.EyeSignal(delay_period_indices(1):delay_period_indices(2),2)]; %find eye within limits
        
        total_delay_data = length(eye_delay); % Find the total amount of eye datapoints during the delay period.
        missed_data_delay = length(find(eye_delay(:,1) <= Xerror_value & eye_delay(:,2) <= Yerror_value)); % Find the amount of missed eye datapoints during the delay period.
        eyelink_performance_delay(i) = 1 - missed_data_delay/total_delay_data; %Find what the eyelink tracking performance is during the delay period.
    end
    
    mean_performance_delay = mean(eyelink_performance_delay);
    
    if mean_performance_delay < 0.5
        disp(' '); warning(['Only ' num2str(mean_performance_delay*100) '% of eye tracking is valid during the delay period.'])
    else
    end
    
    %% Assess if eye data from BR .ns2 and from ML are similar
    
    %Load continuous data from .ns2
    load([PathName FileName{1}])
    
    %Asses the absolute difference between the eye data curves from BR and ML
    for i = 1:length(BHV.TrialError) %For each trial
        
        BReyeX = AnalogData.Channel_97(round(words_per_trials(i).timestamps(1)*1000):round(words_per_trials(i).timestamps(end)*1000));
        BHVeyeX = BHV.AnalogData{1,i}.EyeSignal(BHV.CodeTimes{1,i}(1):BHV.CodeTimes{1,i}(end),1)*1000;
        BReyeY = AnalogData.Channel_98(round(words_per_trials(i).timestamps(1)*1000):round(words_per_trials(i).timestamps(end)*1000));
        BHVeyeY = BHV.AnalogData{1,i}.EyeSignal(BHV.CodeTimes{1,i}(1):BHV.CodeTimes{1,i}(end),2)*1000;
        if length(BReyeY) ~= length(BHVeyeY)
            for a = 1:(length(BHVeyeY)-length(BReyeY))
                BReyeY(length(BReyeY)+1)=0;
                BReyeX(length(BReyeX)+1)=0;
            end
        end
        
        %Correct the fact that sometimes the Blackrock eye vector has one more sample point. Delete the first sample in this vector.
        if length(BReyeX) == length(BHVeyeX)+1
            BReyeX(1) = [];
        end
        if length(BReyeY) == length(BHVeyeY)+1
            BReyeY(1) = [];
        end
        if length(BReyeX) ~= length(BHVeyeX) || length(BReyeY) ~= length(BHVeyeY)
            error('The Blackrock and BHV eye vectors dont have the same length for this trial')
        end
        
        if sum(abs(BReyeX - BHVeyeX))/length(BHVeyeX) > 500  % if the mean difference between the two curves is more than 500mv, output an error message.
            error('The eyeX data is considered too different between the BHV and BR files for this trial.')
        else
        end
        
        if sum(abs(BReyeY - BHVeyeY))/length(BHVeyeY) > 500  % if the mean difference between the two curves is more than 500mv, output an error message.
            error('The eyeY data is considered too different between the BHV and BR files for this trial.')
        else
        end
    end
    
    
end % End of eye data sanity checking

% Double check if some channels contain an "invalid spikes" unit as a byproduct of offline sorting.
for i = 1:length(fields(SpikeData))
    for j = 2:length(SpikeData.(['Channel_' num2str(i)]))
        if length(SpikeData.(['Channel_' num2str(i)]){j}) < 1000
            disp(['Channel ' num2str(i) ' contains a unit with very few spikes'])
        end
    end
end

%Count total number of neurons
neuron_count = structfun(@numel,SpikeData)-1; neuron_count(neuron_count < 0) = 0;
total_neuron = sum(neuron_count);
disp(' '); disp(['This nev file contains ' num2str(total_neuron) ' units. See if this fits with the count in Offline Sorter.']); disp(' ')

clearvars -except SpikeData Timestamp_words Words BHV  head_position words_per_trials omit_correction units_to_discard session_name

disp(['End of data sanity checking in ' num2str(ceil(toc)) ' seconds. Status: *** Passed ***']); disp(' ')

return

%% End of script.

