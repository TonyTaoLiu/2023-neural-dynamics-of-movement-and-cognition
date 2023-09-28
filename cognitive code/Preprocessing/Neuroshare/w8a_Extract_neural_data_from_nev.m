%w8a_Extract_neural_data
%Reads in a spike sorted .nev or .plx file and extracts relevant
%information about spike timing using Neuroshare.

%% Load basic information about neural file
% Prompt for the correct DLL SdT : choose the right library according to
% the file type you want to analyze (.nev, .plx)
clear all; clc
disp(' ');  % Blank line
Library = 2; %input('Type "1" for Plx, "2" for Nev: '); Selects .nev automically

if Library == 1
    DLLName = 'nsPlxLibrary64.dll';
elseif Library == 2
    DLLName = 'nsNEVLibrary64.dll';
else
    error('Wrong number')
end

cd('C:\Users\Julio\Google Drive\Coding\Neuroshare')
% Load the appropriate DLL
[nsresult] = ns_SetLibrary(DLLName);
if (nsresult ~= 0)
    disp('DLL was not found!');
    return
end

% Find out the data file from user SdT : your neural file (i.e. .nev)
cd('D:\Sebastien\w8a');  % Blank line
[filename, pathname] = uigetfile({'*.plx; *.nev; *.ns2'});
cd(pathname)
tic

% Load data file and display some info about the file
% Open data file
[nsresult, hfile] = ns_OpenFile(filename);
if (nsresult ~= 0)
    disp('Data file did not open!');
    return
end

% Get file information
[nsresult, FileInfo] = ns_GetFileInfo(hfile);
% Gives you EntityCount, TimeStampResolution and TimeSpan and the time at
% which the file was created
if (nsresult ~= 0)
    disp('Data file information did not load!');
    return
end

% Build catalogue of entities SdT : Entities contains one or more indexed
% data entries that are ordered by increasing time. There are four types of
% entities, listed below. Each entity type contains specific information
% about your neural data.
[nsresult, EntityInfo] = ns_GetEntityInfo(hfile, [1 : FileInfo.EntityCount]);

% List of EntityIDs needed to retrieve the information and data
NeuralList = find([EntityInfo.EntityType] == 4); %Neural entity: Timestamp of spikes. One entity per unit including the unsorted cluster.
SegmentList = find([EntityInfo.EntityType] == 3); %Segment entity: Waveform of each spike. One per channel (128 channels + 16 analog ins).
AnalogList = find([EntityInfo.EntityType] == 2); %Analog entity: Digitized analog signal, such as LFP or eye positions
EventList = find([EntityInfo.EntityType] == 1); %Event entity: Digital input such as words or other binary events.

% How many of a particular entity do we have
cNeural = length(NeuralList);
cSegment = length(SegmentList);
cAnalog = length(AnalogList);
cEvent = length(EventList);

if (cNeural == 0)
    disp('No neural events available!');
end
if (cSegment == 0)
    disp('No segment entities available!');
    return;     % It does not make sense to continue in this particular analysis
    % if there are no segment entities.
end
if (cAnalog == 0)
    disp('No analog entities available!'); %SdT : I beleive you should only expect analog entities in your .ns files, not your .nev
end
if (cEvent == 0)
    disp('No event entities available!');
end

%% Extract Event entities
% Get some event identity info (i.e. Digital input, Serial input/output
% Define Entity of interest
EntityID = EventList(1); %digital input of cerebus
%Get info
[nsresult, nsEventInfo] = ns_GetEventInfo(hfile, EntityID);

%Define index to look at
Index =  EntityInfo(EventList(1)).ItemCount;
%Retrieve event data by index
[nsresult, Timestamp_words, Words, DataSize] = ns_GetEventData(hfile, EntityID, 1:Index);


%% Extract data on waveforms (segment entity)
channel = 96; % Number of electrodes on the array
% 
%Extract information about sampling rates and filters of channels
[nsresult, nsSegmentInfo] = ns_GetSegmentInfo(hfile, SegmentList(1:channel));
[nsresult, nsSegmentSourceInfo] = ns_GetSegmentSourceInfo(hfile, SegmentList(1:channel), 1);

%SdT : timestamps_wf is the time stamp of each waveform. Samplecount is the number of samples per waveform (48 for
%all of them. Waveform is the actual y values of each waveforms. unitIDs is the identifier of a sorted unit, 0 being noise or unsorted, 1to 5 being sorted ID
for i = 1:channel
    [nsresult, timestamps_wf, waveforms, sampleperwaveform, unitIDs] = ns_GetSegmentData(hfile, SegmentList(i), 1 : EntityInfo(SegmentList(i)).ItemCount);
    Waveforms(i).sampleperwaveform = sampleperwaveform; 
    Waveforms(i).timestamps_wf = timestamps_wf; 
    Waveforms(i).unitIDs = unitIDs;
    Waveforms(i).waveshape = waveforms;
end

%Compute mean waveform
count = 1;
for chan = 1:channel
    for unit = 1:length(unique(Waveforms(chan).unitIDs))-1 %For each unit on channel except the unsorted cluster
        Waves(count).mean_waveform = mean(Waveforms(chan).waveshape(:,Waveforms(chan).unitIDs' == unit), 2);
        Waves(count).std_waveform = std(Waveforms(chan).waveshape(:,Waveforms(chan).unitIDs' == unit), 1, 2);        
        %figure;plot(mean_waveform);hold on; plot(mean_waveform + std_waveform); plot(mean_waveform - std_waveform)
        count = count + 1;
    end
end

%% Extract spike timings (Neural entity)

% Creates a list of labels of all the electrodes from which a Unit as been identified.
for i = 1:length(NeuralList)
    NeuralLabels{i} = EntityInfo(NeuralList(i)).EntityLabel;
end

%Loop through channels
for cChannel =   1:channel 
    
    % Find the Neural entities that belong to the current channel
    list = find(strcmp(EntityInfo(SegmentList(cChannel)).EntityLabel, NeuralLabels));

    clear NeuralData timestamps_units
    
    %Extract spike times for each unit on this channel
    for i = 1:length(list)
        [nsresult, NeuralData] = ns_GetNeuralData(hfile, NeuralList(list(i)), 1, [EntityInfo(NeuralList(list(i))).ItemCount]);
        timestamps_units{i} = NeuralData;
        
        if i >= 2 %For all units except the unsorted cluster
            if EntityInfo(NeuralList(list(i))).ItemCount < 2000 %If the unit has lower than 3000 spikes, mark it
            Low_spikes(cChannel) = 1;
            end
        end
    end
    
    SpikeData.(['Channel_' num2str(cChannel)]) = timestamps_units;
    
end
disp(' ')
if exist('Low_spikes')
disp(['WARNING! You have very low number of spikes on channels # ' num2str(find(Low_spikes)) '. Check for artefacts!'] )
end
 
% Saving
%save(filename(1:end-4), 'SpikeData', 'Words', 'Timestamp_words') %save Spikes and Words
%save([filename(1:end-16) 'Waveforms'], 'Waveforms') %save waveforms
 
% Close data file. 
ns_CloseFile(hfile);

% Unload DLL
clear mexprog;
clearvars -except Words SpikeData Timestamp_words Waves

disp(' ')
disp(['Successfully completed in ' num2str(toc) ' seconds'])

%SdT : End of neural data extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


