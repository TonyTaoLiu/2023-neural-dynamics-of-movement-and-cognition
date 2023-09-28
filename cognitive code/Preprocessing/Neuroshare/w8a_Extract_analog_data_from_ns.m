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
cd('D:\Projects\Wireless_8A\Coding\Neuroshare')
% Load the appropriate DLL
[nsresult] = ns_SetLibrary(DLLName);
if (nsresult ~= 0)
    disp('DLL was not found!');
    return
end

% Find out the data file from user SdT : your neural file (i.e. .nev)
cd('D:\Projects\Wireless_8A\Neural_data');  % Blank line
[filename, pathname] = uigetfile({'*.plx; *.ns2'});
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
end
if (cAnalog == 0)
    disp('No analog entities available!');
end
if (cEvent == 0)
    disp('No event entities available!');
end

disp(' ');
disp(['There are ' num2str(cAnalog) ' analog channels.']);
disp(' ');

for cChannel = 1 : length(AnalogList) %for all channels
    [nsresult, ~, data] = ns_GetAnalogData(hfile, AnalogList(cChannel), 1, EntityInfo(AnalogList(cChannel)).ItemCount);
    AnalogData.(['Channel_' num2str(cChannel)]) = data;
    data = [];
end

% Close data file.
ns_CloseFile(hfile);

% Unload DLL
clear mexprog;


disp(' ')
disp(['Successfully completed in ' num2str(toc) ' seconds'])

%SdT : End of neural data extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


