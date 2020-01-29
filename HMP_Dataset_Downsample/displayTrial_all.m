function displayTrial_all(folder)
% function displayModel(folder)
%
% displayModel plots the acceleration recorded in the modelling trials
% stored in the given folder. Acceleration data are decoded and filtered
% with median filtering.
%
% Input:
%   folder --> name of the folder containing the dataset to be displayed
%
% Output:
%   ---
%
% Example:
%   folder = 'Climb_stairs_MODEL/';
%   displayModel(folder);


% READ THE ACCELEROMETER DATA FILES
files = dir([folder,'*.txt']);

numFiles = length(files);
dataFiles = zeros(1,numFiles);
disp(numFiles);


for i=1:1:numFiles
    disp("Iteraciones i");
    disp(i);
    dataFiles(i) = fopen([folder files(i).name],'r');
    textLine = fgets(dataFiles(i));
    lineCounter = 1;
        while ischar(textLine)

        data = fscanf(dataFiles(i),'%d\t%d\t%d\n',[3,inf]);

        %%%%%%%%%%%%%%%%%%%%%%
        numSamples = length(data);
        disp("Numsamples");
        disp(numSamples);
            for k = 1 : length(numFiles)
              disp("Iteraciones k");
              disp(k);
              disp(data(1,:));
              noisy_x(:,i) = -14.709 + (data(i,:)/63)*(2*14.709);
              disp(noisy_x);
            end

            textLine = fgets(dataFiles(i));
            lineCounter = lineCounter + 1;

        % CONVERT THE ACCELEROMETER DATA INTO REAL ACCELERATION VALUES
        % mapping from [0..63] to [-14.709..+14.709]

        end
    
end

% REDUCE THE NOISE ON THE SIGNALS BY MEDIAN FILTERING
n = 3;      % order of the median filter
x_set = medfilt1(noisy_x,n);

numSamples = length(x_set(:,1));
disp(numSamples)

% DISPLAY THE RESULTS
time = 1:1:numSamples;
% noisy signal
figure,
    subplot(3,1,1);
    plot(time,noisy_x,'-');
    axis([0 numSamples -14.709 +14.709]);
    title('Noisy accelerations along the x axis');
    
% clean signal
figure,
    subplot(3,1,1);
    plot(time,x_set,'-');
    axis([0 numSamples -14.709 +14.709]);
    title('Filtered accelerations along the x axis');

    