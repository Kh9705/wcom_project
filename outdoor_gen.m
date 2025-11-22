% =================================================
% OUTDOOR 5G DATA GENERATOR (Hong Kong Scenario)
% =================================================

% 1. Setup the Environment (Fixed Base Station)
fc = 6e9; 
bsPosition = [22.287495, 114.140706]; % Base Station (Lat, Lon)
bsHeight = 40; % Put BS high up (e.g., 40m) to see over some buildings
% Note: Original example had 4m, but 40m ensures better rays for 1000 samples

% Create Site Viewer (Hidden for speed)
% viewer = siteviewer("Basemap","openstreetmap","Buildings","hongkong.osm");

% Define Propagation Model (SBR)
pm = propagationModel("raytracing","Method","sbr", ...
    "MaxNumReflections",1, "MaxNumDiffractions",0);

bsSite = txsite("Name","Base station", ...
    "Latitude",bsPosition(1),"Longitude",bsPosition(2),...
    "AntennaHeight",bsHeight, "TransmitterFrequency",fc);

% 2. Generate 1000 Samples
numSamples = 1000;
dataMatrix = zeros(numSamples, 3); % Lat, Lon, PathLoss

disp('Starting Outdoor Ray Tracing Loop...');

for i = 1:numSamples
    % Randomize UE Position (Small area around the BS)
    % +/- 0.001 degrees is roughly +/- 100 meters
    randLat = bsPosition(1) + (rand - 0.5) * 0.002; 
    randLon = bsPosition(2) + (rand - 0.5) * 0.002;
    
    % Create UE Site
    ueSite = rxsite("Name","UE", ...
        "Latitude",randLat, "Longitude",randLon, ...
        "AntennaHeight",1); % Standard mobile height
    
    % Run Ray Trace
    % We wrap in try-catch because some points might be INSIDE buildings (fail)
    try
        rays = raytrace(bsSite, ueSite, pm, "Type", "pathloss");
        
        if ~isempty(rays{1})
            % Extract the strongest path (Min Path Loss)
            pl = min([rays{1}.PathLoss]);
            dataMatrix(i, :) = [randLat, randLon, pl];
        else
            dataMatrix(i, :) = [randLat, randLon, NaN];
        end
    catch
        dataMatrix(i, :) = [randLat, randLon, NaN];
    end
    
    if mod(i, 50) == 0
        fprintf('Processed %d / %d\n', i, numSamples);
    end
end

% 3. Clean and Save
% Remove NaNs (Points inside buildings or unreachable)
dataMatrix = dataMatrix(~isnan(dataMatrix(:,3)), :);

T = array2table(dataMatrix, 'VariableNames', {'Latitude', 'Longitude', 'PathLoss'});
writetable(T, 'outdoor_ray_tracing_data.csv');
disp(['Data generation complete. Saved ', num2str(height(T)), ' valid samples.']);